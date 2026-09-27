"""Regression checks for geometry truncation and decoder disagreement."""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch


class Stub(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.kind = kind

    def forward(self, x, *args, **kwargs):
        b = len(x)
        if self.kind == 'poly':
            # Short native vertical spans; weak confidence must not erase geometry.
            lane = [float(np.log(.3/.7)), .25, .75, 0., 0., .05, .2]
            return torch.tensor([lane, lane]).reshape(1, -1).repeat(b, 1), None
        if self.kind == 'scnn':
            logits = torch.full((b, 3, 40, 40), -8.)
            logits[:, 0] = 0.
            logits[:, 1, :, 3:16] = 8.
            logits[:, 2, :, 24:37] = 8.
            return logits, torch.tensor([[.3, .9]]).repeat(b, 1), None, None, None
        if self.kind == 'ufld':
            loc = torch.full((b, 20, 12, 2), -5.)
            loc[:, 4, :, 0] = 5.
            loc[:, 15, :, 1] = 5.
            exist = torch.zeros((b, 2, 12, 2))
            exist[:, 0] = 2.
            exist[:, 1, 3:9] = 6.
            return dict(loc_row=loc, exist_row=exist)
        if self.kind == 'laneatt':
            proposal = torch.zeros((1, 5+72))
            proposal[0, :5] = torch.tensor([0., 2., .4, 500., 12.])
            proposal[0, 5:] = 500.
            return [(proposal, None, None, None) for _ in range(b)]
        raise AssertionError(self.kind)


class FullHeightDecoding(unittest.TestCase):
    def test_checkpoint_presence_rule_is_explicit(self):
        from lcms_lane_benchmark.literature._val_strict_iou import _selection_presence
        output=dict(pred_exists_L=True, pred_exists_R=False,
                    pred_conf_L=.2, pred_conf_R=.8)
        self.assertEqual(_selection_presence(output,'native'),(True,False))
        self.assertEqual(_selection_presence(output,'confidence'),(False,True))
        with self.assertRaises(ValueError):_selection_presence(output,'unknown')

    def test_polynomial_span_and_confidence_do_not_truncate_geometry(self):
        p = importlib.import_module('lcms_lane_benchmark.literature.polylanenet_faithful.predict')
        model = Stub('poly')
        image = np.zeros((101, 81), dtype=np.uint8)
        batch = p.predict_batch_with_model(model, [image, image], img_h=8, img_w=8)
        single = p.predict_with_model(model, image, img_h=8, img_w=8)
        adapter = p.PolyLaneNetFaithfulMethod(device='cpu', img_h=8, img_w=8)
        adapter._model = model
        direct = adapter.predict_image(image)
        expected = (0.2 + .05 * np.arange(101)/100)*80
        for result in batch + [single, direct]:
            np.testing.assert_allclose(result['pred_x_L'], expected, atol=1e-5)
            self.assertFalse(result['pred_exists_L'])
            self.assertAlmostEqual(result['pred_conf_L'], .3, places=6)

    def test_scnn_low_confidence_keeps_available_curve(self):
        p = importlib.import_module('lcms_lane_benchmark.literature.scnn_faithful.predict')
        model = Stub('scnn')
        image = np.zeros((101, 81), dtype=np.uint8)
        batch = p.predict_batch_with_model(model, [image, image], input_h=40, input_w=40)
        single = p.predict_with_model(model, image, input_h=40, input_w=40)
        adapter = p.SCNNFaithfulMethod(device='cpu', input_h=40, input_w=40)
        adapter._model = model
        direct = adapter.predict_image(image)
        for result in batch + [single, direct]:
            self.assertFalse(result['pred_exists_L'])
            self.assertTrue(np.isfinite(result['pred_x_L']).all())
            np.testing.assert_allclose(result['pred_x_L'], single['pred_x_L'])
            self.assertAlmostEqual(result['pred_conf_L'], .3, places=6)

    def test_unet_low_support_preserves_fittable_curve(self):
        p = importlib.import_module('lcms_lane_benchmark.literature.unet_seg.predict')
        prob = np.zeros((200, 40), dtype=np.float32)
        prob[50:64, 12:27] = 1.
        curve, exists, support = p._decode_side(prob, 301, 81, .3, .5, 5)
        self.assertFalse(exists)
        self.assertLess(support, .5)
        self.assertTrue(np.isfinite(curve).all())

    def test_short_anchor_support_produces_full_height_curves(self):
        lane = importlib.import_module('lcms_lane_benchmark.literature.laneatt_faithful.predict')
        clr = importlib.import_module('lcms_lane_benchmark.literature.clrnet_faithful.predict')
        n = 72
        prop = np.zeros(5+n)
        prop[2], prop[4], prop[5:] = .4, 12, 300.
        curve = lane._proposal_to_full_height_curve(prop, n, 360, 640, 301, 1001)
        self.assertTrue(np.isfinite(curve).all())
        np.testing.assert_allclose(curve, 300*1000/639, atol=1e-7)
        prop2 = np.zeros(6+n)
        prop2[2], prop2[5], prop2[6:] = .4, 12, .7
        curve2 = clr._proposal_to_full_height_curve(prop2, np.linspace(1, 0, n), 800, n-1, 301, 1001)
        np.testing.assert_allclose(curve2, 700., atol=1e-7)
        prop[2] = -10  # Invalid starts must not index from the end or raise.
        lane._proposal_to_full_height_curve(prop, n, 360, 640, 301, 1001)

    def test_ufld_and_laneatt_batch_single_class_agree(self):
        image = np.zeros((301, 1001), dtype=np.uint8)
        for kind, module, cls, kw in [
            ('ufld', 'ufldv2_faithful', 'UFLDv2FaithfulMethod',
             dict(input_h=40, input_w=40, num_cls_row=12, num_cell_row=20)),
            ('laneatt', 'laneatt_faithful', 'LaneATTFaithfulMethod', dict(img_h=360, img_w=640)),
        ]:
            p = importlib.import_module(f'lcms_lane_benchmark.literature.{module}.predict')
            model = Stub(kind)
            batched = p.predict_batch_with_model(model, [image, image], **kw)
            single = p.predict_with_model(model, image, **kw)
            adapter = getattr(p, cls)(device='cpu', **kw)
            adapter._model = model
            direct = adapter.predict_image(image)
            for result in batched + [direct]:
                for side in ['L','R']:
                    np.testing.assert_allclose(result['pred_x_'+side],single['pred_x_'+side], equal_nan=True)
                    self.assertEqual(result['pred_exists_'+side],single['pred_exists_'+side])
                    self.assertEqual(result['pred_conf_'+side],single['pred_conf_'+side])
                    if result['pred_exists_'+side]:
                        self.assertTrue(np.isfinite(result['pred_x_'+side]).all())


if __name__ == '__main__':
    unittest.main()
