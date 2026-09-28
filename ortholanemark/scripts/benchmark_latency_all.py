"""Fair same-harness latency: every method timed identically — warmup then
N reps of its compute path (preprocess+forward+adapter, excluding image
I/O) on one fixed image, median+p95 with cuda.synchronize. Replaces the
inflated per-image eval-loop numbers so 'CAR ≈2× faster' is defensible."""

from __future__ import annotations
import argparse, json, time
import cv2, numpy as np, torch


def time_method(fn, reps, warmup, cuda):
    ts = []
    for i in range(warmup + reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return float(np.median(a)), float(np.percentile(a, 95))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',
                   default='ortholanemark/clean_gt/v1/manifest_paper.json')
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    args = p.parse_args(argv)
    cuda = torch.cuda.is_available()

    man = json.load(open(args.manifest))
    img = cv2.imread(man['splits']['test'][0]['image_path'],
                     cv2.IMREAD_GRAYSCALE)

    rows = []
    # Registered learning methods (lazy-load their checkpoints)
    from ortholanemark.literature import build_method
    deep = ['clrnet', 'scnn', 'polylanenet',
            'laneatt', 'ufldv2', 'ufldv2_tusimple',
            'unet_seg']
    for name in deep:
        try:
            m = build_method(name)
            m.predict_image(img)              # trigger lazy load
            med, p95 = time_method(lambda: m.predict_image(img),
                                   args.reps, args.warmup, cuda)
            rows.append((name, med, p95))
        except Exception as ex:
            rows.append((name, float('nan'), float('nan')))
            print(f"  {name}: {ex}")

    # CAR (forward+WLS+adapter via its model)
    try:
        from ortholanemark.car.model import CAR
        ck = torch.load('ortholanemark/runs_car/car_d1_pooled/best.pth',
                        map_location='cuda' if cuda else 'cpu',
                        weights_only=False)
        c = ck['config']
        cm = CAR(c['n_anchor'], c['x_bins'], c['degree'],
                 c['existence_mode']).to('cuda' if cuda else 'cpu').eval()
        cm.load_state_dict(ck['model'])
        dev = 'cuda' if cuda else 'cpu'
        ay = torch.linspace(0, 1, c['n_anchor'], device=dev).unsqueeze(0)
        ey = torch.linspace(0, 1, img.shape[0], device=dev).unsqueeze(0)
        Wc = c['input_w']

        @torch.no_grad()
        def car_call():
            ir = cv2.resize(img, (c['input_w'], c['input_h']),
                            interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(np.stack([ir] * 3, 0).astype(np.float32)
                                 / 255.0).unsqueeze(0).to(dev)
            o = cm(x, ey, ay, W=float(Wc))
            _ = (o['pred_x_L'][0].cpu().numpy(),
                 torch.sigmoid(o['exist_logit_L'])[0].item())
        car_call()
        med, p95 = time_method(car_call, args.reps, args.warmup, cuda)
        rows.append(('CAR (D1, ours)', med, p95))
    except Exception as ex:
        print(f"  CAR: {ex}")

    rows.sort(key=lambda r: (np.isnan(r[1]), r[1]))
    print(f"\n=== Same-harness latency (fixed input, {args.reps} reps warmed,"
          f" {'GPU' if cuda else 'CPU'}) ===")
    print(f"{'method':<28}{'median ms':>11}{'p95 ms':>9}")
    for n, med, p95 in rows:
        print(f"{n:<28}{med:>11.1f}{p95:>9.1f}")


if __name__ == '__main__':
    main()
