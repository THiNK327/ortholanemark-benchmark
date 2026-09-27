"""Synthetic tests of existing evaluator/decoder conventions; no model inference."""
import sys
sys.dont_write_bytecode = True
import ast
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcms_lane_benchmark.evaluation import metrics as m
from lcms_lane_benchmark.data.lane_gt import LaneGeometricTruth

def extract(path, names):
    text=(ROOT/path).read_text(encoding='utf-8-sig')
    tree=ast.parse(text)
    tree.body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace=dict(np=np, cv2=cv2, MM_PER_PIXEL=4.0)
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace

w=extract('lcms_lane_benchmark/scripts/wheelpath_diagnostic.py',['strips','disagreement'])
lane=extract('lcms_lane_benchmark/literature/laneatt_faithful/predict.py',['_proposal_to_full_height_curve'])['_proposal_to_full_height_curve']
clr=extract('lcms_lane_benchmark/literature/clrnet_faithful/predict.py',['_proposal_to_full_height_curve'])['_proposal_to_full_height_curve']
unet=extract('lcms_lane_benchmark/literature/unet_seg/predict.py',['_decode_side'])['_decode_side']
results=[]
def check(name,fn):
    try:
        fn()
        results.append(dict(name=name,status='pass'))
    except Exception as e:
        results.append(dict(name=name,status='fail',error=repr(e)))
def eq(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
def gt(x):
    return LaneGeometricTruth(True,np.array([0.0,x]),1)

def absent_full_image():
    eq(m.operational_region_iou(None,False,None,False,None,None,3,8),1)
    eq(m.gated_region_accuracy(1,False,False,None,None),1)
    eq(m.gated_region_accuracy(1,True,False,None,None),0)
check('correct_absence_full_image_GRA_one_and_presence_mismatch_zero',absent_full_image)

def closure():
    l,r=m._operational_xs(np.array([np.nan,np.inf,-np.inf,-8,99]),True,np.array([np.nan,np.inf,-np.inf,-8,99]),True,5,8)
    eq(l,[0,0,0,0,7]); eq(r,[7,7,7,0,7])
check('nonfinite_prediction_corresponding_edge_and_finite_coordinate_clipping',closure)

def pixel_edges():
    r=m._rasterize_region(np.array([2.1,2,4,2.5]),np.array([4.8,2,2,2.5]),4,8)
    eq(r.sum(1),[2,1,0,0])
check('inclusive_integer_pixels_reversed_empty_equal_integer_one_pixel',pixel_edges)

def empty_union():
    eq(m.operational_region_iou(np.array([6.]),True,np.array([2.]),True,gt(6),gt(2),1,8),1)
    eq(w['disagreement'](np.array([6.]),np.array([2.]),np.array([6.]),np.array([2.]),8),0)
check('two_empty_regions_IoU_one_wheelpath_disagreement_zero',empty_union)

def wheel_equal():
    lo,hi=w['strips'](np.array([520.0]),1040,np.array([True]))
    eq(np.maximum(hi-lo+1,0).sum(),500)
    eq(w['disagreement'](np.array([520.]),np.array([520.]),np.array([520.]),np.array([520.]),1040),0)
check('wheelpath_equal_boundaries_valid_strips_instead_of_lane_one_pixel',wheel_equal)

def malformed_gt():
    g=gt(float('nan'))
    eq(m._operational_xs_from_gt(g,None,2,8)[0],[np.nan,np.nan])
    eq(m.operational_region_iou(None,False,None,False,g,None,2,8),0)
check('GT_NaN_coefficients_are_not_prediction_style_sanitized',malformed_gt)

def tol_and_missing():
    x=np.array([2.5,6.25,12.5,np.nan])
    val=m.within_sigma_fractions(x,True,None,False,gt(0),None,4,100)
    eq([val[10.0],val[25.0],val[50.0]],[0,1/3,2/3])
    eq(m.boundary_mae_on_gt(x,gt(0),4,100),21.25/3)
check('strict_less_than_tolerances_and_finite_row_mask',tol_and_missing)

def aggregation():
    row=m.per_image_boundary_errors_full(np.array([1.,1.]),True,np.array([4.,np.nan]),True,gt(0),gt(2),2,10)
    eq(row['boundary_mae_on_gt_macro'],1.5)
    eq(row['within_10mm'],1)
    agg=m.aggregate_per_image([row,dict(row,boundary_mae_on_gt_macro=None)])
    eq(agg['boundary_mae_on_gt_macro'],1.5)
check('MAE_equal_side_weights_and_images_without_rows_excluded',aggregation)

def degree_rules():
    recorded=[]
    original=np.polyfit
    def spy(x,y,degree):
        recorded.append(degree)
        return original(x,y,degree)
    np.polyfit=spy
    try:
        for n in [1,2,4,5]:
            p=np.r_[0.,1.,0.,0.,float(n),np.full(6,40.)]
            c=lane(p,6,60,100,120,200)
            assert (c is None)==(n<2)
        for n in [1,2,4,5]:
            p=np.r_[0.,1.,0.,0.,0.,float(n),np.full(6,.4)]
            c=clr(p,np.linspace(1,0,6),100,5,120,200)
            assert (c is None)==(n<2)
    finally:
        np.polyfit=original
    eq(recorded,[1,1,2,1,1,2])
check('LaneATT_CLRNet_two_to_four_linear_five_quadratic_no_curve_under_two',degree_rules)

def segmentation():
    prob=np.zeros((60,30)); prob[:,8:17]=1
    curve,exists,support=unet(prob,120,100,.3,.1,5)
    assert exists and support==1 and np.isfinite(curve).all()
    curve,exists,support=unet(np.zeros((60,30)),120,100,.3,.1,5)
    assert not exists and support==0 and np.isnan(curve).all()
    # Literal shared score comparison: zero score is present at threshold zero.
    assert bool(support>=0.0) and not bool(support>=0.02)
check('segmentation_support_full_height_absence_and_zero_threshold_endpoint',segmentation)

def wheel_raster():
    rng=np.random.RandomState(20260924)
    for _ in range(30):
        x=rng.uniform(0,1039,(4,7))
        pl,pr,gl,gr=x
        a,b=w['strips']((pl+pr)/2,1040,pl<=pr)
        c,d=w['strips']((gl+gr)/2,1040,gl<=gr)
        cols=np.arange(1040)[None,:,None]
        pm=((cols>=a[:,None,:])&(cols<=b[:,None,:])).any(2)
        gm=((cols>=c[:,None,:])&(cols<=d[:,None,:])).any(2)
        union=(pm|gm).sum()
        expected=1-(pm&gm).sum()/union if union else 0
        eq(w['disagreement'](pl,pr,gl,gr,1040),expected)
check('wheelpath_interval_calculation_equals_independent_synthetic_raster',wheel_raster)

(ROOT/'verification').mkdir(exist_ok=True)
out=dict(scope='Synthetic software tests only; original source/results unchanged.',test_count=len(results),passed=sum(r['status']=='pass' for r in results),results=results)
(ROOT/'verification/edge_cases.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
assert out['passed']==out['test_count']
