"""Run historical synthetic checks; never load a real image/checkpoint."""
import argparse
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decoders',action='store_true',help='Also exercise synthetic Torch decoder stubs (requires model dependencies)')
    a=p.parse_args()
    count=0
    for name in ['lcms_lane_benchmark.data.test_lane_gt','lcms_lane_benchmark.evaluation.test_gra']:
        module=importlib.import_module(name)
        for key in sorted(vars(module)):
            if key.startswith('test_') and callable(getattr(module,key)):
                getattr(module,key)();count+=1
    from lcms_lane_benchmark.scripts.check_wheelpath_geometry import main as wheelpath
    wheelpath()
    print('PASS:',count,'ground-truth/GRA checks')
    if a.decoders:
        suite=unittest.defaultTestLoader.loadTestsFromName('lcms_lane_benchmark.evaluation.test_full_height_decoding')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
