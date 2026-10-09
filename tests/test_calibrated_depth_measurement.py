import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.calibrated_depth_measurement import MonotoneDepthCalibration, measure_axial_depth

class Checks(unittest.TestCase):
    def test_monotone_and_serialization(self):
        x=np.linspace(1,5,4096);g=60/x
        model=MonotoneDepthCalibration.fit(x,g,increasing=False)
        self.assertTrue((np.diff(model.predict(x))<=1e-12).all())
        saved=model.to_dict();restored=MonotoneDepthCalibration(np.array(saved['x']),np.array(saved['y']))
        np.testing.assert_array_equal(model.predict(x),restored.predict(x))
        self.assertLess(np.abs(model.predict(x)-g).mean(),.05)
    def test_axial_rays_and_scale(self):
        yy,xx=np.mgrid[-10:11,-20:21];rays=np.stack([xx/100,yy/100,np.ones_like(xx)],axis=-1)
        mask=np.ones(xx.shape,bool);depth=np.full(xx.shape,50.)
        a=measure_axial_depth(depth,mask,rays)
        b=measure_axial_depth(depth*2,mask,rays/np.linalg.norm(rays,axis=-1,keepdims=True))
        self.assertAlmostEqual(a['major_q01_q99_mm'],20.)
        self.assertAlmostEqual(b['major_q01_q99_mm'],40.)
    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):MonotoneDepthCalibration.fit(np.ones(100),np.full(100,np.nan),increasing=True)
        with self.assertRaises(ValueError):measure_axial_depth(np.ones((3,3)),np.ones((3,3)),np.ones((3,3,3)))
    def test_tied_calibration(self):
        m=MonotoneDepthCalibration.fit(np.ones(100),np.full(100,20.),increasing=False)
        np.testing.assert_array_equal(m.predict(np.array([0,1,2])),[20,20,20])
if __name__=='__main__':unittest.main()
