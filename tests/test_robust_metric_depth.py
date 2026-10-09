import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.robust_metric_depth import RobustMetricSpline,depth_feature,interpolation_basis
class RobustDepthTests(unittest.TestCase):
    def test_normalization_affine_invariance(self):
        x=np.arange(300.).reshape(15,20)
        np.testing.assert_allclose(depth_feature(x,'frame_quantile'),depth_feature(x*3+7,'frame_quantile'),atol=1e-12)
    def test_robustness_monotonicity_and_restore(self):
        x=np.tile(np.linspace(0,1,300),3);y=45-30*x;groups=np.repeat(np.arange(3),300)
        y[::19]=95
        m=RobustMetricSpline.fit(x,y,groups,smoothness=.02)
        pred=m.predict(np.linspace(0,1,300).reshape(15,20))
        self.assertLess(np.abs(pred.ravel()-(45-30*np.linspace(0,1,300))).mean(),.3)
        self.assertTrue((np.diff(pred.ravel())<=1e-7).all())
        np.testing.assert_array_equal(pred,RobustMetricSpline.from_dict(m.to_dict()).predict(np.linspace(0,1,300).reshape(15,20)))
    def test_basis_and_invalid(self):
        b=interpolation_basis([-2,0,.5,1,3],[0,.5,1]);np.testing.assert_allclose(b.sum(axis=1),1)
        self.assertTrue((b>=0).all())
        with self.assertRaises(ValueError):depth_feature(np.array([[np.nan]]),'raw')
    def test_scene_replication_invariance(self):
        x=np.tile(np.linspace(0,1,60),2);y=40-20*x;g=np.repeat([0,1],60)
        a=RobustMetricSpline.fit(x,y,g,scene_temperature=2)
        b=RobustMetricSpline.fit(np.r_[x,x[:60]],np.r_[y,y[:60]],np.r_[g,g[:60]],scene_temperature=2)
        # Knot locations can change with replication; linear clean relation should remain fitted.
        q=np.linspace(.05,.95,40).reshape(5,8)
        np.testing.assert_allclose(a.predict(q),b.predict(q),atol=.15)
if __name__=='__main__':unittest.main()
