import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.radial_affine_calibration import fit,predict,radius_squared
class RadialTests(unittest.TestCase):
 def test_recovers_known_radial_field(self):
  rng=np.random.default_rng(17);d=rng.uniform(.1,8,4000);r=rng.uniform(0,1,4000);z=1/(.01*d+.004*d*r+.02+.003*r)
  h=fit(d,r,z,np.repeat(['a','b'],2000),True,0)
  np.testing.assert_allclose(h['coefficients_a0_a1_b0_b1'],[.01,.004,.02,.003],atol=1e-10)
  np.testing.assert_allclose(predict(h,d,r),z,atol=1e-9)
 def test_ridge_shrinks_standardized_slopes(self):
  rng=np.random.default_rng(1);d=rng.uniform(0,5,2000);r=rng.uniform(0,1,2000);z=1/(.01*d+.003*r+.02);g=np.zeros(2000)
  a=fit(d,r,z,g,True,0);b=fit(d,r,z,g,True,100)
  self.assertLess(np.linalg.norm(b['standardized_beta'][:-1]),np.linalg.norm(a['standardized_beta'][:-1]))
 def test_depth_bounds_and_radius(self):
  r=radius_squared((1080,1350));self.assertEqual(r[540,675],0);self.assertEqual(r[0,0],1)
  p=predict({'coefficients_a0_a1_b0_b1':[-1,0,0,0]},np.array([0.,1.,-2.]),np.zeros(3));np.testing.assert_equal(p,[100,100,1])
if __name__=='__main__':unittest.main()
