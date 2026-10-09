"""Analytical correctness tests, not independent physical/clinical performance."""
import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.radial_affine_calibration import predict,radius_squared
from src.calibrated_depth_measurement import measure_axial_depth
class Integration(unittest.TestCase):
 def test_known_field_recovers_same_axial_geometry(self):
  h,w=120,160;y,x=np.mgrid[:h,:w];r=radius_squared((h,w));rays=np.stack([(x-w/2)/150,(y-h/2)/150,np.ones_like(x)],axis=-1)
  z=30+.015*x+.02*y;mask=((x-80)/25)**2+((y-60)/15)**2<1
  c=[.01,.004,.02,.003];d=(1/z-c[2]-c[3]*r)/(c[0]+c[1]*r)
  recovered=predict({'coefficients_a0_a1_b0_b1':c},d)
  reference=measure_axial_depth(z,mask,rays);result=measure_axial_depth(recovered,mask,rays)
  self.assertAlmostEqual(reference['major_q01_q99_mm'],result['major_q01_q99_mm'],places=9)
 def test_rays_must_match_pixel_grid(self):
  with self.assertRaises(ValueError):measure_axial_depth(np.ones((10,10))*30,np.ones((10,10)),np.ones((9,10,3)))
if __name__=='__main__':unittest.main()
