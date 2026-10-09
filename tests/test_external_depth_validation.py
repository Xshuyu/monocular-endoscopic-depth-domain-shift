"""Tests for reference decoding and train-only scale-fit semantics."""
import sys,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
from lesion_3d_measurement.tests.c3vdv2_external_depth_validation import read_reference,fit_calibration,predictors,metrics
class ExternalDepthTests(unittest.TestCase):
 def test_uint16_axial_units_and_invalid_pixels(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'depth.png';cv2.imwrite(str(p),np.array([[0,16384,32768,65535]],dtype=np.uint16))
   d,v=read_reference(p)
   np.testing.assert_array_equal(v,[[False,True,True,False]])
   self.assertAlmostEqual(float(d[0,1]),100.*16384/65535,places=5)
   self.assertAlmostEqual(float(d[0,2]),100.*32768/65535,places=5)
 def test_visualized_uint8_depth_is_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'depth.png';cv2.imwrite(str(p),np.array([[0,64,128,255]],dtype=np.uint8))
   with self.assertRaises(ValueError):read_reference(p)
 def test_calibration_recovers_known_models(self):
  gt=np.linspace(10,80,100);photo=gt/2;relative=(1/gt-.005)/.003
  fit=fit_calibration(photo,relative,gt)
  self.assertAlmostEqual(fit['photometric_scale'],2.)
  self.assertAlmostEqual(fit['depth_anything_inverse_affine_a'],.003)
  self.assertAlmostEqual(fit['depth_anything_inverse_affine_b'],.005)
  predictions,bad=predictors(photo,relative,fit)
  np.testing.assert_allclose(predictions['photometric_calibration_scale'],gt)
  np.testing.assert_allclose(predictions['depth_anything_v2_calibration_inverse_affine'],gt)
  self.assertFalse(bad.any())
 def test_metrics_have_mm_and_relative_units(self):
  gt=np.array([[10.,20.]]);pred=np.array([[12.,18.]])
  m=metrics(pred,gt,np.ones_like(gt,dtype=bool))
  self.assertEqual(m['mae_mm'],2.);self.assertEqual(m['rmse_mm'],2.)
  self.assertAlmostEqual(m['abs_rel'],.15);self.assertEqual(m['bias_mm'],0.)
if __name__=='__main__':unittest.main()
