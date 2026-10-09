import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from publication_calibration_study import balanced_samples,fit_head,apply_head,BUDGET
class PublicationStudyTests(unittest.TestCase):
 def samples(self):
  samples={}
  for scene in ['a','b','c']:
   x=np.linspace(.1,2,8192)
   samples[(scene,'0')]={'gt':1/(.02*x+.03),'small':x,'large':2*x,'endodac':x/2,'small_norm':x}
  return samples
 def test_held_scene_excluded_and_budget_matched(self):
  samples=self.samples();samples[('c','0')]['gt'][:]=9999
  single=balanced_samples(samples,['a']);multiple=balanced_samples(samples,['a','b'])
  self.assertEqual(len(single['gt']),BUDGET);self.assertEqual(len(multiple['gt']),BUDGET)
  self.assertFalse((multiple['groups']=='c').any());self.assertLess(multiple['gt'].max(),100)
 def test_inverse_affine_recovery(self):
  d=balanced_samples(self.samples(),['a','b']);h=fit_head({'kind':'affine'},d)
  self.assertAlmostEqual(h['a'],.02,places=10);self.assertAlmostEqual(h['b'],.03,places=10)
  np.testing.assert_allclose(apply_head(h,d['small']),d['gt'],atol=1e-10)
 def test_constant_and_monotone_bounds(self):
  d=balanced_samples(self.samples(),['a','b']);h=fit_head({'kind':'monotone'},d)
  p=apply_head(h,np.array([-1.,.1,1,2,100.]));self.assertTrue((np.diff(p)<=0).all());self.assertTrue(((p>=1)&(p<=100)).all())
if __name__=='__main__':unittest.main()
