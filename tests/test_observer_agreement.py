"""Analytical software checks, not evidence of clinical agreement."""
import unittest
import numpy as np
from src.observer_agreement import absolute_agreement_icc

class AgreementChecks(unittest.TestCase):
 def test_identical_measurements(self):
  self.assertAlmostEqual(absolute_agreement_icc([[1,1,1],[2,2,2],[5,5,5]])['icc_a1'],1.)
 def test_systematic_rater_offset_is_penalized(self):
  # MS objects=2, MS raters=1.5, MS error=0 -> A1=2/3, Ak=0.8.
  v=absolute_agreement_icc([[1,2],[2,3],[3,4]])
  self.assertAlmostEqual(v['icc_a1'],2/3)
  self.assertAlmostEqual(v['icc_ak'],.8)
 def test_invalid_and_degenerate_values_rejected(self):
  for x in [np.ones((3,2)),np.array([[1,np.nan],[2,3]]),np.array([[1,2]])]:
   with self.assertRaises(ValueError):absolute_agreement_icc(x)
if __name__=='__main__':unittest.main()
