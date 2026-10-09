"""Exploratory development-only anchor sensitivity, never used to change frozen heads."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from publication_calibration_study import OUT,prepare_samples,balanced_samples,fit_head,sampled_score,csv_write,sha
frames=json.loads((OUT/'development_frames.json').read_text());samples=prepare_samples(frames);scenes=sorted({r['scene'] for r in frames})
before=sha(OUT/'frozen_heads.json');rows=[]
for held in scenes:
 for anchor in scenes:
  if anchor==held:continue
  head=fit_head({'kind':'affine'},balanced_samples(samples,[anchor]))
  rows.append({'held_scene':held,'calibration_anchor':anchor,'sampled_mae_mm':sampled_score(head,samples,held)})
csv_write(OUT/'exploratory_anchor_sensitivity.csv',rows)
assert sha(OUT/'frozen_heads.json')==before
print('30 development-only anchor evaluations; final model unchanged')
