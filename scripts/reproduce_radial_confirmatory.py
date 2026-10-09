"""Isolated replay of an observed study, never a new untouched test."""
import argparse,shutil,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PYTHON='/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();dest=a.destination.resolve()/'lesion_3d_measurement'
if dest.exists():raise SystemExit('Choose a new destination; refusing overwrite')
dest.mkdir(parents=True)
for name in ['src','scripts','tests']:shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['datasets','third_party']:(dest/name).symlink_to(ROOT/name,target_is_directory=True)
(dest/'outputs').mkdir();(dest/'outputs/publication_study').symlink_to(ROOT/'outputs/publication_study',target_is_directory=True)
out=dest/'outputs/radial_confirmatory';out.mkdir();source=ROOT/'outputs/radial_confirmatory';shutil.copy2(source/'protocol.json',out/'protocol.json')
for scene in json.loads((source/'protocol.json').read_text())['new_test']:shutil.copy2(source/(scene+'_manifest.json'),out/(scene+'_manifest.json'))
commands=[[PYTHON,'scripts/run_radial_confirmatory.py',phase] for phase in ['development','inference','final']]+[[PYTHON,'scripts/analyze_radial_confirmatory.py']]
(dest/'replay.json').write_text(json.dumps({'note':'Reproduction after test observation; shares existing data and frozen-backbone development predictions','commands':commands},indent=2))
print(dest,flush=True)
if a.run:
 for cmd in commands:subprocess.run(cmd,cwd=dest,check=True)
