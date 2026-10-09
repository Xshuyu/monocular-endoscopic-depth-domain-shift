"""Prepare an isolated replay; --run executes the already-observed protocol."""
import argparse, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--destination',type=Path,required=True)
 p.add_argument('--run',action='store_true')
 a=p.parse_args(); target=a.destination.resolve()/'lesion_3d_measurement'
 if target.exists():raise SystemExit('Destination already exists; choose a new directory.')
 target.mkdir(parents=True)
 for name in ['src','scripts','tests']:
  shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 if (ROOT/'__init__.py').exists():shutil.copy2(ROOT/'__init__.py',target/'__init__.py')
 for name in ['datasets','third_party']:(target/name).symlink_to(ROOT/name,target_is_directory=True)
 out=target/'outputs/publication_study';out.mkdir(parents=True)
 for name in ['c3vdv2_external_validation','robust_depth_optimization']:
  (target/'outputs'/name).symlink_to(ROOT/'outputs'/name,target_is_directory=True)
 source=ROOT/'outputs/publication_study'
 for name in ['protocol.json','cecum_t1_a_manifest.json','desc_t4_a_manifest.json','sigmoid_t1_a_manifest.json']:
  shutil.copy2(source/name,out/name)
 commands=[[str(PYTHON),f'scripts/{script}.py','--phase',phase] for phase in ['development','final'] for script in ['publication_inference','publication_calibration_study']]
 commands.append([str(PYTHON),'scripts/build_publication_analysis.py'])
 (target/'replay.json').write_text(json.dumps({'purpose':'Reproduction of observed results; not a new blind test','shared_readonly_inputs':str(ROOT),'commands':commands},indent=2))
 print(target,flush=True)
 if a.run:
  for command in commands:subprocess.run(command,cwd=target,check=True)
if __name__=='__main__':main()
