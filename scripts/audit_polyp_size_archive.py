"""Audit official clinical archive; reserve every video and do not fit any model."""
from pathlib import Path,PurePosixPath
import subprocess,json,csv,zipfile,hashlib,io,re,cv2,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];archive=ROOT/'datasets/external/Polyp_Size_Dataset.rar'
out=ROOT/'outputs/next_validation';dest=ROOT/'datasets/external/Polyp_Size_official';dest.mkdir(exist_ok=True)
expected='c4f2fb4b2bbc207b618a155f2b2eeac1';h=hashlib.md5()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(4<<20),b''):h.update(block)
if h.hexdigest()!=expected:raise ValueError('Official archive checksum mismatch')
listing=subprocess.run(['unrar','lb',str(archive)],check=True,capture_output=True,text=True).stdout.splitlines()
for name in listing:
 path=PurePosixPath(name.replace('\\','/'))
 if path.is_absolute() or '..' in path.parts or ':' in str(path):raise ValueError('Unsafe archive member')
(out/'polyp_size_archive_inventory.json').write_text(json.dumps(listing,indent=2)+'\n')
subprocess.run(['unrar','x','-o-','-idq',str(archive),str(dest)+'/'],check=True)
for zpath in list(dest.rglob('*.zip')):
 with zipfile.ZipFile(zpath) as z:
  for n in z.namelist():
   p=PurePosixPath(n.replace('\\','/'))
   if p.is_absolute() or '..' in p.parts or ':' in str(p):raise ValueError('Unsafe nested archive member')
  z.extractall(zpath.parent)
files=[p for p in dest.rglob('*') if p.is_file()]
metadata=[]
for p in files:
 if p.suffix.lower()!='.csv':continue
 raw=p.read_bytes()
 if raw.startswith(b'PK'):
  ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   shared=[]
   if 'xl/sharedStrings.xml' in z.namelist():
    shared=[''.join(t.text or '' for t in item.findall('.//m:t',ns)) for item in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
   sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
   parsed=[]
   for row in sheet.findall('.//m:sheetData/m:row',ns):
    values={}
    for cell in row.findall('m:c',ns):
     ref=cell.attrib['r'];letters=''.join(c for c in ref if c.isalpha());col=0
     for c in letters:col=col*26+ord(c)-64
     v=cell.find('m:v',ns);value=v.text if v is not None else ''
     if cell.attrib.get('t')=='s':value=shared[int(value)]
     elif cell.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in cell.findall('.//m:t',ns))
     values[col-1]=value
    if any(values.values()):parsed.append([values.get(i,'') for i in range(max(values)+1)])
  columns=parsed[0];rows=parsed[1:];encoding='XLSX container mislabeled .csv'
  with (out/'polyp_size_labels_normalized.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(columns);w.writerows(rows)
 else:
  content=None
  for encoding in ['utf-8-sig','utf-16','gb18030','cp1252']:
   try:content=raw.decode(encoding);break
   except UnicodeError:continue
  if content is None:raise ValueError('Unrecognized label format')
  reader=csv.DictReader(io.StringIO(content));rows=list(reader);columns=reader.fieldnames
 metadata.append({'file':str(p.relative_to(dest)),'encoding':encoding,'columns':columns,'rows':len(rows)})

videos=sorted(p for p in files if p.suffix.lower()=='.mp4')
manifest=[{'video':str(p.relative_to(dest)),'role':'external_clinical_holdout','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mask_status':'not_audited','camera_calibration_status':'not_audited'} for p in videos]
(out/'polyp_size_holdout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={'official_md5_verified':True,'video_count':len(videos),'metadata_schema':metadata,'nonvideo_files':[str(p.relative_to(dest)) for p in files if p.suffix.lower() not in ['.mp4','.zip']],
'no_training_or_prediction':True,'labels_not_used_for_model_selection':True,'paper':'https://doi.org/10.1038/s41597-025-05251-x',
'endpoint_from_paper':'Resected specimen maximum diameter; largest of three observer measurements; not in-vivo 3D quantile extent',
'published_icc':{'value':.987,'ci':[.979,.993],'status':'Author-reported, not recomputed here'}}
labels=list(csv.DictReader((out/'polyp_size_labels_normalized.csv').open()))
label_map={r['Video_ID']:r for r in labels}
if len(label_map)!=42 or len(manifest)!=42:raise ValueError('Expected 42 unique video labels')
records=[]
for r in manifest:
 match=re.fullmatch(r'Video(\d+)\.mp4',Path(r['video']).name)
 if match is None:raise ValueError('Unexpected video naming')
 vid=match.group(1)
 value=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*mm\s*',label_map[vid]['Polyp_Size'])
 if value is None or float(value.group(1))<=0:raise ValueError('Invalid size units/value')
 size=float(value.group(1));cap=cv2.VideoCapture(str(dest/r['video']))
 if not cap.isOpened():raise ValueError('Unreadable video header')
 info={'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':float(cap.get(cv2.CAP_PROP_FPS))};cap.release()
 r.update({'video_id':vid,'label_size_mm':size,'mask_status':'not included in official archive','camera_calibration_status':'not included in official archive',**info})
 records.append({'video_id':vid,'video_path':str(dest/r['video']),'reference_size_mm':size,'mask_path':'','calibrated_rays_path':'','reference_endpoint':'post-resection max caliper diameter, maximum of three observers','role':'external_holdout'})
if len({r['video_id'] for r in manifest})!=42:raise ValueError('Duplicate video ID')
(out/'polyp_size_holdout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with (out/'clinical_measurement_inputs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
result.update({'video_label_pairs':42,'video_label_mapping':'Video_ID NN to VideoNN.mp4; exact 1:1, no missing IDs','size_unit':'explicit mm suffix parsed; all positive','patient_id_available':False,'raw_three_observer_measurements_available':False,'masks_available':False,'camera_calibration_available':False,'video_headers_readable':42,'pixel_frames_decoded_for_model':False})
(out/'polyp_size_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
