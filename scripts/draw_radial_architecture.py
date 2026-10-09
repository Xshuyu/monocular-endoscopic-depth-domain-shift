"""Editable vector architecture mapped to the executed source-code workflow."""
from pathlib import Path
import json,xml.etree.ElementTree as ET,html
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.family':'DejaVu Sans'})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'manuscript_cmig/figures';OUT.mkdir(exist_ok=True)
W,H=1200,1160
# Coordinates are shared by the paper vector and editable diagrams.net objects.
lanes=[('development',20,20,1160,360,'A  DEVELOPMENT AND CALIBRATION','#edf4fb','#245b85'),('evaluation',20,435,1160,420,'B  FROZEN NEW-SEQUENCE EVALUATION','#eef6f2','#22624d'),('measurement',20,900,1160,240,'C  OPTIONAL SIZE MEASUREMENT','#faf5eb','#8a612a')]
boxes=[
 {'id':'dev_inputs','xy':[40,90,320,255],'title':'Development inputs','lines':['C3VDv2: 52 frames / 6 scenes','Frozen DA V2 Small → cached d','Reference axial depth Z','Native-grid radius r²','Valid reference-pixel mask'],'color':'#245b85','source':'scripts/run_radial_confirmatory.py: sample; scripts/publication_inference.py: development inference'},
 {'id':'calibration','xy':[440,90,320,255],'title':'Four matched calibration heads','lines':['Affine LS  |  Affine ridge','Radial LS  |  Radial ridge','16,384 pixels per fitted head','Scene-balanced sampling','Train-only feature scaling','Fit inverse-depth squared loss'],'color':'#245b85','source':'src/radial_affine_calibration.py: fit; scripts/run_radial_confirmatory.py: balanced'},
 {'id':'selection','xy':[840,90,320,255],'title':'Grouped selection and freeze','lines':['6 outer × 5 inner scene folds','Select ridge λ on development','Final λ: 6-fold development CV','Refit heads on development','Save coefficients and scaling','Freeze code + protocol hashes'],'color':'#245b85','source':'scripts/run_radial_confirmatory.py: development, verify_freeze'},
 {'id':'new_rgb','xy':[40,500,320,220],'title':'New RGB → frozen backbone','lines':['C3VD: 4 videos / 48 frames','Acquired after head freezing','Frozen DA V2 Small inference','Resize d to native image grid','No reference values in inference'],'color':'#22624d','source':'scripts/run_radial_confirmatory.py: acquire, inference'},
 {'id':'apply','xy':[440,500,320,220],'title':'Apply frozen calibration','lines':['Image-center radius r²','q = (a0 + a1 r²)d + b0 + b1 r²','Affine heads: a1 = b1 = 0','q > 0: Z = clip(1/q, 1, 100) mm','q ≤ 0: Z = 100 mm'],'color':'#22624d','source':'src/radial_affine_calibration.py: radius_squared, predict'},
 {'id':'score','xy':[840,500,320,220],'title':'Video-level depth evaluation','lines':['MAE / RMSE / AbsRel / δ1','Equal-video aggregation','Paired video diagnostics','Radius × reference-depth bins','No test-based refitting'],'color':'#22624d','source':'scripts/run_radial_confirmatory.py: evaluate, summarize, final; scripts/analyze_radial_confirmatory.py'},
 {'id':'reference','xy':[840,775,320,58],'title':'Test reference Z · scoring only','lines':[],'color':'#22624d','source':'tests/c3vdv2_external_depth_validation.py: read_reference'},
 {'id':'camera_inputs','xy':[40,965,320,145],'title':'Additional required inputs','lines':['Lesion mask','Calibrated per-pixel camera rays','Same native pixel grid'],'color':'#8a612a','source':'scripts/predict_radial_measurement.py: required mask/rays pairing'},
 {'id':'backproject','xy':[440,965,320,145],'title':'Axial-depth back-projection','lines':['P = Z · ray / ray_z','Radial MAD point filtering','PCA 1st–99th percentile extent'],'color':'#8a612a','source':'src/calibrated_depth_measurement.py: measure_axial_depth'},
 {'id':'size','xy':[840,965,320,145],'title':'Research size output','lines':['Not a clinical caliper diameter','Analytical interface tests only','Physical-size validation pending'],'color':'#8a612a','source':'scripts/predict_radial_measurement.py; tests/test_radial_measurement_integration.py'}]
edges=[('e1','dev_inputs','calibration',[(360,215),(440,215)],False),('e2','calibration','selection',[(760,215),(840,215)],False),('e3','selection','apply',[(1000,345),(1000,404),(600,404),(600,500)],False),('e4','new_rgb','apply',[(360,610),(440,610)],False),('e5','apply','score',[(760,610),(840,610)],False),('e6','reference','score',[(1000,775),(1000,720)],False),('e7','apply','backproject',[(600,720),(600,965)],True),('e8','camera_inputs','backproject',[(360,1035),(440,1035)],True),('e9','backproject','size',[(760,1035),(840,1035)],True)]
fig,ax=plt.subplots(figsize=(12,11.6));fig.subplots_adjust(0,0,1,1);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
for ident,x,y,w,h,title,fill,color in lanes:
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor=fill,edgecolor=color,linewidth=1,linestyle='--' if ident=='measurement' else '-');p.set_gid(ident);ax.add_patch(p);ax.text(x+20,y+32,title,fontsize=12,fontweight='bold',color=color,gid=ident+'_title')
for b in boxes:
 x,y,w,h=b['xy'];p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',facecolor='white',edgecolor=b['color'],linewidth=1.2);p.set_gid(b['id']);ax.add_patch(p)
 ax.text(x+w/2,y+28,b['title'],ha='center',va='center',fontsize=12,fontweight='bold',color=b['color'],gid=b['id']+'_title')
 for j,line in enumerate(b['lines']):ax.text(x+w/2,y+65+j*28,line,ha='center',va='center',fontsize=11 if b['id']=='apply' else 11.5,color='#243442',gid=b['id']+f'_line{j}')
for ident,source,target,points,dashed in edges:
 color='#8a612a' if dashed else '#425666'
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,linewidth=1.7,linestyle='--' if dashed else '-')
 p=FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.7,linestyle='--' if dashed else '-');p.set_gid(ident);ax.add_patch(p)
ax.text(795,399,'Frozen heads',ha='center',va='bottom',fontsize=11,color='#425666',bbox={'facecolor':'white','edgecolor':'none','pad':2})
ax.text(610,870,'Predicted axial depth Z',va='center',fontsize=11,color='#8a612a')
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'radial_architecture.{ext}',dpi=180,facecolor='white')
plt.close(fig)
# Editable draw.io: every box, label and connector is a separate native object.
mx=ET.Element('mxfile',host='app.diagrams.net');diagram=ET.SubElement(mx,'diagram',name='Radial calibration architecture',id='radial-architecture');model=ET.SubElement(diagram,'mxGraphModel',dx=str(W),dy=str(H),grid='1',gridSize='10',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H));root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
def cell(ident,value,x,y,w,h,style):
 c=ET.SubElement(root,'mxCell',id=ident,value=value,style=style,vertex='1',parent='1');ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
for ident,x,y,w,h,title,fill,color in lanes:cell(ident,title,x,y,w,h,f'rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={color};fontColor={color};fontStyle=1;fontSize=17;verticalAlign=top;align=left;spacingTop=12;spacingLeft=20;'+('dashed=1;' if ident=='measurement' else ''))
for b in boxes:
 x,y,w,h=b['xy'];value='<b>'+html.escape(b['title'])+'</b>' + ('<br><br>'+'<br>'.join(map(html.escape,b['lines'])) if b['lines'] else '')
 cell(b['id'],value,x,y,w,h,f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={b["color"]};fontColor=#243442;fontSize=16;spacing=12;')
for ident,source,target,points,dashed in edges:
 sx,sy=points[0];tx,ty=points[-1]
 # Explicit endpoints and waypoints preserve layout while remaining editable.
 source_box=next(b['xy'] for b in boxes if b['id']==source);target_box=next(b['xy'] for b in boxes if b['id']==target)
 anchors=f'exitX={(sx-source_box[0])/source_box[2]};exitY={(sy-source_box[1])/source_box[3]};entryX={(tx-target_box[0])/target_box[2]};entryY={(ty-target_box[1])/target_box[3]};exitPerimeter=0;entryPerimeter=0;'
 c=ET.SubElement(root,'mxCell',id=ident,value='',source=source,target=target,style=anchors+'edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor='+('#8a612a;dashed=1;' if dashed else '#425666;'),edge='1',parent='1')
 geo=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});ET.SubElement(geo,'mxPoint',x=str(sx),y=str(sy),attrib={'as':'sourcePoint'});ET.SubElement(geo,'mxPoint',x=str(tx),y=str(ty),attrib={'as':'targetPoint'})
 if len(points)>2:
  arr=ET.SubElement(geo,'Array',attrib={'as':'points'})
  for x,y in points[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
cell('freeze_label','Frozen heads',705,378,180,28,'text;html=0;fontSize=16;fillColor=#ffffff;strokeColor=none;')
cell('depth_label','Predicted axial depth Z',608,852,275,30,'text;html=0;fontSize=16;fontColor=#8a612a;strokeColor=none;')
ET.ElementTree(mx).write(OUT/'radial_architecture.drawio',encoding='utf-8',xml_declaration=True)
(OUT/'radial_architecture_sources.json').write_text(json.dumps({'scope':'Executed primary calibration workflow plus optional, unvalidated clinical sizing interface; historical fixed controls omitted for clarity','nodes':boxes,'edges':[{'id':e[0],'from':e[1],'to':e[2],'optional':e[4]} for e in edges]},indent=2)+'\n')
print('Generated SVG (editable text), drawio, PDF, PNG and source mapping')
