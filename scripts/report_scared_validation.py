"""Independently check SCARED summaries and write a transparent validation report."""
from pathlib import Path
import csv,json,hashlib,zipfile,sys
import numpy as np,tifffile
R=Path(__file__).resolve().parents[1];O=R/'outputs/scared_validation'
sys.path.insert(0,str(R))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):return list(csv.DictReader(p.open()))
frames=json.loads((O/'frames.json').read_text());rows=readcsv(O/'per_frame.csv');objs=readcsv(O/'per_object.csv');summ=json.loads((O/'summary.json').read_text());heads=json.loads((R/'outputs/independent_validation/frozen_heads.json').read_text())['heads']
checks={};deltas=[];tails=[]
for r in frames:
 xyz=tifffile.imread(R/r['depth']);gt=xyz[...,2].astype(float);valid=np.isfinite(xyz).all(-1)&(gt>0)
 tails.append({'object':r['object'],'frame':r['frame'],'quantiles_mm':dict(zip(['p50','p95','p99','p999','max'],np.quantile(gt[valid],[.5,.95,.99,.999,1]).tolist())),'fraction_gt250':float((gt[valid]>250).mean()),'fraction_gt1000':float((gt[valid]>1000).mean())})
 for m,h in heads.items():
  with np.load(O/'predictions'/m.split('_')[0]/f"{r['object']}_{r['frame']}.npz") as z: d=z['prediction'].astype(float)
  height,width=d.shape;yy,xx=np.indices(d.shape);rad=((xx-width/2)**2+(yy-height/2)**2)/((width/2)**2+(height/2)**2)
  a,b,c,e=h['coefficients_a0_a1_b0_b1'];inv=(a+b*rad)*d+c+e*rad;pred=np.full_like(inv,100);np.divide(1,inv,out=pred,where=inv>0);pred=np.minimum(np.maximum(pred,1),100)
  for support,mask in [('all_reference',valid),('reference_1_100',valid&(gt>1)&(gt<100))]:
   match=[x for x in rows if (x['object'],x['frame'],x['method'],x['support'])==(r['object'],r['frame'],m,support)];assert len(match)==1
   mae=float(np.sum(np.abs(pred[mask]-gt[mask]))/np.count_nonzero(mask));deltas.append(abs(mae-float(match[0]['mae_mm'])))
for x in objs:
 rr=[v for v in rows if (v['object'],v['method'],v['support'])==(x['object'],x['method'],x['support'])];assert len(rr)==5
 assert abs(float(x['mae_mm'])-sum(float(v['mae_mm']) for v in rr)/5)<1e-10
for x in summ:
 rr=[v for v in objs if (v['method'],v['support'])==(x['method'],x['support'])];assert len(rr)==2
 assert abs(x['mae_mm']-sum(float(v['mae_mm']) for v in rr)/2)<1e-10
checks['all_160_frame_mae_independently_recomputed']=max(deltas)<1e-9
checks['five_frames_per_subject_and_equal_subject_aggregation']=True
checks['ten_unique_rgb']=len(set(x['rgb_sha256'] for x in frames))==10
checks['all_30_acquired_members_hashed']=all(sha(R/x['local'])==x['sha256'] for x in json.loads((O/'keyframe_acquisition.json').read_text()))
checks['frozen_heads_unchanged']=sha(R/'outputs/independent_validation/frozen_heads.json')==json.loads((O/'protocol.json').read_text())['heads_sha256']
checks['inference_source_unchanged']=sha(R/'scripts/run_scared_validation.py')==json.loads((O/'inference.json').read_text())['script_sha256']
assert all(checks.values())
(O/'validation_checks.json').write_text(json.dumps({'checks':checks,'max_independent_mae_difference_mm':max(deltas)},indent=2)+'\n');(O/'reference_tail_audit.json').write_text(json.dumps(tails,indent=2)+'\n')
primary={x['method']:x for x in summ if x['support']=='all_reference'}
lines=['# SCARED 官方测试数据下载与冻结迁移验证','', '已完成两个精简测试包及官方代码下载，并从完整归档提取 dataset_8/9 的全部10个实测左关键帧、XYZ标签和相机标定。未下载完整训练集或全部插值视频；未训练模型、未拟合新校准头。','', '## 主要结果','', '单位：mm。每帧按全部有限XYZ且Z>0的原生有效像素计算MAE，先对每对象5帧取平均，再对2对象等权平均。','', '|方法|dataset_8|dataset_9|平均MAE|解释|','|---|---:|---:|---:|---|']
for m in ['small_affine_ls','small_affine_ridge','small_radial_ls','small_radial_ridge','large_affine_ls','large_radial_ls','endodac_affine_ls','endodac_radial_ls']:
 vals={x['object']:float(x['mae_mm']) for x in objs if x['method']==m and x['support']=='all_reference'}
 lines.append(f"|{m}|{vals['dataset_8']:.3f}|{vals['dataset_9']:.3f}|{primary[m]['mae_mm']:.3f}|{'训练清单存在对象重叠，仅供参考' if m.startswith('endodac') else '冻结C3VD校准迁移'}|")
a=primary['small_affine_ls']['mae_mm'];b=primary['small_radial_ridge']['mae_mm'];c=primary['small_radial_ls']['mae_mm']
lines+=['',f'预设主要对比：Small radial ridge 相对 affine LS 的MAE差为 {b-a:.3f} mm（下降 {(a-b)/a*100:.2f}%），两个对象均改善。但径向正则化结果比未正则化 radial LS 高 {b-c:.3f} mm，不能将优势归因于正则化。Small radial ridge 平均有符号误差为 '+f"{primary['small_radial_ridge']['bias_mm']:.3f} mm，明显低估深度；尚不支持高精度毫米测量或病灶尺寸有效性。",'', '## 下载、身份和公平性依据','', '- 官方仓库 https://huggingface.co/datasets/maxhallan7/scared ，固定 revision c1246e4f9f52e1ffd51fa45f8544288771785189。test_dataset_8.zip 为252229192字节，test_dataset_9.zip 为396748898字节，两者以及code.zip均与官方LFS SHA256一致且ZIP CRC检查通过。','- 完整dataset_8.zip/9.zip约51GB，因此使用HTTP Range只提取30个必要成员，约182MB传输；成员CRC与ZIP目录一致，另存本地SHA256。没有下载整个大归档，因此不声称验证了大归档整体SHA256。','- 原论文 https://arxiv.org/abs/2101.01133 第II节说明每个dataset对应一个猪对象；单位按原论文毫米深度指标与官方XYZ/Z表示读取。旧README的三个dataset描述与现行发布数量不符，采用论文和实际文件目录，保留原文快照。','- EndoDAC本地发布版 train_files.txt 包含dataset8共3134行、dataset9共2737行（总15351行）。这与此前依据常见挑战划分推测只训练1–7不一致，现已更正。实际权重逐样本训练历史无法独立核查，因此保守标为对象重叠风险，不能用来支撑公平的独立泛化排名。','- DA V2基础模型预训练数据不能完全审计；只能确认本项目未用SCARED进行拟合。对象级外部测试不等于基础预训练绝对无重叠。','- 新增的是两个SCARED测试对象。SERV-CT与SCARED跨研究供体关系未获得直接证明，因此分数据集报告，不直接宣称合计四个完全独立供体。','', '## 数据质量与评估边界','', '- 保留原生1024×1280图像，读取float32 XYZ的第三通道作为轴向深度，不使用点到相机的欧氏距离；预测插值回原生RGB尺寸，未重采样真值。','- 所有头保持原始1–100 mm输出范围；主结果不按真值上限删像素，另报告预先约定的1<GT<100支持域。两种支持域不是两个模型选优机会。','- 参照存在超出100mm的有效值以及少量极端值，最大约6567mm。reference_tail_audit.json记录每帧分位数和超界比例；没有根据观察到的误差事后删除这些点。','- XYZ经提供的K和畸变投影后的残差p95约1.14–1.79像素。这是内部投影一致性检查，不代表相机真实标定误差或物理深度精度。','- 只有两个对象，不进行像素级显著性检验，不把帧或随机种子充当独立样本。','- 仅实测关键帧、单目冻结校准迁移；官方挑战还包括插值视频和双目方法。因此这些数字不能直接与挑战排行榜比较，不声称复现官方排行榜或达到SOTA。','', '## 复现命令','', '```bash','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_scared_test.py','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_scared_keyframes.py','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_validation.py inference','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_validation.py evaluate','/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/report_scared_validation.py','```','', '在项目根目录运行。evaluate为保护已评分结果拒绝覆盖现有summary.json；现有结果可直接运行report脚本独立重算核对。下载可复用已校验文件；推理复用缓存。若开展新协议，应使用独立输出目录并先锁定方案。','', '关键文件：scripts/acquire_scared_test.py（下载与协议）、scripts/acquire_scared_keyframes.py（按范围提取）、scripts/run_scared_validation.py（推理与评分）、src/radial_affine_calibration.py（原冻结算法）、scripts/report_scared_validation.py（独立复算）。','', '新数据报告存于outputs/scared_validation。所有160行逐帧MAE已独立复算，汇总和原冻结头哈希通过检查。未更改论文正文，也未把深度验证当成真实病灶尺寸验证。']
(R/'docs/SCARED_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
files=[R/'docs/SCARED_VALIDATION_REPORT.md']+[R/'scripts'/x for x in ['acquire_scared_test.py','acquire_scared_keyframes.py','run_scared_validation.py','report_scared_validation.py']]+[p for p in O.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.log'}]
(O/'delivery_sha256.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in files},indent=2)+'\n')
with zipfile.ZipFile(R/'outputs/scared_validation_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[O/'delivery_sha256.json']:z.write(p,str(p.relative_to(R)))
print(json.dumps(checks,indent=2));print('MAIN_DELTA_MM',b-a,'RELATIVE_PERCENT',(a-b)/a*100)
