"""Summarize the frozen selection, retaining unsuccessful mathematical ablations."""
from pathlib import Path
import json,csv,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/robust_depth_optimization'
frozen=json.loads((out/'frozen_selection.json').read_text());selected=frozen['selected']
summary=json.loads((out/'test_summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
names={'new_c2_sigmoid_video':'本轮新增c2乙状结肠视频（8帧）','reused_c3vdv2_test':'复用C3VDv2测试（72帧/6视频）','reused_c3vd_test':'复用C3VD测试（12帧/1视频）'}
lines=['# 以毫米MAE为目标的数学优化实验','',
'已实现并实测。验证集和旧C3VDv2的MAE改善，但新增视频没有实质改善，尚不能宣称稳定泛化、SOTA或临床尺寸精度提升。原算法及论文结果未被覆盖。','',
'## 数学方法','',
'固定Depth Anything V2 Small骨干，仅拟合从相对逆深度到毫米Z的校准头。相对深度的12个分位数节点上定义分段线性函数f。约束节点毫米值在[1,100]内且单调递减，用SLSQP求解凸约束优化：','',
'`min_v mean_scene mean_pixel [sqrt((Bv-z)^2 + 0.5^2) - 0.5] + lambda * mean((D2 v)^2)`','',
'B为分段线性插值矩阵，D2为节点值二阶差分；后者是离散节点粗糙度惩罚，并非非均匀节点上的精确物理曲率。另比较场景损失的温度2mm的log-sum-exp稳健聚合，以及逐帧5–95%相对深度归一化。它们都是待验证的数学组合，不宣称原创或前沿SOTA。','',
'优化目标直接作用于毫米残差。旧仿射标定最小化逆深度平方误差，二者并不等价。单调性保留近远顺序；固定少量节点限制自由度；平滑绝对误差降低大残差对拟合的影响。','',
'## 数据与选择规则','',
'拟合：原c1 ascending的12帧，加新增c1 cecum/rectum/sigmoid各8帧，共36帧；每帧2048个均匀有效像素。数学优化各场景等权；对照仿射基线也按场景等权拟合。验证：c1 transverse1和transverse2各8帧，视频等权MAE。','',
'2种输入表示×3种平滑系数×2种场景聚合，共12个预先规定方案，与同36帧拟合的仿射和64分箱单调基线共同选择。64分箱基线采用其原始样本加权实现，因此不是场景权重完全匹配的消融。','',
'本轮新留出为此前未评分的c2_sigmoid_t1_v3的8帧，选帧先于预测。它是新视频，但来自已经观察过的c2解剖模型，不是新患者/新解剖模型。其他两个测试集已在先前实验中观察，全部标为复用。校准、验证、测试图像做字节重复检查。镜像文件校验通过，但未与不可访问的官方Dataverse逐字节比对。','',
f'验证集选中 `{selected}`：不使用逐帧归一化、额外二阶差分惩罚或最差场景聚合。不能把未被选中的复杂项写成有效创新。验证MAE：同数据仿射 {frozen["validation_mae"]["matched_affine"]:.3f} → 候选 {frozen["validation_mae"][selected]:.3f} mm；64分箱单调基线 {frozen["validation_mae"]["matched_monotone"]:.3f} mm。','',
'## 冻结后结果','',
'| 数据 | 原12帧仿射MAE | 同36帧仿射MAE | 冻结候选MAE |','|---|---:|---:|---:|']
for dataset,title in names.items():
 m={r['method']:r for r in summary if r['evaluation_set']==dataset}
 lines.append(f"| {title} | {m['original_small_affine']['mae_mm']:.3f} | {m['matched_affine']['mae_mm']:.3f} | {m[selected]['mae_mm']:.3f} |")
lines += ['',
'单位均为mm，越小越好。候选在旧C3VDv2相对原仿射MAE降低约2.51%，但AbsRel从0.572升至0.604，不能宣称所有指标提升。在新乙状结肠视频上12.549→12.548mm基本没有变化；同数据的简单仿射达到9.636mm，优于候选。在复用C3VD上，候选改善到10.656mm，但同数据仿射达到7.521mm。扩充数据与数学方法的贡献必须分开。','',
'下面保留预先指定的数学消融；不根据测试结果换选模型。','',
'| 方法 | 旧C3VDv2 MAE | 新乙状结肠 MAE | 旧C3VD MAE |','|---|---:|---:|---:|']
for name in ['matched_affine','matched_monotone',*protocol['predeclared_test_ablations']]:
 vals=[next(r['mae_mm'] for r in summary if r['method']==name and r['evaluation_set']==d) for d in ['reused_c3vdv2_test','new_c2_sigmoid_video','reused_c3vd_test']]
 lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
lines += ['',
'## 可运行入口','',
'```bash',
'/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B scripts/predict_optimized_depth.py \\\n  --image /path/to/image.png --output /tmp/optimized_depth_result',
'```','',
'输出depth_mm.npy、relative_depth.npy和result.json。只与固定Small骨干相匹配；使用原始相对深度NPY时，可用--relative替代--image。可选--mask与--rays仅接受真实像素对应标定输入，输出仍为MAD-PCA的1–99%几何范围，不是临床卡尺直径。','',
'重跑完整实验（使用尚不存在的新目录；先拟合冻结，再评估）：','',
'```bash',
'OPENBLAS_NUM_THREADS=1 /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/run_robust_depth_optimization.py --output /tmp/robust_depth_repeat',
'OPENBLAS_NUM_THREADS=1 /ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python -B tests/run_robust_depth_optimization.py --output /tmp/robust_depth_repeat --evaluate',
'```','',
'## 证据边界','',
'只有两个c1验证视频，14种候选选择可能带来验证集乐观偏差；新增c2视频仅一个，不能估计患者总体泛化。网络没有微调，只训练/拟合小型校准头。没有真实标定的临床尺寸评估，不能声称病灶毫米测量改善；也未声称实现了多帧几何优化。多帧方法需要连续数据、遮挡处理和可靠相机模型，不能拿稀疏抽帧冒充时序验证。','',
'关键代码：src/robust_metric_depth.py；tests/run_robust_depth_optimization.py；scripts/predict_optimized_depth.py。所有12个数学模型均收敛，模型与协议在测试前冻结，保留所有预定消融与负结果。']
(ROOT/'docs/ROBUST_DEPTH_OPTIMIZATION_REPORT.md').write_text('\n'.join(lines)+'\n')
print('Report written')
