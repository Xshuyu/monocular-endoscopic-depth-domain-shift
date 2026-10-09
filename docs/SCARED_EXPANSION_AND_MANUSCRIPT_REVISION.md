# 对象扩展、无 SCARED 训练的双目参考与论文修订

已完成新增五个SCARED对象的实测关键帧验证；与此前两个对象合计七对象35帧，按原论文dataset级身份统计。不同研究供体关系未证实，不与SERV-CT简单合并宣称九个独立供体。

|方法|新增5对象 MAE|此前2对象 MAE|累计7对象 MAE|
|---|---:|---:|---:|
|Small + affine LS|45.624|40.115|44.050|
|Small + affine ridge|45.624|40.115|44.050|
|Small + radial LS|42.216|36.611|40.615|
|Small + radial ridge|42.829|37.115|41.196|
|Large + affine LS|42.166|37.250|40.761|
|Large + radial LS|40.460|35.163|38.947|
|SGBM (stereo)|13.111|15.133|13.689|
|RAFT-Stereo (SceneFlow)|8.142|9.857|8.632|

预设主对比（Small径向正则化−仿射LS）新增五对象差值 -2.795 mm；累计七对象 -2.854 mm。负值才表示改善。累计对象bootstrap探索性95%区间 [-3.406, -2.143] mm。不得用已观察两个对象后的扩展称为全新七对象盲测。

已加入官方 SceneFlow RAFT-Stereo 与无学习参数 SGBM。它们使用左右目与相机标定，比单目方法多输入信息，只能单列为几何参考。领域强基线问题部分解决：有无SCARED适配的强双目参考，但仍没有核查合格的、同等输入的领域适配单目基线；未将这项缺口伪装为完成。

真实毫米病灶尺寸参照、临床重复标注和ICC仍未完成；本轮未进行神经网络训练。扩展只评实测关键帧，不按测试结果选择lambda或调整裁剪上限。

RA​FT权重来源：https://github.com/princeton-vl/RAFT-Stereo （官方SceneFlow checkpoint）；预训练来源依据发布说明，不能声称审计了原作者全部训练日志。

论文更新：两个版本均新增SCARED实验、基线输入差异、对象级图表、训练重叠说明，重写摘要与结论。主要叙事仍是校准迁移的作用及局限，不声称毫米临床精度或SOTA。

复现（项目根目录，统一指定Python）：
```bash
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/acquire_scared_expansion.py
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_expansion.py mono
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_expansion.py stereo
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/run_scared_expansion.py score
/ssd-1/temp/kimicode/multmodel/venv-gpu/bin/python scripts/revise_scared_expansion_manuscript.py
```
score保护已有结果，不允许直接覆盖；RAFT源码和权重可通过 scripts/acquire_raft_reference.py 恢复，opt_einsum==3.4.0 安装在 third_party/raft_runtime_deps 隔离目录。扩展的5对象属于原挑战训练分区，但未用于本研究冻结C3VD头的拟合；不能将它们声称为官方挑战测试集。若运行旧稿生成器，最后必须重新运行本修订脚本，再编译PDF。
