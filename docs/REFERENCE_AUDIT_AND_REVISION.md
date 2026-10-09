> 更新：此前 22 篇版本已由本次 26 篇版本替代。上一轮偏重经典出处，对 2025–2026 年领域进展覆盖不足。以下历史核查仍保留；最新补充见文末。

# 参考文献核查与修订报告

核查日期：2026-10-02。对象：manuscript_mbec 和 manuscript_cmig。

## 结论

原稿 9 篇参考文献对应真实文献，未发现虚构条目；但正式发表信息不完整，相关工作与统计方法的直接引用不足。已修订原有条目并增加 13 篇，共 22 篇。这个数量是本稿覆盖需求的结果，不是期刊硬性要求，也不代表必然达到录用标准。

核查包括 DOI 与题名、作者、年份、期刊或会议的一致性，以及引用位置是否支持相应论述。证据来自 Crossref 和出版商、作者主页或 arXiv；逐条结构化记录保存在 outputs/reference_audit/verified_references.json。此核查不等于对每篇被引论文的全部实验进行独立复现。

## 原有九篇逐项结论

| 文献 | 结论 | 核查来源 |
|---|---|---|
| Bobrow2023：Colonoscopy 3D video dataset with paired depth from 2D-3D registration | 真实；题名、期刊和 DOI 对应，保留。 | [原文/DOI](https://doi.org/10.1016/j.media.2023.102956) |
| Golhar2026：C3VDv2 - Colonoscopy 3D video dataset with enhanced realism | 真实；已正式发表于 2026 年，并非未来占位条目，保留。 | [原文/DOI](https://doi.org/10.1038/s41597-026-07471-1) |
| Yang2024：Depth Anything V2 | 原为 arXiv 引用；更新为 NeurIPS 2024 正式版本、卷页及 DOI。 | [原文/DOI](https://doi.org/10.52202/079017-0688) |
| Crane2013：Geodesics in heat: A new approach to computing distance based on heat flow | 真实；保留 2013 年 TOG 原始版本，补全题名与文章页码。 | [原文/DOI](https://doi.org/10.1145/2516971.2516977) |
| Edwards2022：SERV-CT: A disparity dataset from cone-beam CT for validation of endoscopic 3D reconstruction | 真实；2022 年卷期与含 2021 的 DOI 不矛盾，保留。 | [原文/DOI](https://doi.org/10.1016/j.media.2021.102302) |
| Cui2024：EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera | 原为 arXiv 引用；更新为 MICCAI 2024 正式章节、卷页及 DOI。 | [原文/DOI](https://doi.org/10.1007/978-3-031-72089-5_20) |
| Song2025：Polyp-Size: A Precise Endoscopic Dataset for AI-Driven Polyp Sizing | 真实；补全 Scientific Data 12、文章号 918。 | [原文/DOI](https://doi.org/10.1038/s41597-025-05251-x) |
| Allan2021：Stereo Correspondence and Reconstruction of Endoscopic Data Challenge | 保留 arXiv 预印本；规范 McLeod 拼写和 Congcong Wang 的名字缩写 C.。 | [原文/DOI](https://arxiv.org/abs/2101.01133) |
| Lipson2021：RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching | 真实；补全 3DV 页码 218–227 和正式 DOI。 | [原文/DOI](https://doi.org/10.1109/3DV53792.2021.00032) |

## 新增文献与目的

新增文献在正文有对应引用，不是单纯扩充文末列表。

| 文献 | 对应论述 | 来源 |
|---|---|---|
| Ranftl2022 | 相对深度跨域泛化与毫米尺度的区别 | [原文/DOI](https://doi.org/10.1109/TPAMI.2020.3019967) |
| Bhat2023 | 相对预训练与度量深度预测的区别 | [原文/DOI](https://arxiv.org/abs/2302.12288) |
| Ozyoruk2021 | 内镜深度与视觉里程计相关工作 | [原文/DOI](https://doi.org/10.1016/j.media.2021.102058) |
| Shao2022 | 内镜亮度不一致及 appearance flow 方法 | [原文/DOI](https://doi.org/10.1016/j.media.2021.102338) |
| Ou2026 | 近期不均匀照明建模；界定本方法并未恢复照明物理量 | [原文/DOI](https://doi.org/10.1609/aaai.v40i10.37766) |
| Mayer2016 | SceneFlow 数据来源 | [原文/DOI](https://doi.org/10.1109/CVPR.2016.438) |
| Hirschmuller2008 | 半全局匹配方法来源 | [原文/DOI](https://doi.org/10.1109/TPAMI.2007.1166) |
| Zhang2000 | 相机几何标定与深度数值映射的区别 | [原文/DOI](https://doi.org/10.1109/34.888718) |
| Hoerl1970 | 岭回归的经典来源，避免将正则化本身称为创新 | [原文/DOI](https://doi.org/10.1080/00401706.1970.10488634) |
| Varma2006 | 嵌套模型选择与评估偏差 | [原文/DOI](https://doi.org/10.1186/1471-2105-7-91) |
| Cawley2010 | 反复模型选择与选择偏差 | [原文/DOI](https://jmlr.org/papers/v11/cawley10a.html) |
| Koo2016 | ICC 模型选择及报告要求 | [原文/DOI](https://doi.org/10.1016/j.jcm.2016.02.012) |
| Bland1986 | 测量一致性不能由相关性替代 | [原文/DOI](https://doi.org/10.1016/S0140-6736(86)90837-8) |

## 引用边界与剩余问题

- 新增相关工作不表示这些算法均已复现。尚缺统一协议下、训练数据无重叠的领域强单目基线。
- Zhang 标定文献不证明本研究已完成新增设备标定；径向回归系数也不能直接解释为光照或桶形畸变的物理参数。
- ICC 和 Bland–Altman 文献支持未来真实尺寸与重复标注评价的设计，不表示已获得临床 ICC 或一致性结果。
- SceneFlow 原论文说明数据集来源；具体模型的训练来源仍需官方权重说明等证据。
- 参考文献增多不能替代独立对象验证、基线复现和真实毫米参照。
- 投稿前应按最终目标期刊调整格式，并检查截稿前是否出现与本方法直接竞争的新工作；本次不主张穷尽所有相关文献。

## 文件和验证

两版正文补充相关工作、正则化、模型选择及一致性评价引用；同时提供 references.bib 供后续迁移。当前 LaTeX 仍使用内嵌 thebibliography，BibTeX 文件为同步导出。

代码引用检查：两版均为 22 个唯一文献条目，无未定义引用、无未引用条目。Tectonic 编译成功；PDF 文本没有未解析的 ??，新增相关工作和文献可检索。MBEC 版 22 页，CMIG 版 21 页。编译仍有附录表格的松行、CMIG 页眉轻微溢出及 lineno 包编码警告，不能将编译成功等同于最终出版排版验收。

没有修改算法、原始数据或实验数值，没有重跑训练。原稿源码备份为 outputs/reference_audit/manuscripts_before_reference_revision.zip。

复现：使用指定 venv-gpu 环境运行 scripts/revise_references.py，再分别编译主 tex。若重新运行旧版手稿生成脚本，须最后运行本参考文献修订脚本，以免引用修订被覆盖。crossref_metadata.json 包含检索过程中被排除的候选 DOI，它们不是原稿错误条目，也未进入最终参考文献。

## 2025–2026 年文献覆盖修订

截至 2026-10-02 核查。新增四篇均有正式出版来源，未把搜索候选或未核实稿件写入正文。

- **MonoPCC: Photometric-invariant cycle constraint for monocular depth estimation of endoscopic images** (2025)，Medical Image Analysis。来源：https://doi.org/10.1016/j.media.2025.103534。引用用途：Photometric-invariant cycle training; not locally reproduced。
- **Scale-consistent 3D reconstruction in monocular colonoscopy via camera-intrinsics-guided learning** (2026)，Computerized Medical Imaging and Graphics。来源：https://doi.org/10.1016/j.compmedimag.2026.102734。引用用途：Intrinsics-guided scale-consistent colonoscopy reconstruction; not equivalent to clinical size validation。
- **A Dataset for Depth in Robotic Endoscopy with Dynamic Scenarios (DRENDS)** (2026)，Scientific Data。来源：https://doi.org/10.1038/s41597-026-07775-2。引用用途：Dynamic endoscopy dataset and alignment-aware evaluation; not evaluated locally。
- **Geometry-Aware Depth-Guided Explainable Multimodal Polyp Size Estimation: A Fusion Model Beyond RGB** (2026)，Proceedings of The 9th International Conference on Medical Imaging with Deep Learning, Proceedings of Machine Learning Research。来源：https://proceedings.mlr.press/v315/lawrence26a.html。引用用途：Direct multimodal polyp-size regression; not a comparable local depth-MAE result。

当前共 26 篇，其中 2024–2026 年共 9 篇。不能以年份比例替代文献相关性评价：经典统计和算法出处应保留；近期方法必须用于界定现有差距，而不是充当已完成基线验证的证据。新增正文明确尚未在匹配协议下证明优于 MonoPCC 或内参引导方法，未完成 DRENDS 动态验证，也未证明临床尺寸精度。本次未修改实验数值。两版均编译通过，26 条引用无未定义或漏引。旧版交付包保留为历史版本，新包为 recent_literature_revision_delivery.zip。
