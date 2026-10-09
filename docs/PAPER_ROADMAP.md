# 顶会/顶刊论文大纲与实验方案 (前沿数学增强版)

**Target Venues:** Medical Image Analysis (MedIA) / IEEE Transactions on Medical Imaging (IEEE TMI) / MICCAI (Oral Target)  
**Title:** *MetricEndo: In-Vivo 3D Lesion Metrology and Paris Morphometry in Monocular Endoscopy via Lie Group Optimization and Differential Manifold Profiling*

---

## 1. 核心理论创新 (Theoretical Highlights)
1. **Lie Algebra $\mathfrak{so}(3)$ Riemannian Optimization for Robust OBB:**
   消除了内镜临床视频中反光伪影导致的 PCA 轴向发散，在李代数切空间构建抗噪体积最小化泛函。
2. **Heat Method on Continuous Point Manifolds:**
   引入拉普拉斯-贝尔特拉米算子与瓦拉丹对数热核势，在非刚性消化道曲面上实现旋转不变、网格无偏的连续测地线长径恢复。
3. **Weingarten Curvature Tensor & Automated Paris Morphometry:**
   首次将微分几何中的形状指数（Shape Index）引入单目内镜，实现从单一尺寸估计到恶变浸润形态学（Paris 0-Ip/Is/IIa/IIb/IIc）的深度数学表征。

---

## 2. 对比消融实验设计 (Ablation & Baseline Matrix)

| 实验组别 | 3D 姿态估计方法 | 测地线计算方式 | 尺度标定方式 | 预期长径误差 (MRE %) | 极端反光下鲁棒性 | 形态学量化 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1** | 2D 像素等效圆 | 无 (平面欧氏) | 经验视场估算 | ~ 28.5% | 极差 | 无 |
| **Baseline 2** | 传统 PCA | 离散 Dijkstra | 统计物距先验 | ~ 14.2% | 差 (反光点漂移) | 无 |
| **Ours (Lite)**| 传统 PCA | Heat Method | 器械几何先验 | ~ 8.6% | 中等 | 无 |
| **Ours (Full)**| **李代数 $\mathfrak{so}(3)$ 优化** | **Heat Method (LB算子)** | **器械几何先验** | **< 4.8%** | **极高 (Huber稳健)** | **Weingarten 形状指数** |

---

## 3. 实验验证数据集配置
- **尺度基准与器械分割：** Kvasir-Instrument (590 例，已下载解压并对齐)
- **病灶分割与形态学标注：** Kvasir-SEG (1,000 例，已完成测试管线对接)
- **真值 3D 几何与深度：** 
  - 计算机断层扫描（CT）配准真实 3D 肠道模型数据（Colonoscopy 3D Video, MedIA 2023）
  - 动态多器官真实 ToF 深度基准（DRENDS, Scientific Data 2026）
