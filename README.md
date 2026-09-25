# FA-DAWRF：抗标签噪声的密度感知加权随机森林

> **Density-Aware Weighted Random Forest with Factor Analysis for Label-Noise-Robust Classification**

本仓库是论文 **《抗标签噪声的密度感知加权随机森林算法》**（中文核心期刊投稿稿）的官方复现代码与实验数据，包含全部算法实现、可一键复现的实验脚本，以及论文引用的完整结果 JSON。

---

## 1. 方法概述

标签噪声（label noise）会破坏"同类内邻域"假设：被错误标注的样本在其（带噪）标注类中通常缺乏近邻、密度异常低。FA-DAWRF 的核心思路是：

1. **因子分析降维**：用 Factor Analysis 把标准化特征投影到低维隐空间，缓解高维噪声对密度估计的干扰；
2. **类内密度异常分**：在因子空间上，按"（带噪）标签类"内计算 k 近邻密度，得到每个样本的密度异常分 $N_i \in [0,1]$；
3. **门控式难例加权**：构造门控分数

$$
s_i = \big(\alpha\, e_i + \beta\, H_i + \gamma\, B_i\big)\cdot\big(1 - N_i\big) - \delta\, N_i
$$

其中 $e_i$ 为袋外（OOB）错分指示，$H_i$ 为归一化熵（预测不确定性），$B_i$ 为边界距离项，$N_i$ 为密度异常分；随后以 $w \leftarrow w\cdot\exp(\eta\, s)$ 做乘法式重加权并迭代 $T$ 轮。其效果是：**高密度的"干净难例"升权、低密度的"噪声异常"降权**，从而在不依赖显式噪声率估计的前提下提升鲁棒性。

论文调优后的超参：$\delta=5.0,\ \alpha=1.0,\ k_{nn}=12,\ \beta=0.5,\ \gamma=0.5,\ \eta=0.5,\ T=4,\ m=5,\ n_{estimators}=100$（数值稳定常数 $\varepsilon_0=10^{-12}$）。

---

## 2. 环境要求

- Python 3.8+（开发环境：3.12 / 3.13）
- 依赖见 `requirements.txt`，核心仅需 numpy 与 scikit-learn：

```bash
pip install -r requirements.txt
```

---

## 3. 快速开始（最小示例）

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from fa_dawrf import FADAWRF, add_label_noise

X, y = load_breast_cancer(return_X_y=True)
y_noisy = add_label_noise(y, noise_rate=0.3, random_state=42)   # 注入 30% 对称标签噪声

model = FADAWRF()          # 默认超参即论文调优值
model.fit(X, y_noisy)
acc = (model.predict(X) == y).mean()
print(f"Accuracy = {acc:.4f}")
```

---

## 4. 复现论文全部实验

在仓库根目录下运行（脚本用相对路径、随机种子固定，结果与论文一致）：

```bash
python reproduce.py main        # 主实验：4 数据集 × 5 噪声 × 10 方法
python reproduce.py supp        # 补充实验：非对称/实例相关噪声 + wall-robot-navigation
python reproduce.py sens        # 敏感度：δ 与 α×k_nn
python reproduce.py ablation    # 消融实验（breast_cancer，30% 噪声）
python reproduce.py viz         # 因子空间可视化数据
python reproduce.py all         # 依次执行以上全部
```

| 实验 | 协议 | 产出文件 | 大致耗时 |
| --- | --- | --- | --- |
| `main` | 5 折 × 3 重复（NN 类仅 1 重复），train-noisy/test-clean | `results/exp_results_v2.json` | 数十小时（CPU） |
| `supp` | 5 折 × 1 重复 | `results/noise_robustness.json` | 数小时 |
| `sens` | 5 折 × 1 重复 | `results/tune_grid_v2.json`、`results/sens_alpha_knn.json` | 数小时 |
| `ablation` | 5 折 × 1 重复 | `results/ablation.json` | 数分钟 |
| `viz` | 30% 噪声单次 | `results/factor_viz.json` | 数分钟 |

> **无需重跑即可核对论文数字**：论文引用的全部结果 JSON 已随附于 `results/`，与正文中的表/图一一对应。

---

## 5. 数据集

| 数据集 | 样本 | 维度 | 类别 | 来源 |
| --- | --- | --- | --- | --- |
| Breast Cancer | 569 | 30 | 2 | scikit-learn |
| Wine | 178 | 13 | 3 | scikit-learn |
| Digits | 1797 | 64 | 10 | scikit-learn |
| Banknote | 1372 | 4 | 2 | OpenML（banknote-authentication） |
| Wall-Robot Navigation | 5456 | 24 | 4 | OpenML（wall-robot-navigation） |

前四个为主实验数据集，Wall-Robot Navigation 为补充验证用更大规模数据集。除 scikit-learn 内置数据外，其余通过 `sklearn.datasets.fetch_openml` 在线获取（首次运行需联网）。

---

## 6. 目录结构

```
fa-dawrf/
├── fa_dawrf.py          # 核心算法模块（FADAWRF 及全部基线，自包含）
├── reproduce.py         # 一键复现脚本（main/supp/sens/ablation/viz/all）
├── requirements.txt     # 依赖
├── LICENSE              # MIT 许可证
├── README.md            # 本文件
└── results/             # 论文引用的全部结果 JSON（随附，可直接核对）
    ├── exp_results_v2.json     # 主实验（含显著性 sig 与消融 ablation）
    ├── noise_robustness.json   # 非对称/实例相关噪声 + 更大规模数据集
    ├── tune_grid_v2.json       # δ 敏感度
    ├── sens_alpha_knn.json     # α×k_nn 联合敏感度
    ├── factor_viz.json         # 因子空间散点 + 密度异常分
    └── app_metrics.json        # 应用场景（纸币真伪鉴别）AUROC
```

---

## 7. 引用

若本工作对您的研究有帮助，请引用：

```bibtex
@article{fa_dawrf,
  title   = {抗标签噪声的密度感知加权随机森林算法},
  author  = {待补充},
  journal = {待补充},
  year    = {2026},
  note    = {投稿中}
}
```

---

## 8. 许可证

本项目采用 [MIT License](LICENSE)。
