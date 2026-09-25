# -*- coding: utf-8 -*-
"""生成 v2 结果图（4 数据集 × 3 指标 = 12 张），png/eps/svg 三格式，
并分类归档到 output/figures/results/；同时把过程数据/脚本归档到 output/process_data/。"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:\WorkBuddy\2026-09-21-07-08-04"
OUT_RES = os.path.join(BASE, "output", "figures", "results")
os.makedirs(OUT_RES, exist_ok=True)

with open(os.path.join(BASE, "exp_results_v2.json"), "r", encoding="utf-8") as f:
    RES = json.load(f)

DATASETS = RES["datasets"]
NOISES = [float(x) for x in RES["noises"]]
METHODS = RES["methods"]

# 图表用英文名（避免 CJK 字体缺失）
CHART_NAME = {"RF": "RF", "AdaBoost": "AdaBoost", "GBDT": "GBDT",
              "BalancedRF": "Balanced RF", "PlainWRF": "PlainWRF",
              "ConfidentLearningRF": "CL", "CoTeachingMLP": "Co-teaching",
              "MentorNetMLP": "MentorNet", "FA-DAWRF": "FA-DAWRF",
              "BootstrapRF": "BootstrapRF"}
# 颜色（FA-DAWRF 红，其余中性）
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
           "#17becf", "#bcbd22", "#7f7f7f", "#C00000"]
COLOR = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(METHODS)}
COLOR["FA-DAWRF"] = "#C00000"   # 本文方法固定红色
COLOR["BootstrapRF"] = "#e377c2"  # 表格专用鲁棒基线，醒目品红

METRICS = [("acc", "Accuracy"), ("f1", "F1-macro"), ("auc", "AUC")]

def plot_one(d, mkey, mlabel):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = NOISES
    for m in METHODS:
        ys = [RES["results"][d][str(n)][m][mkey][0] * 100 for n in NOISES]
        ss = [RES["results"][d][str(n)][m][mkey][1] * 100 for n in NOISES]
        is_fa = (m == "FA-DAWRF")
        ax.plot(x, ys, marker="o", color=COLOR[m], linewidth=2.6 if is_fa else 1.3,
                markersize=6 if is_fa else 3.5, label=CHART_NAME[m],
                zorder=10 if is_fa else 1)
        if is_fa:
            ax.fill_between(x, [a - b for a, b in zip(ys, ss)], [a + b for a, b in zip(ys, ss)],
                            color=COLOR[m], alpha=0.15, zorder=0)
    ax.set_xlabel("Label noise rate"); ax.set_ylabel(mlabel + " (%)")
    ax.set_title(f"{RES['dname_cn'][d]}", fontsize=11)
    ax.set_xticks(NOISES); ax.set_xticklabels([f"{int(n*100)}%" for n in NOISES])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    base = os.path.join(OUT_RES, f"{d}_{mkey}")
    for ext in ("png", "eps", "svg"):
        fig.savefig(base + "." + ext, dpi=200)
    plt.close(fig)


def plot_combined(d):
    """同一数据集的 4 个真实子图合并到 2x2 画布：
    (a) Accuracy、(b) F1-macro、(c) AUC 折线图 + (d) 各方法平均准确率柱状图；
    底部为共享图例，不占用子图单元格，版面更平衡。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8))
    ax = axes.ravel()
    sub_labels = ["(a) Accuracy", "(b) F1-macro", "(c) AUC", "(d) Mean Accuracy"]
    # 三个折线子图
    for idx in range(3):
        mkey, mlabel = METRICS[idx]
        a = ax[idx]
        for m in METHODS:
            ys = [RES["results"][d][str(n)][m][mkey][0] * 100 for n in NOISES]
            ss = [RES["results"][d][str(n)][m][mkey][1] * 100 for n in NOISES]
            is_fa = (m == "FA-DAWRF")
            a.plot(NOISES, ys, marker="o", color=COLOR[m],
                   linewidth=2.6 if is_fa else 1.3,
                   markersize=6 if is_fa else 3.5, label=CHART_NAME[m],
                   zorder=10 if is_fa else 1)
            if is_fa:
                a.fill_between(NOISES, [u - v for u, v in zip(ys, ss)],
                               [u + v for u, v in zip(ys, ss)],
                               color=COLOR[m], alpha=0.15, zorder=0)
        a.set_xlabel("Label noise rate"); a.set_ylabel(mlabel + " (%)")
        a.set_title(sub_labels[idx], fontsize=11, loc="left", pad=6)
        a.set_xticks(NOISES); a.set_xticklabels([f"{int(n*100)}%" for n in NOISES])
        a.grid(True, linestyle=":", alpha=0.5)
    # (d) 各方法平均准确率柱状图（跨 0~40% 噪声取均值，降序排列）
    means = {m: float(np.mean([RES["results"][d][str(n)][m]["acc"][0] * 100
                               for n in NOISES])) for m in METHODS}
    order = sorted(METHODS, key=lambda m: means[m], reverse=True)
    vals = [means[m] for m in order]
    cols = ["#C00000" if m == "FA-DAWRF" else COLOR[m] for m in order]
    ax[3].bar(range(len(order)), vals, color=cols, width=0.62, zorder=3)
    ax[3].set_xticks(range(len(order)))
    ax[3].set_xticklabels([CHART_NAME[m] for m in order], rotation=35,
                          ha="right", fontsize=7)
    ax[3].set_ylabel("Mean Accuracy (%)")
    ax[3].set_ylim(min(50, min(vals) - 5), 103)
    ax[3].set_title(sub_labels[3], fontsize=11, loc="left", pad=6)
    ax[3].grid(True, axis="y", linestyle=":", alpha=0.5)
    # 共享图例置于画布底部，ncol=5 自适应两行
    handles, lbls = ax[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.suptitle(RES["dname_cn"][d], fontsize=13, y=0.985)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    base = os.path.join(OUT_RES, f"combined_{d}")
    for ext in ("png", "eps", "svg"):
        fig.savefig(base + "." + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_factor_space():
    """图 7：因子空间二维投影散点 + 密度异常分数 N_i 分布（基于 factor_viz.json）。"""
    fv = os.path.join(BASE, "factor_viz.json")
    if not os.path.exists(fv):
        print("factor_viz.json 缺失，跳过图 7", flush=True)
        return
    data = json.load(open(fv, encoding="utf-8"))
    dname = "breast_cancer" if "breast_cancer" in data else list(data.keys())[0]
    d = data[dname]
    F2 = np.array(d["F2"]); N = np.array(d["N"]); flip = np.array(d["flipped"])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    ax.scatter(F2[flip == 0, 0], F2[flip == 0, 1], s=14, c="#1f77b4", alpha=0.5, label="干净样本")
    ax.scatter(F2[flip == 1, 0], F2[flip == 1, 1], s=28, c="#C00000", marker="x", label="被翻转标签样本")
    ax.set_xlabel("Factor 1"); ax.set_ylabel("Factor 2")
    ax.set_title(f"因子空间散点（{RES['dname_cn'].get(dname, dname)}，30% 对称噪声）", fontsize=11)
    ax.legend(fontsize=8, loc="best"); ax.grid(True, linestyle=":", alpha=0.4)
    ax = axes[1]
    ax.hist(N[flip == 0], bins=30, alpha=0.6, color="#1f77b4", label="干净样本", density=True)
    ax.hist(N[flip == 1], bins=30, alpha=0.6, color="#C00000", label="被翻转标签样本", density=True)
    ax.set_xlabel("密度异常分数 $N_i$"); ax.set_ylabel("密度")
    ax.set_title("密度异常分数分布对比", fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    base = os.path.join(OUT_RES, "factor_space")
    for ext in ("png", "svg", "eps"):
        fig.savefig(base + "." + ext, dpi=200)
    plt.close(fig)
    print("factor_space done", flush=True)


for d in DATASETS:
    for mkey, mlabel in METRICS:
        plot_one(d, mkey, mlabel)
        print("plot", d, mkey)
    plot_combined(d)
    print("combined", d)

plot_factor_space()

# ---------------- 过程数据/脚本归档 ----------------
PD = os.path.join(BASE, "output", "process_data")
os.makedirs(PD, exist_ok=True)
os.makedirs(os.path.join(PD, "scripts"), exist_ok=True)
for jf in ("exp_results_v2.json", "app_metrics.json", "sens_alpha_knn.json",
           "noise_robustness.json", "factor_viz.json"):
    if os.path.exists(os.path.join(BASE, jf)):
        shutil.copy(os.path.join(BASE, jf), os.path.join(PD, jf))
for sf in ("run_experiments_v2.py", "build_paper_v2.py", "make_exports_v2.py",
           "extend_experiments.py", "smoke_test.py", "run_experiments.py", "build_paper.py"):
    if os.path.exists(os.path.join(BASE, sf)):
        shutil.copy(os.path.join(BASE, sf), os.path.join(PD, "scripts", sf))
if os.path.exists(os.path.join(BASE, "source_content.txt")):
    shutil.copy(os.path.join(BASE, "source_content.txt"), os.path.join(PD, "source_content.txt"))

print("RESULT CHARTS + ARCHIVE DONE")
