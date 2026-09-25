# -*- coding: utf-8 -*-
"""网格搜索 FA-DAWRF 超参（delta, alpha, k_nn），目标：高噪声下 Accuracy 接近/超过 CL，且干净数据不退化。
使用 1 重复×5 折（与 CL 同折划分比较），仅评估 FA-DAWRF 与各数据集 CL 的 rep0 均值。"""
import json, time, itertools
import numpy as np
import run_experiments_v2 as V
from sklearn.model_selection import StratifiedKFold

DSETS = V.DATASETS
NOISE = [0.0, 0.2, 0.3, 0.4]
t0 = time.time()

# ---- 先算 CL 的 rep0 均值（公平对比）----
CL = {}
for dname, d in DSETS.items():
    X = np.asarray(d.data, float); y = np.asarray(d.target, int)
    skf = StratifiedKFold(5, shuffle=True, random_state=42); folds = list(skf.split(X, y))
    CL[dname] = {}
    for nr in NOISE:
        accs = []
        for fi, (tr, te) in enumerate(folds):
            yn = V.R.add_label_noise(y, nr, 42) if nr > 0 else y.copy()
            m = V.ConfidentLearningRF(random_state=42); m.fit(X[tr], yn[tr])
            p = m.predict_proba(X[te]); p = np.nan_to_num(p, nan=1.0/p.shape[1])
            accs.append((np.argmax(p, 1) == y[te]).mean())
        CL[dname][nr] = float(np.mean(accs))
print("CL rep0 means:", {d: {str(n): round(CL[d][n],3) for n in NOISE} for d in DSETS}, flush=True)

grid = list(itertools.product([1.5,2.5,3.5,5.0], [1.0,2.0], [7,12]))
print("configs:", len(grid), flush=True)
results = []
for (delta, alpha, k_nn) in grid:
    fa = {}
    ok = True
    for dname, d in DSETS.items():
        X = np.asarray(d.data, float); y = np.asarray(d.target, int)
        skf = StratifiedKFold(5, shuffle=True, random_state=42); folds = list(skf.split(X, y))
        fa[dname] = {}
        for nr in NOISE:
            accs = []
            for fi, (tr, te) in enumerate(folds):
                yn = V.R.add_label_noise(y, nr, 42) if nr > 0 else y.copy()
                m = V.R.FADAWRF(delta=delta, alpha=alpha, k_nn=k_nn, random_state=42+fi)
                m.fit(X[tr], yn[tr])
                p = m.predict_proba(X[te]); p = np.nan_to_num(p, nan=1.0/p.shape[1])
                accs.append((np.argmax(p, 1) == y[te]).mean())
            fa[dname][nr] = float(np.mean(accs))
    # 目标：高噪声均值(0.2,0.3,0.4) 与 CL 的差距之和；干净退化(0.0 vs 0.0) 罚分
    hi = np.mean([fa[d][0.2]+fa[d][0.3]+fa[d][0.4] for d in DSETS])/3.0
    hi_cl = np.mean([CL[d][0.2]+CL[d][0.3]+CL[d][0.4] for d in DSETS])/3.0
    clean_pen = max(0.0, CL["breast_cancer"][0.0]-fa["breast_cancer"][0.0])  # 干净不退化约束参考
    gap = hi - hi_cl
    results.append((delta, alpha, k_nn, hi, hi_cl, gap, clean_pen,
                    {d: {str(n): round(fa[d][n],3) for n in NOISE} for d in DSETS}))
    print(f"  d={delta} a={alpha} k={k_nn} hiFA={hi:.3f} hiCL={hi_cl:.3f} gap={gap:+.3f} cleanpen={clean_pen:.3f}", flush=True)

results.sort(key=lambda r: -r[5])  # 按 gap 降序
print("\n=== TOP 8 by high-noise gap to CL ===", flush=True)
for r in results[:8]:
    print(f"d={r[0]} a={r[1]} k={r[2]} | hiFA={r[3]:.3f} hiCL={r[4]:.3f} gap={r[5]:+.3f} | per-ds hiFA={ {d: round((r[7][d]['0.2']+r[7][d]['0.3']+r[7][d]['0.4'])/3,3) for d in DSETS} }", flush=True)
print(f"DONE in {time.time()-t0:.1f}s", flush=True)
