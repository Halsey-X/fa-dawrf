# -*- coding: utf-8 -*-
"""应用章节补充：密度异常分数 N_i 对“真实噪声标签”的识别 AUROC。"""
import json
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import run_experiments as R

BASE = r"D:\WorkBuddy\2026-09-21-07-08-04"
DATASETS = {
    "breast_cancer": (load_breast_cancer().data, load_breast_cancer().target, "乳腺癌"),
    "wine": (load_wine().data, load_wine().target, "葡萄酒"),
    "digits": (load_digits().data, load_digits().target, "数字"),
}
NOISE = 0.3
res = {"auc": {}, "auc_w": {}, "order": [], "dname_cn": {}}
for dname, (X, y, cn) in DATASETS.items():
    y = np.asarray(y, int)
    res["order"].append(dname)
    res["dname_cn"][dname] = cn
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs_n, aucs_w = [], []
    for tr, te in skf.split(X, y):
        rng = np.random.RandomState(42)
        n = len(y)
        idx = rng.choice(n, int(NOISE * n), replace=False)
        yn = y.copy()
        for i in idx:
            yn[i] = rng.choice([c for c in np.unique(y) if c != y[i]])
        isn = np.zeros(n, bool)
        isn[idx] = True
        m = R.FADAWRF(random_state=0)
        m.fit(X[tr], yn[tr])
        F = m.fa.transform(m.scaler.transform(X[tr]))
        N = m._intra_class_density(F, yn[tr])
        w = m._last_w
        aucs_n.append(roc_auc_score(isn[tr].astype(int), N))
        aucs_w.append(roc_auc_score(isn[tr].astype(int), -w))  # 权重越低越可疑
    res["auc"][dname] = float(np.mean(aucs_n))
    res["auc_w"][dname] = float(np.mean(aucs_w))
    print(f"{dname}: AUROC(N)={res['auc'][dname]:.3f}  AUROC(w)={res['auc_w'][dname]:.3f}")

with open(BASE + r"\app_metrics.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("SAVED app_metrics.json")
