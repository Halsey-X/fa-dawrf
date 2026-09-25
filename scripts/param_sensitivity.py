# -*- coding: utf-8 -*-
"""参数敏感性分析：固定 alpha=1.0, k_nn=12，扫描门控强度 delta ∈ {1.5, 2.5, 3.5, 5.0}，
记录 FA-DAWRF 在四个数据集、0%~40% 对称噪声下的 Accuracy（5 折分层 CV，单次重复，与超参搜索协议一致）。
结果写入 tune_grid_v2.json，供论文 5.5 节“参数敏感性分析”制表。断点续跑。"""
import json, os, time
import numpy as np
import run_experiments_v2 as V
from sklearn.model_selection import StratifiedKFold

BASE = r"D:\WorkBuddy\2026-09-21-07-08-04"
OUT = os.path.join(BASE, "tune_grid_v2.json")
DELTAS = [1.5, 2.5, 3.5, 5.0]
NOISE = [0.0, 0.1, 0.2, 0.3, 0.4]
t0 = time.time()

res = {}
if os.path.exists(OUT):
    try:
        res = json.load(open(OUT, encoding="utf-8"))
        print(f"[resume] loaded {sum(1 for d in res for _ in res[d])} (ds,delta) entries", flush=True)
    except Exception as ex:
        print(f"[resume] failed ({ex})", flush=True)

for dname, d in V.DATASETS.items():
    X = np.asarray(d.data, float); y = np.asarray(d.target, int)
    skf = StratifiedKFold(5, shuffle=True, random_state=42); folds = list(skf.split(X, y))
    res.setdefault(dname, {})
    for delta in DELTAS:
        if delta in res[dname] and all(str(n) in res[dname][delta] for n in NOISE):
            continue
        res[dname][delta] = {}
        for nr in NOISE:
            accs = []
            for fi, (tr, te) in enumerate(folds):
                yn = V.R.add_label_noise(y, nr, 42) if nr > 0 else y.copy()
                m = V.R.FADAWRF(delta=delta, alpha=1.0, k_nn=12, random_state=42 + fi)
                m.fit(X[tr], yn[tr])
                p = m.predict_proba(X[te]); p = np.nan_to_num(p, nan=1.0 / p.shape[1])
                accs.append((np.argmax(p, 1) == y[te]).mean())
            res[dname][delta][str(nr)] = float(np.mean(accs))
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  {dname} delta={delta}: " + ", ".join(f"{n}:{res[dname][delta][str(n)]:.3f}" for n in NOISE)
              + f"  (elapsed {time.time()-t0:.1f}s)", flush=True)

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("PARAM SENSITIVITY DONE ->", OUT, flush=True)
