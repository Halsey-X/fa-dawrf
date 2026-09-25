# -*- coding: utf-8 -*-
"""FA-DAWRF 论文实验复现脚本（相对路径，开箱即用）。

用法（在仓库根目录 fa-dawrf/ 下运行）：
    python reproduce.py main        # 主实验：4 数据集 × 5 噪声 × 10 方法 → results/exp_results_v2.json
    python reproduce.py supp        # 补充实验：非对称/实例相关噪声 + wall-robot-navigation → results/noise_robustness.json
    python reproduce.py sens        # 敏感度：δ 与 α×k_nn → results/tune_grid_v2.json / results/sens_alpha_knn.json
    python reproduce.py ablation    # 消融实验（breast_cancer，30% 噪声）
    python reproduce.py viz         # 因子空间可视化数据 → results/factor_viz.json
    python reproduce.py all         # 依次执行以上全部

说明：
- 协议统一为"训练注入噪声、测试保持干净"（train noisy, test clean）。
- 随机种子固定：主实验 5 折 StratifiedKFold(shuffle=True, random_state=42)，3 次重复，
  噪声种子 42/142/242，模型种子 42+rep*10+fold；神经网络类基线（Co-teaching、MentorNet）仅第 0 次重复。
- 结果 JSON 与论文完全一致；论文引用结果已随附于 results/（可直接核对，无需重跑）。
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from fa_dawrf import (
    FADAWRF, PlainWRF, BootstrapRF, ConfidentLearningRF, CoTeachingMLP, MentorNetMLP,
    add_label_noise, add_asymmetric_noise, add_instance_noise,
    load_datasets, load_wall_robot, evaluate, make_factory,
)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

DATASETS = load_datasets()
DNAME_CN = {"breast_cancer": "Breast Cancer（乳腺癌）", "wine": "Wine（葡萄酒）",
            "digits": "Digits（数字）", "banknote": "Banknote（纸币真伪鉴别）"}

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
METHODS = ["RF", "AdaBoost", "GBDT", "BalancedRF", "PlainWRF",
           "ConfidentLearningRF", "CoTeachingMLP", "MentorNetMLP", "BootstrapRF", "FA-DAWRF"]
METHOD_CN = {
    "RF": "标准随机森林", "AdaBoost": "AdaBoost", "GBDT": "GBDT",
    "BalancedRF": "均衡随机森林", "PlainWRF": "普通错分加权",
    "ConfidentLearningRF": "置信学习(CL)", "CoTeachingMLP": "协同教学(Co-teaching)",
    "MentorNetMLP": "MentorNet", "BootstrapRF": "自举鲁棒RF", "FA-DAWRF": "FA-DAWRF(本文)",
}
NEURAL = {"CoTeachingMLP", "MentorNetMLP"}

# FA-DAWRF 论文调优超参（FADAWRF 默认值已对齐，此处仅作文档说明）
FA_TUNED = dict(delta=5.0, alpha=1.0, k_nn=12, beta=0.5, gamma=0.5, eta=0.5,
                T=4, n_factors=5, n_estimators=100, max_weight=10.0)


# ============================================================
# 主实验
# ============================================================
def run_main():
    t0 = time.time()
    results, sig = {}, {}
    for dname, d in DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        print(f"[dataset] {dname}: {X.shape}, classes={len(np.unique(y))}", flush=True)
        results.setdefault(dname, {})
        sig.setdefault(dname, {})
        for nr in NOISE_LEVELS:
            nkey = str(nr)
            results[dname][nkey] = {}
            sig[dname][nkey] = {}
            agg = {m: {k: [] for k in ("acc", "f1", "auc")} for m in METHODS}
            sig_acc = {m: [] for m in METHODS}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(skf.split(X, y))
            for rep in range(3):
                y_noisy = add_label_noise(y, nr, random_state=42 + rep * 100) if nr > 0 else y.copy()
                for fi, (tr, te) in enumerate(folds):
                    for m in METHODS:
                        if m in NEURAL and rep != 0:
                            continue
                        rs = 42 + rep * 10 + fi
                        acc, f1, auc = evaluate(make_factory(m, rs, nr)(),
                                                X[tr], y_noisy[tr], X[te], y[te])
                        agg[m]["acc"].append(acc)
                        agg[m]["f1"].append(f1)
                        agg[m]["auc"].append(auc)
                        if rep == 0:
                            sig_acc[m].append(acc)
            for m in METHODS:
                results[dname][nkey][m] = {
                    "acc": [float(np.mean(agg[m]["acc"])), float(np.std(agg[m]["acc"]))],
                    "f1": [float(np.mean(agg[m]["f1"])), float(np.std(agg[m]["f1"]))],
                    "auc": [float(np.mean(agg[m]["auc"])), float(np.std(agg[m]["auc"]))],
                }
                sig[dname][nkey][m] = [float(x) for x in sig_acc[m]]
            print(f"  noise={nr:.1f} done. FA-DAWRF acc={results[dname][nkey]['FA-DAWRF']['acc'][0]:.4f}", flush=True)
    out = {
        "datasets": list(DATASETS.keys()), "dname_cn": DNAME_CN,
        "methods": METHODS, "method_cn": METHOD_CN,
        "noises": [str(x) for x in NOISE_LEVELS],
        "results": results, "sig": sig, "ablation": run_ablation(return_only=True),
        "meta": {"rf_reps": 3, "nn_folds": 5, "protocol": "train_noisy_test_clean",
                 "neural_methods": sorted(NEURAL)},
    }
    path = os.path.join(RES, "exp_results_v2.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[main] done in {time.time()-t0:.1f}s -> {path}", flush=True)


# ============================================================
# 消融实验
# ============================================================
def run_ablation(return_only=False):
    bc = DATASETS["breast_cancer"]
    X, y = np.asarray(bc.data, float), np.asarray(bc.target, int)
    y_n = add_label_noise(y, 0.3, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    folds = list(skf.split(X, y_n))
    ablation = {}

    def _run(tag, model):
        accs, f1s, aucs = [], [], []
        for tr, te in folds:
            a, f, u = evaluate(type(model)(**model.__dict__), X[tr], y_n[tr], X[te], y[te])
            accs.append(a)
            f1s.append(f)
            aucs.append(u)
        ablation[tag] = {
            "acc": [float(np.mean(accs)), float(np.std(accs))],
            "f1": [float(np.mean(f1s)), float(np.std(f1s))],
            "auc": [float(np.mean(aucs)), float(np.std(aucs))],
        }

    _run("full", FADAWRF(random_state=0))
    _run("wo_density", FADAWRF(ignore_density=True, random_state=0))
    _run("wo_boundary", FADAWRF(gamma=0.0, random_state=0))
    _run("wo_misclass", FADAWRF(alpha=0.0, random_state=0))
    for nf in (3, 5, 8):
        _run(f"nfactors_{nf}", FADAWRF(n_factors=nf, random_state=0))
    for Tt in (2, 4, 6):
        _run(f"T_{Tt}", FADAWRF(T=Tt, random_state=0))
    print("[ablation] done:", {k: round(v["acc"][0], 4) for k, v in ablation.items()}, flush=True)
    if not return_only:
        json.dump(ablation, open(os.path.join(RES, "ablation.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    return ablation


# ============================================================
# 补充实验：非对称/实例相关噪声 + 更大规模数据集
# ============================================================
SUPP_METHODS = ["RF", "AdaBoost", "GBDT", "BalancedRF", "PlainWRF",
                "ConfidentLearningRF", "BootstrapRF", "FA-DAWRF"]
SUPP_CN = {"RF": "RF", "AdaBoost": "AdaBoost", "GBDT": "GBDT", "BalancedRF": "均衡RF",
           "PlainWRF": "朴素错分加权", "ConfidentLearningRF": "CL", "BootstrapRF": "BootstrapRF",
           "FA-DAWRF": "FA-DAWRF(本文)"}


def run_supp():
    t0 = time.time()
    results, dname_cn = {}, {}
    for dname, d in DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        dname_cn[dname] = DNAME_CN[dname]
        results.setdefault(dname, {})
        for ntype in ("asymmetric", "instance"):
            results[dname][ntype] = {}
            for nr in (0.3,):
                agg = {m: [] for m in SUPP_METHODS}
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for fi, (tr, te) in enumerate(skf.split(X, y)):
                    yn = add_asymmetric_noise(y, nr, random_state=42 + fi) if ntype == "asymmetric" \
                        else add_instance_noise(y, X, nr, random_state=42 + fi)
                    for m in SUPP_METHODS:
                        a, f, u = evaluate(make_factory(m, 42 + fi, nr)(), X[tr], yn[tr], X[te], y[te])
                        agg[m].append(a)
                results[dname][ntype][str(nr)] = {
                    m: [float(np.mean(agg[m])), float(np.std(agg[m]))] for m in SUPP_METHODS}
                print(f"  [supp] {dname} {ntype} FA={results[dname][ntype][str(nr)]['FA-DAWRF'][0]:.4f}", flush=True)
    # 更大规模数据集 wall-robot-navigation（对称噪声 0.3/0.4）
    try:
        ldX, ldY = load_wall_robot()
        ld_name = "wall_robot"
        dname_cn[ld_name] = "Wall-Robot（墙边机器人导航，5456 样本/24 维/4 类）"
        results.setdefault(ld_name, {})
        results[ld_name]["symmetric"] = {}
        for nr in (0.3, 0.4):
            agg = {m: [] for m in SUPP_METHODS}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fi, (tr, te) in enumerate(skf.split(ldX, ldY)):
                yn = add_label_noise(ldY, nr, random_state=42 + fi) if nr > 0 else ldY.copy()
                for m in SUPP_METHODS:
                    a, f, u = evaluate(make_factory(m, 42 + fi, nr)(), ldX[tr], yn[tr], ldX[te], ldY[te])
                    agg[m].append(a)
            results[ld_name]["symmetric"][str(nr)] = {
                m: [float(np.mean(agg[m])), float(np.std(agg[m]))] for m in SUPP_METHODS}
            print(f"  [supp-large] {ld_name} n={nr} FA={results[ld_name]['symmetric'][str(nr)]['FA-DAWRF'][0]:.4f}", flush=True)
    except Exception as e:
        print("[warn] wall-robot-navigation 加载失败，跳过更大规模数据集:", e, flush=True)
    json.dump({"datasets": list(results.keys()), "dname_cn": dname_cn,
               "noise_types": ["asymmetric", "instance", "symmetric"],
               "rates": ["0.3", "0.4"], "methods": SUPP_METHODS, "method_cn": SUPP_CN,
               "results": results},
              open(os.path.join(RES, "noise_robustness.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[supp] done in {time.time()-t0:.1f}s -> noise_robustness.json", flush=True)


# ============================================================
# 敏感度实验：δ 与 α×k_nn
# ============================================================
def run_sens():
    t0 = time.time()
    # δ 敏感度（5 折单次，噪声 0/0.2/0.3/0.4）
    deltas = [1.5, 2.5, 3.5, 5.0]
    noises_sens = [0.0, 0.2, 0.3, 0.4]
    tune = {d: {} for d in DATASETS}
    for dname, d in DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
        for dl in deltas:
            tune[dname][str(dl)] = {}
            for nr in noises_sens:
                accs = []
                for fi, (tr, te) in enumerate(folds):
                    yn = add_label_noise(y, nr, random_state=42) if nr > 0 else y.copy()
                    m = FADAWRF(delta=dl, random_state=42 + fi)
                    a, _, _ = evaluate(m, X[tr], yn[tr], X[te], y[te])
                    accs.append(a)
                tune[dname][str(dl)][str(nr)] = float(np.mean(accs))
            print(f"  [delta] {dname} δ={dl}: { {str(n): round(tune[dname][str(dl)][str(n)],3) for n in noises_sens} }", flush=True)
    json.dump(tune, open(os.path.join(RES, "tune_grid_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # α × k_nn 联合敏感度（5 折单次，噪声 0.2/0.4）
    alphas = [0.5, 1.0, 2.0]
    knns = [7, 12]
    rates = [0.2, 0.4]
    s2 = {d: {} for d in DATASETS}
    for dname, d in DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
        for a in alphas:
            s2[dname][str(a)] = {str(k): {} for k in knns}
            for k in knns:
                for nr in rates:
                    accs = []
                    for fi, (tr, te) in enumerate(folds):
                        yn = add_label_noise(y, nr, random_state=42) if nr > 0 else y.copy()
                        m = FADAWRF(alpha=a, k_nn=k, random_state=42 + fi)
                        acc, _, _ = evaluate(m, X[tr], yn[tr], X[te], y[te])
                        accs.append(acc)
                    s2[dname][str(a)][str(k)][str(nr)] = float(np.mean(accs))
            print(f"  [a/k] {dname} α={a}: { {str(k): {str(nr): round(s2[dname][str(a)][str(k)][str(nr)],3) for nr in rates} for k in knns} }", flush=True)
    json.dump(s2, open(os.path.join(RES, "sens_alpha_knn.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[sens] done in {time.time()-t0:.1f}s -> tune_grid_v2.json / sens_alpha_knn.json", flush=True)


# ============================================================
# 因子空间可视化数据
# ============================================================
def run_viz():
    out = {}
    for dname in ("breast_cancer", "digits"):
        d = DATASETS[dname]
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        yn = add_label_noise(y, 0.3, random_state=42)
        flipped = (yn != y).astype(int)
        model = FADAWRF(random_state=0)
        model.fit(X, yn)
        F = model.fa.transform(model.scaler.transform(np.nan_to_num(X, nan=0.0)))
        N = model._intra_class_density(F, yn)
        out[dname] = {
            "F2": [[float(F[i, 0]), float(F[i, 1])] for i in range(F.shape[0])],
            "N": [float(x) for x in N],
            "flipped": [int(x) for x in flipped],
            "clean_label": [int(x) for x in y],
            "noisy_label": [int(x) for x in yn],
        }
    json.dump(out, open(os.path.join(RES, "factor_viz.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[viz] done -> factor_viz.json", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    dispatch = {"main": run_main, "supp": run_supp, "sens": run_sens,
                "ablation": run_ablation, "viz": run_viz, "all": None}
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    if cmd == "all":
        run_main()
        run_supp()
        run_sens()
        run_viz()
    else:
        dispatch[cmd]()
