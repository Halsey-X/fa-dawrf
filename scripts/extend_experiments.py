# -*- coding: utf-8 -*-
"""FA-DAWRF 补充实验（回应中文核心严厉审稿意见 #3/#4/#5/#8/#10）。

产出文件（均落盘到项目根，供 build_paper_v2.py / make_exports_v2.py 读取）：
  - 把 BootstrapRF（表格专用鲁棒基线）并入主实验结果 exp_results_v2.json（更新 methods 数组 + 各 (d,n) 补算）
  - sens_alpha_knn.json        : α 与 k_nn 联合敏感度（5 折单次重复）
  - noise_robustness.json      : 非对称/实例相关噪声 + 更大规模数据集（wall-robot-navigation）的补充验证
  - factor_viz.json            : 因子空间散点 + N_i 分布可视化所需数据

协议：训练注入噪声、测试干净（train noisy, test clean）；随机种子固定。
"""
import json, os, time, warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import run_experiments as R
import run_experiments_v2 as V   # 复用 DATASETS / ConfidentLearningRF / FA_TUNED / evaluate_one

warnings.filterwarnings("ignore")
BASE = r"D:\WorkBuddy\2026-09-21-07-08-04"
t0 = time.time()


# ============================================================
# 1) 新基线 BootstrapRF（软自举，Reed et al. 2014 的表格化实现）
# ============================================================
class BootstrapRF:
    """软自举鲁棒随机森林：先用 OOB 概率估计样本可信度，对低置信（疑似错标）样本降权后重训。
    属标签噪声鲁棒学习中的样本加权路线，且天然适配表格数据（回应“稻草人”质疑 #4）。"""
    def __init__(self, n_estimators=100, beta=0.3, random_state=42):
        self.n_estimators = n_estimators
        self.beta = beta
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        base = RandomForestClassifier(n_estimators=self.n_estimators, oob_score=True,
                                      random_state=self.random_state, n_jobs=-1)
        base.fit(Xs, y)
        proba = base.oob_decision_function_
        if proba is None:
            proba = base.predict_proba(Xs)
        proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
        proba = proba / proba.sum(axis=1, keepdims=True)
        # 软自举权重：低置信（疑似噪声）样本降权，高置信样本保持 ~1
        p_given = np.maximum(proba[np.arange(len(y)), y.astype(int)], 1e-6)
        w = (1.0 - self.beta) + self.beta * p_given
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          random_state=self.random_state, n_jobs=-1)
        self.clf.fit(Xs, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, float))
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return self.clf.predict_proba(Xs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# 2) 更真实的噪声注入：非对称（依类翻转）+ 实例相关（置信依赖）
# ============================================================
def add_asymmetric_noise(y, noise_rate, random_state=42):
    """非对称噪声：每个类 i 以噪声率翻转到“下一类” (i+1)%C，模拟易混淆类的系统性错标。"""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y); y_n = y.copy(); C = len(np.unique(y))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_flip = int(noise_rate * len(idx))
        if n_flip == 0:
            continue
        pick = rng.choice(idx, n_flip, replace=False)
        y_n[pick] = (c + 1) % C
    return y_n


def add_instance_noise(y, X, noise_rate, random_state=42):
    """实例相关噪声：以“干净模型预测正确概率”为难易度，越难的样本越可能被错标，
    翻转到当前最易混淆的类别。属特征依赖的实例相关噪声（instance-dependent noise）。"""
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, float); y = np.asarray(y)
    Xs = StandardScaler().fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    clean = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(Xs, y)
    p = clean.predict_proba(Xs); p = np.nan_to_num(p, nan=1.0 / p.shape[1])
    y_n = y.copy()
    for i in range(len(y)):
        flip_p = noise_rate * (1.0 - p[i, int(y[i])])   # 越难越易翻
        if rng.rand() < flip_p:
            others = [c for c in range(p.shape[1]) if c != int(y[i])]
            y_n[i] = others[int(np.argmax(p[i, others]))]
    return y_n


# ============================================================
# 3) 更大规模/高维数据集（回应 #5）
# ============================================================
def load_larger_dataset():
    """优先 wall-robot-navigation（5456 样本、24 维、4 类），失败则回退 wine_quality。"""
    try:
        d = fetch_openml("wall-robot-navigation", version=1, as_frame=False, parser="auto")
        X, y = np.asarray(d.data, dtype=float), LabelEncoder().fit_transform(np.asarray(d.target))
        return "wall_robot", X, y, "Wall-Robot（墙边机器人导航，5456 样本/24 维/4 类）"
    except Exception as e:
        print("[warn] wall-robot fetch failed:", e, flush=True)
        try:
            d = fetch_openml("wine_quality", version=2, as_frame=False, parser="auto")
            X, y = np.asarray(d.data, dtype=float), LabelEncoder().fit_transform(np.asarray(d.target))
            return "wine_quality", X, y, "Wine-Quality（葡萄酒质量，4898 样本/11 维/7 类）"
        except Exception as e2:
            print("[warn] wine_quality fetch failed:", e2, flush=True)
            return None, None, None, None


# ============================================================
# 4) 评估与工厂
# ============================================================
def make_factory_supp(name, rs, nr):
    if name == "RF":
        return lambda: RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1)
    if name == "AdaBoost":
        return lambda: AdaBoostClassifier(n_estimators=100, random_state=rs)
    if name == "GBDT":
        return lambda: GradientBoostingClassifier(n_estimators=100, random_state=rs)
    if name == "BalancedRF":
        return lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                              random_state=rs, n_jobs=-1)
    if name == "PlainWRF":
        return lambda: R.PlainWRF(n_estimators=100, T=4, eta=0.5, random_state=rs)
    if name == "ConfidentLearningRF":
        return lambda: V.ConfidentLearningRF(random_state=rs)
    if name == "BootstrapRF":
        return lambda: BootstrapRF(random_state=rs)
    if name == "FA-DAWRF":
        return lambda: R.FADAWRF(random_state=rs, **V.FA_TUNED)
    raise ValueError(name)


def evaluate_factory(factory, Xtr, ytr, Xte, yte):
    m = factory(); m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte); proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")
    C = proba.shape[1]
    auc = roc_auc_score(yte, proba[:, 1]) if C == 2 else roc_auc_score(yte, proba, multi_class="ovr", average="macro")
    return acc, f1, auc


# ============================================================
# 5) (A) BootstrapRF 并入主实验（4 数据集 × 5 噪声 × 15 运行）
# ============================================================
def run_bootstrap_main():
    exp_path = os.path.join(BASE, "exp_results_v2.json")
    exp = json.load(open(exp_path, encoding="utf-8"))
    methods = exp["methods"]
    if "BootstrapRF" not in methods:
        methods = methods[:-1] + ["BootstrapRF", "FA-DAWRF"] if methods[-1] == "FA-DAWRF" else methods + ["BootstrapRF"]
    datasets = exp["datasets"]; noises = exp["noises"]
    for dname in datasets:
        d = V.DATASETS[dname]
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        for n in noises:
            key = str(n)
            if "BootstrapRF" in exp["results"][dname][key]:
                continue
            accs, f1s, aucs, sig_acc = [], [], [], []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(skf.split(X, y))
            for rep in range(3):
                yn = R.add_label_noise(y, float(n), random_state=42 + rep * 100) if float(n) > 0 else y.copy()
                for fi, (tr, te) in enumerate(folds):
                    rs = 42 + rep * 10 + fi
                    a, f, u = evaluate_factory(make_factory_supp("BootstrapRF", rs, float(n)),
                                               X[tr], yn[tr], X[te], y[te])
                    accs.append(a); f1s.append(f); aucs.append(u)
                    if rep == 0:
                        sig_acc.append(a)
            exp["results"][dname][key]["BootstrapRF"] = {
                "acc": [float(np.mean(accs)), float(np.std(accs))],
                "f1": [float(np.mean(f1s)), float(np.std(f1s))],
                "auc": [float(np.mean(aucs)), float(np.std(aucs))]}
            exp["sig"][dname][key]["BootstrapRF"] = [float(x) for x in sig_acc]
            print(f"  [BootstrapRF] {dname} n={n}: acc={exp['results'][dname][key]['BootstrapRF']['acc'][0]:.4f}", flush=True)
    exp["methods"] = methods
    json.dump(exp, open(exp_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[A] BootstrapRF merged into exp_results_v2.json", flush=True)


# ============================================================
# 6) (B) α/k_nn 联合敏感度（5 折单次重复）
# ============================================================
def run_alpha_knn_sens():
    out = {d: {} for d in V.DATASETS}
    alphas = [0.5, 1.0, 2.0]; knns = [7, 12]; rates = [0.2, 0.4]
    for dname, d in V.DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        for a in alphas:
            out[dname][str(a)] = {str(k): {} for k in knns}
            for k in knns:
                for nr in rates:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    accs = []
                    for fi, (tr, te) in enumerate(skf.split(X, y)):
                        yn = R.add_label_noise(y, nr, random_state=42) if nr > 0 else y.copy()
                        m = R.FADAWRF(random_state=42 + fi, k_nn=k, alpha=a, **{kk: vv for kk, vv in V.FA_TUNED.items() if kk not in ("alpha", "k_nn")})
                        a_, f_, u_ = evaluate_factory(lambda: m, X[tr], yn[tr], X[te], y[te])
                        accs.append(a_)
                    out[dname][str(a)][str(k)][str(nr)] = float(np.mean(accs))
                    print(f"  [a/k] {dname} a={a} k={k} n={nr}: {np.mean(accs):.4f}", flush=True)
    json.dump(out, open(os.path.join(BASE, "sens_alpha_knn.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[B] sens_alpha_knn.json done", flush=True)


# ============================================================
# 7) (C) 补充验证：非对称/实例相关噪声 + 更大规模数据集（5 折单次重复）
# ============================================================
SUPP_METHODS = ["RF", "AdaBoost", "GBDT", "BalancedRF", "PlainWRF",
                "ConfidentLearningRF", "BootstrapRF", "FA-DAWRF"]
SUPP_CN = {"RF": "RF", "AdaBoost": "AdaBoost", "GBDT": "GBDT", "BalancedRF": "均衡RF",
           "PlainWRF": "朴素错分加权", "ConfidentLearningRF": "CL", "BootstrapRF": "BootstrapRF",
           "FA-DAWRF": "FA-DAWRF(本文)"}


def run_supplementary():
    results = {}
    dname_cn = {}
    # 4 个原数据集 × {asymmetric, instance} × {0.3}
    for dname, d in V.DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        dname_cn[dname] = V.DNAME_CN[dname]
        results.setdefault(dname, {})
        for ntype in ("asymmetric", "instance"):
            results[dname][ntype] = {}
            for nr in (0.3,):
                agg = {m: [] for m in SUPP_METHODS}
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for fi, (tr, te) in enumerate(skf.split(X, y)):
                    if ntype == "asymmetric":
                        yn = add_asymmetric_noise(y, nr, random_state=42 + fi)
                    else:
                        yn = add_instance_noise(y, X, nr, random_state=42 + fi)
                    for m in SUPP_METHODS:
                        a, f, u = evaluate_factory(make_factory_supp(m, 42 + fi, nr),
                                                   X[tr], yn[tr], X[te], y[te])
                        agg[m].append(a)
                results[dname][ntype][str(nr)] = {
                    m: [float(np.mean(agg[m])), float(np.std(agg[m]))] for m in SUPP_METHODS}
                print(f"  [supp] {dname} {ntype} n={nr} FA={results[dname][ntype][str(nr)]['FA-DAWRF'][0]:.4f}", flush=True)
    # 更大规模数据集：对称噪声 0.3/0.4
    ld_name, ldX, ldY, ld_cn = load_larger_dataset()
    if ldX is not None:
        dname_cn[ld_name] = ld_cn
        results.setdefault(ld_name, {})
        results[ld_name]["symmetric"] = {}
        for nr in (0.3, 0.4):
            agg = {m: [] for m in SUPP_METHODS}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fi, (tr, te) in enumerate(skf.split(ldX, ldY)):
                yn = R.add_label_noise(ldY, nr, random_state=42 + fi) if nr > 0 else ldY.copy()
                for m in SUPP_METHODS:
                    a, f, u = evaluate_factory(make_factory_supp(m, 42 + fi, nr),
                                               ldX[tr], yn[tr], ldX[te], ldY[te])
                    agg[m].append(a)
            results[ld_name]["symmetric"][str(nr)] = {
                m: [float(np.mean(agg[m])), float(np.std(agg[m]))] for m in SUPP_METHODS}
            print(f"  [supp-large] {ld_name} n={nr} FA={results[ld_name]['symmetric'][str(nr)]['FA-DAWRF'][0]:.4f}", flush=True)
    json.dump({"datasets": list(results.keys()), "dname_cn": dname_cn,
               "noise_types": ["asymmetric", "instance", "symmetric"],
               "rates": ["0.3", "0.4"], "methods": SUPP_METHODS, "method_cn": SUPP_CN,
               "results": results},
              open(os.path.join(BASE, "noise_robustness.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[C] noise_robustness.json done", flush=True)


# ============================================================
# 8) (D) 因子空间可视化数据（breast_cancer + digits，30% 对称噪声）
# ============================================================
def run_factor_viz():
    out = {}
    for dname in ("breast_cancer", "digits"):
        d = V.DATASETS[dname]
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        yn = R.add_label_noise(y, 0.3, random_state=42)
        flipped = (yn != y).astype(int)
        model = R.FADAWRF(random_state=0, **V.FA_TUNED)
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
        print(f"  [factor] {dname}: N mean(flipped)={np.mean([N[i] for i in range(len(N)) if flipped[i]==1]):.3f} "
              f"vs clean={np.mean([N[i] for i in range(len(N)) if flipped[i]==0]):.3f}", flush=True)
    json.dump(out, open(os.path.join(BASE, "factor_viz.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[D] factor_viz.json done", flush=True)


if __name__ == "__main__":
    print("=== extend_experiments start ===", flush=True)
    run_bootstrap_main()
    run_alpha_knn_sens()
    run_supplementary()
    run_factor_viz()
    print(f"ALL EXTEND DONE in {time.time()-t0:.1f}s", flush=True)
