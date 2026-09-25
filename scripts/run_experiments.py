import json
import time
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

warnings.filterwarnings("ignore")


# ---------------- 标签噪声 ----------------
def add_label_noise(y, noise_rate, random_state=42):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    y_noisy = y.copy()
    n = len(y)
    n_noise = int(noise_rate * n)
    idx = rng.choice(n, n_noise, replace=False)
    for i in idx:
        choices = [c for c in np.unique(y) if c != y[i]]
        y_noisy[i] = rng.choice(choices)
    return y_noisy


# ---------------- FA-DAWRF ----------------
class FADAWRF:
    def __init__(self, n_factors=5, n_estimators=100, T=4, k_nn=10,
                 alpha=1.0, beta=0.5, gamma=0.5, delta=1.0, eta=0.5,
                 max_weight=10.0, random_state=42, ignore_density=False,
                 density_space="factor"):
        self.n_factors = n_factors
        self.n_estimators = n_estimators
        self.T = T
        self.k_nn = k_nn
        self.alpha, self.beta, self.gamma, self.delta, self.eta = alpha, beta, gamma, delta, eta
        self.max_weight = max_weight
        self.random_state = random_state
        self.ignore_density = ignore_density
        self.density_space = density_space  # "factor"（默认）或 "orig"（原始标准化空间）

    def _intra_class_density(self, F, y):
        """同类（带噪标签）内 k 近邻密度：被翻转标签的样本在所属标注类中缺乏近邻，密度低、异常分数高。"""
        n = F.shape[0]
        d = np.zeros(n)
        for c in np.unique(y):
            mask = (y == c)
            Fc = F[mask]
            if len(Fc) <= 1:
                d[mask] = 0.0
                continue
            kk = min(self.k_nn, len(Fc) - 1)
            nn = NearestNeighbors(n_neighbors=kk).fit(Fc)
            dist, _ = nn.kneighbors(Fc)
            mean_dist = dist[:, 1:].mean(axis=1) + 1e-12   # 排除自身
            d[mask] = 1.0 / mean_dist
        dmin, dmax = d.min(), d.max()
        d_norm = (d - dmin) / (dmax - dmin + 1e-12)
        N = 1 - d_norm
        return N

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        self.fa = FactorAnalysis(n_components=self.n_factors, random_state=self.random_state)
        F = self.fa.fit_transform(Xs)
        n = X.shape[0]
        w = np.ones(n) / n
        self.models = []
        for t in range(self.T):
            rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                        oob_score=True, bootstrap=True,
                                        random_state=self.random_state + t, n_jobs=-1)
            rf.fit(Xs, y, sample_weight=w)
            self.models.append(rf)
            proba = rf.oob_decision_function_
            if proba is None:
                proba = rf.predict_proba(Xs)
            proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
            proba = proba / proba.sum(axis=1, keepdims=True)
            C = proba.shape[1]
            pred = np.argmax(proba, axis=1)
            e = (pred != y).astype(float)
            H = -np.sum(proba * np.log(proba + 1e-12), axis=1) / np.log(C)
            maxp = np.max(proba, axis=1)
            B = 1 - (maxp - 1.0 / C) / (1.0 - 1.0 / C + 1e-12)
            if self.ignore_density:
                N = np.zeros(n)
            else:
                # 密度估计所用表征：默认在因子空间，可选在原始标准化空间（用于隔离因子分析贡献）
                F_density = Xs if self.density_space == "orig" else F
                N = self._intra_class_density(F_density, y)
            # 门控式难例分数：对高密度的“干净难例”升权，对低密度的“噪声异常”降权
            gate = 1.0 - N
            s = (self.alpha * e + self.beta * H + self.gamma * B) * gate - self.delta * N
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            w = w * np.exp(self.eta * s)
            w = np.clip(w, 1e-8, self.max_weight)
            w = w / w.sum()
        self._last_w = w.copy()
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, float))
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.mean([rf.predict_proba(Xs) for rf in self.models], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------- 普通错分加权 RF（仅按 e 加权，作为消融/基线） ----------------
class PlainWRF:
    """仅使用错分指示 e 进行加权的随机森林，用于隔离密度感知的贡献。"""
    def __init__(self, n_estimators=100, T=4, eta=0.5, max_weight=10.0, random_state=42):
        self.n_estimators = n_estimators
        self.T = T
        self.eta = eta
        self.max_weight = max_weight
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n = X.shape[0]
        w = np.ones(n) / n
        self.models = []
        for t in range(self.T):
            rf = RandomForestClassifier(n_estimators=self.n_estimators, oob_score=True,
                                        bootstrap=True, random_state=self.random_state + t, n_jobs=-1)
            rf.fit(Xs, y, sample_weight=w)
            self.models.append(rf)
            proba = rf.oob_decision_function_
            if proba is None:
                proba = rf.predict_proba(Xs)
            proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
            proba = proba / proba.sum(axis=1, keepdims=True)
            pred = np.argmax(proba, axis=1)
            e = (pred != y).astype(float)
            e = np.nan_to_num(e, nan=0.0)
            w = w * np.exp(self.eta * e)
            w = np.clip(w, 1e-8, self.max_weight)
            w = w / w.sum()
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, float))
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.mean([rf.predict_proba(Xs) for rf in self.models], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------- 评估 ----------------
def evaluate(model_factory, Xtr, ytr, Xte, yte):
    model = model_factory()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    C = proba.shape[1]
    if C == 2:
        auc = roc_auc_score(yte, proba[:, 1])
    else:
        auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
    return acc, f1, auc


def make_factory(name, random_state):
    if name == "RF":
        return lambda: RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    if name == "AdaBoost":
        return lambda: AdaBoostClassifier(n_estimators=100, random_state=random_state)
    if name == "GBDT":
        return lambda: GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    if name == "BalancedRF":
        return lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                              random_state=random_state, n_jobs=-1)
    if name == "PlainWRF":
        return lambda: PlainWRF(n_estimators=100, T=4, eta=0.5, random_state=random_state)
    if name == "FA-DAWRF":
        return lambda: FADAWRF(n_factors=5, n_estimators=100, T=4, k_nn=10,
                               alpha=1.0, beta=0.5, gamma=0.5, delta=1.0, eta=0.5,
                               max_weight=10.0, random_state=random_state)
    raise ValueError(name)


METHODS = ["RF", "AdaBoost", "GBDT", "BalancedRF", "PlainWRF", "FA-DAWRF"]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
DATASETS = {
    "breast_cancer": load_breast_cancer(),
    "wine": load_wine(),
    "digits": load_digits(),
}

if __name__ == "__main__":
    results = {}
    t0 = time.time()
    for dname, d in DATASETS.items():
        X, y = d.data, d.target
        print(f"[dataset] {dname}: {X.shape}, classes={len(np.unique(y))}", flush=True)
        results[dname] = {}
        for nr in NOISE_LEVELS:
            results[dname][str(nr)] = {}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # 三次重复
            folds_acc, folds_f1, folds_auc = {m: [] for m in METHODS}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
            for rep in range(3):
                y_noisy = add_label_noise(y, nr, random_state=42 + rep * 100) if nr > 0 else y.copy()
                for fi, (tr, te) in enumerate(skf.split(X, y_noisy)):
                    for m in METHODS:
                        try:
                            rs = 42 + rep * 10 + fi
                            acc, f1, auc = evaluate(make_factory(m, rs),
                                                    X[tr], y_noisy[tr], X[te], y[te])
                            folds_acc[m].append(acc)
                            folds_f1[m].append(f1)
                            folds_auc[m].append(auc)
                        except Exception as ex:
                            print(f"  ERR {dname} nr={nr} {m}: {ex}", flush=True)
            for m in METHODS:
                results[dname][str(nr)][m] = {
                    "acc": [float(np.mean(folds_acc[m])), float(np.std(folds_acc[m]))],
                    "f1": [float(np.mean(folds_f1[m])), float(np.std(folds_f1[m]))],
                    "auc": [float(np.mean(folds_auc[m])), float(np.std(folds_auc[m]))],
                }
            print(f"  noise={nr:.1f} done. FA-DAWRF acc={results[dname][str(nr)]['FA-DAWRF']['acc'][0]:.4f}", flush=True)

    # ---------------- 消融实验（breast_cancer，噪声 0.2） ----------------
    print("[ablation] running ...", flush=True)
    abX, aby = DATASETS["breast_cancer"].data, DATASETS["breast_cancer"].target
    aby_n = add_label_noise(aby, 0.3, random_state=42)
    skf_ab = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    ablation = {}


    def ab_factory(builder):
        def _f():
            return builder
        return _f


    def run_ablation(tag, builder):
        accs, f1s, aucs = [], [], []
        for tr, te in skf_ab.split(abX, aby_n):
            a, f, u = evaluate(ab_factory(builder), abX[tr], aby_n[tr], abX[te], aby[te])
            accs.append(a); f1s.append(f); aucs.append(u)
        ablation[tag] = {
            "acc": [float(np.mean(accs)), float(np.std(accs))],
            "f1": [float(np.mean(f1s)), float(np.std(f1s))],
            "auc": [float(np.mean(aucs)), float(np.std(aucs))],
        }


    run_ablation("full", FADAWRF(random_state=0))
    run_ablation("wo_density", FADAWRF(ignore_density=True, random_state=0))   # 去掉密度项（N=0，退化为纯置信/错分加权）
    run_ablation("wo_boundary", FADAWRF(gamma=0.0, random_state=0))     # 去掉边界项
    run_ablation("wo_misclass", FADAWRF(alpha=0.0, random_state=0))     # 去掉错分项
    for nf in [3, 5, 8]:
        run_ablation(f"nfactors_{nf}", FADAWRF(n_factors=nf, random_state=0))
    for Tt in [2, 4, 6]:
        run_ablation(f"T_{Tt}", FADAWRF(T=Tt, random_state=0))
    results["ablation"] = ablation
    print("[ablation] done:", {k: round(v["acc"][0], 4) for k, v in ablation.items()}, flush=True)

    print(f"ALL DONE in {time.time()-t0:.1f}s", flush=True)
    with open(r"D:\WorkBuddy\2026-09-21-07-08-04\exp_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved exp_results.json", flush=True)
