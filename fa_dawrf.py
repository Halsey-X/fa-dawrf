# -*- coding: utf-8 -*-
"""FA-DAWRF：抗标签噪声的密度感知加权随机森林（核心算法模块）。

本文件自包含（仅依赖 numpy + scikit-learn），可直接 ``from fa_dawrf import FADAWRF`` 使用，
也可作为实验脚本的公共依赖。

论文：抗标签噪声的密度感知加权随机森林算法（中文核心期刊投稿稿）。

核心思想
--------
标签噪声会破坏"同类内邻域"假设：被错误标注的样本在其（带噪）标注类中通常缺乏近邻、
密度异常低。FA-DAWRF 在因子分析（Factor Analysis）所得低维隐空间上估计每个样本的
"类内 k 近邻密度异常分" N_i，并构造门控式难例分数：

    s_i = (α·e_i + β·H_i + γ·B_i)·(1 − N_i) − δ·N_i

其中 e_i 为袋外(OOB)错分指示，H_i 为归一化熵（预测不确定性），B_i 为边界距离项，
N_i 为密度异常分。随后以 s_i 对样本做乘法式重加权（w ← w·exp(η·s)），迭代 T 轮，
使高密度的"干净难例"升权、低密度的"噪声异常"降权。

符号约定：α、β、γ、δ、η、T、n_factors、n_estimators、k_nn、max_weight。
数值稳定常数 ε₀ = 1e-12（见公式中所有 ``+1e-12`` 项）。
"""
import numpy as np

from sklearn.datasets import load_breast_cancer, load_wine, load_digits, fetch_openml
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ============================================================
# 1) 标签噪声注入
# ============================================================

def add_label_noise(y, noise_rate, random_state=42):
    """对称（随机）标签噪声：随机挑选 noise_rate 比例的样本，翻转到任意其它类。

    参数
    ----
    y : 干净的类别标签数组。
    noise_rate : 噪声率，∈ [0, 1]。
    random_state : 随机种子。
    """
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


def add_asymmetric_noise(y, noise_rate, random_state=42):
    """非对称（依类）标签噪声：每个类 i 以噪声率翻转到"下一类" (i+1) % C，
    模拟易混淆类之间的系统性错标。

    参数
    ----
    y : 干净的类别标签数组。
    noise_rate : 噪声率（按类内比例）。
    random_state : 随机种子。
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    y_n = y.copy()
    C = len(np.unique(y))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_flip = int(noise_rate * len(idx))
        if n_flip == 0:
            continue
        pick = rng.choice(idx, n_flip, replace=False)
        y_n[pick] = (c + 1) % C
    return y_n


def add_instance_noise(y, X, noise_rate, random_state=42):
    """实例相关（置信度依赖）噪声：以干净模型预测正确概率 p 为难易度，
    越难的样本越可能被错标，翻转到当前最易混淆的类别（特征依赖的实例相关噪声）。

    参数
    ----
    y : 干净的类别标签数组。
    X : 特征矩阵。
    noise_rate : 基础噪声率。
    random_state : 随机种子。
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, float)
    y = np.asarray(y)
    Xs = StandardScaler().fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    clean = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(Xs, y)
    p = clean.predict_proba(Xs)
    p = np.nan_to_num(p, nan=1.0 / p.shape[1])
    y_n = y.copy()
    for i in range(len(y)):
        flip_p = noise_rate * (1.0 - p[i, int(y[i])])   # 越难越易翻
        if rng.rand() < flip_p:
            others = [c for c in range(p.shape[1]) if c != int(y[i])]
            y_n[i] = others[int(np.argmax(p[i, others]))]
    return y_n


# ============================================================
# 2) FA-DAWRF（本文方法）
# ============================================================

class FADAWRF:
    """密度感知加权随机森林（Density-Aware Weighted Random Forest）。

    参数（默认值 = 论文调优后的最终设置）
    ------------------------------------
    n_factors : int = 5        因子分析隐因子个数 m。
    n_estimators : int = 100   每轮随机森林的树数。
    T : int = 4                重加权迭代轮数。
    k_nn : int = 12            类内密度估计所用近邻数。
    alpha : float = 1.0        错分项 e_i 的系数。
    beta : float = 0.5         熵项 H_i 的系数。
    gamma : float = 0.5        边界项 B_i 的系数。
    delta : float = 5.0        密度异常惩罚项 N_i 的系数（主导项）。
    eta : float = 0.5          乘法式重加权的步长。
    max_weight : float = 10.0  样本权重上界。
    random_state : int = 42    随机种子。
    ignore_density : bool = False  置 True 关闭密度项（消融：退化为纯置信/错分加权）。
    density_space : str = "factor" 密度估计表征空间："factor"（默认）或 "orig"（原始标准化空间）。
    """

    def __init__(self, n_factors=5, n_estimators=100, T=4, k_nn=12,
                 alpha=1.0, beta=0.5, gamma=0.5, delta=5.0, eta=0.5,
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
        self.density_space = density_space

    def _intra_class_density(self, F, y):
        """同类（带噪标签）内 k 近邻密度 → 密度异常分 N ∈ [0,1]。

        被翻转标签的样本在所属标注类中缺乏近邻、密度低、异常分数高。
        """
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
            mean_dist = dist[:, 1:].mean(axis=1) + 1e-12   # 排除自身（ε₀）
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
                F_density = Xs if self.density_space == "orig" else F
                N = self._intra_class_density(F_density, y)
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


# ============================================================
# 3) 普通错分加权随机森林 PlainWRF（消融 / 基线）
# ============================================================

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


# ============================================================
# 4) 软自举鲁棒随机森林 BootstrapRF（表格专用鲁棒基线）
# ============================================================

class BootstrapRF:
    """软自举（Soft Bootstrapping, Reed et al. 2014 的表格化实现）。

    先用 OOB 概率估计样本可信度，对低置信（疑似错标）样本降权后重训：
        w = (1 − β) + β · p(y_i | x_i)
    """

    def __init__(self, n_estimators=100, beta=0.3, random_state=42):
        self.n_estimators = n_estimators
        self.beta = beta
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
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
# 5) 置信学习 ConfidentLearningRF（SOTA 基线）
# ============================================================

class ConfidentLearningRF:
    """置信学习 (Confident Learning, Northcutt et al. 2021) 的随机森林实现。

    先用交叉验证得到袋外预测概率，识别"预测类别 != 给定标签"的可疑样本并降权，
    再用加权随机森林重训。
    """

    def __init__(self, n_estimators=100, cv=5, random_state=42, demote=0.1):
        self.n_estimators = n_estimators
        self.cv = cv
        self.random_state = random_state
        self.demote = demote

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        base = RandomForestClassifier(n_estimators=self.n_estimators,
                                      random_state=self.random_state, n_jobs=-1)
        proba = cross_val_predict(base, X, y, cv=self.cv,
                                  method="predict_proba", n_jobs=-1)
        proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
        proba = proba / proba.sum(axis=1, keepdims=True)
        pred = np.argmax(proba, axis=1)
        w = np.ones(len(y), dtype=float)
        bad = (pred != y)
        w[bad] = self.demote
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          random_state=self.random_state, n_jobs=-1)
        self.clf.fit(X, y, sample_weight=w)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.clf.predict(np.asarray(X, float))


# ============================================================
# 6) 协同教学 CoTeachingMLP（SOTA 基线）
# ============================================================

class CoTeachingMLP:
    """协同教学 (Co-teaching, Han et al. 2018) 的双 MLP 实现。

    每轮各自挑选"小损失"（疑似干净）样本，用对方挑选的样本更新自身。
    """

    def __init__(self, hidden=(64,), epochs=25, noise_rate=0.0, random_state=42):
        self.hidden = hidden
        self.epochs = epochs
        self.noise_rate = noise_rate
        self.random_state = random_state

    def _net(self):
        return MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=1,
                             random_state=self.random_state, warm_start=True,
                             learning_rate_init=0.01)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n = Xs.shape[0]
        classes = np.unique(y)
        keep = max(0.1, 1.0 - 2.0 * self.noise_rate)
        k = max(1, int(keep * n))
        net1, net2 = self._net(), self._net()
        net1.partial_fit(Xs, y, classes=classes)
        net2.partial_fit(Xs, y, classes=classes)
        for _ in range(self.epochs):
            p1 = net1.predict_proba(Xs)
            p1 = np.nan_to_num(p1, nan=1e-6)
            p2 = net2.predict_proba(Xs)
            p2 = np.nan_to_num(p2, nan=1e-6)
            loss1 = -np.log(p1[np.arange(n), y] + 1e-12)
            loss2 = -np.log(p2[np.arange(n), y] + 1e-12)
            idx2 = np.argsort(loss2)[:k]
            idx1 = np.argsort(loss1)[:k]
            w1 = np.zeros(n)
            w1[idx2] = 1.0
            w2 = np.zeros(n)
            w2[idx1] = 1.0
            net1.partial_fit(Xs, y, classes=classes, sample_weight=w1)
            net2.partial_fit(Xs, y, classes=classes, sample_weight=w2)
        self.net1, self.net2 = net1, net2
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, float))
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * (self.net1.predict_proba(Xs) + self.net2.predict_proba(Xs))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# 7) MentorNet（SOTA 基线）
# ============================================================

class MentorNetMLP:
    """MentorNet (Jiang et al. 2018) 的自步课程式近似。

    以"损失越低越可信"构造预定义导师权重，迭代训练学生 MLP。
    """

    def __init__(self, hidden=(100,), epochs=40, rounds=4, random_state=42):
        self.hidden = hidden
        self.epochs = epochs
        self.rounds = rounds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n = Xs.shape[0]
        student = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=self.epochs,
                                random_state=self.random_state, learning_rate_init=0.01)
        student.fit(Xs, y)
        for _ in range(self.rounds):
            p = student.predict_proba(Xs)
            p = np.nan_to_num(p, nan=1e-6)
            loss = -np.log(p[np.arange(n), y] + 1e-12)
            lo, hi = loss.min(), loss.max()
            w = 1.0 - (loss - lo) / (hi - lo + 1e-12)
            w = np.clip(w, 0.05, 1.0)
            student = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=self.epochs,
                                    random_state=self.random_state, learning_rate_init=0.01)
            student.fit(Xs, y, sample_weight=w)
        self.student = student
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, float))
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return self.student.predict_proba(Xs)

    def predict(self, X):
        return self.student.predict(X)


# ============================================================
# 8) 数据集加载
# ============================================================

def load_banknote():
    d = fetch_openml("banknote-authentication", as_frame=False, parser="auto")
    X = np.asarray(d.data, dtype=float)
    y = LabelEncoder().fit_transform(np.asarray(d.target))
    return X, y


def load_wall_robot():
    d = fetch_openml("wall-robot-navigation", version=1, as_frame=False, parser="auto")
    X = np.asarray(d.data, dtype=float)
    y = LabelEncoder().fit_transform(np.asarray(d.target))
    return X, y


def load_datasets():
    """返回主实验 4 数据集（Breast Cancer / Wine / Digits / Banknote）字典。"""
    bc = load_breast_cancer()
    wi = load_wine()
    dg = load_digits()
    bX, bY = load_banknote()
    banknote = type("D", (), {"data": bX, "target": bY})()
    return {"breast_cancer": bc, "wine": wi, "digits": dg, "banknote": banknote}


# ============================================================
# 9) 评估与工厂
# ============================================================

def evaluate(model, Xtr, ytr, Xte, yte):
    """训练并评估，返回 (Accuracy, Macro-F1, Macro-AUC)。"""
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    C = proba.shape[1]
    auc = roc_auc_score(yte, proba[:, 1]) if C == 2 else roc_auc_score(yte, proba, multi_class="ovr", average="macro")
    return acc, f1, auc


def make_factory(name, random_state=42, noise_rate=0.0):
    """按名称返回模型工厂（无参 lambda）。覆盖论文全部十类方法。"""
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
    if name == "ConfidentLearningRF":
        return lambda: ConfidentLearningRF(random_state=random_state)
    if name == "BootstrapRF":
        return lambda: BootstrapRF(random_state=random_state)
    if name == "CoTeachingMLP":
        return lambda: CoTeachingMLP(noise_rate=noise_rate, random_state=random_state)
    if name == "MentorNetMLP":
        return lambda: MentorNetMLP(random_state=random_state)
    if name == "FA-DAWRF":
        return lambda: FADAWRF(random_state=random_state)
    raise ValueError(name)
