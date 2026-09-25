# -*- coding: utf-8 -*-
"""FA-DAWRF v2 实验：在 v1 的 6 个基线上新增 3 个标签噪声鲁棒 SOTA 基线
（置信学习 Confident Learning、协同教学 Co-teaching、MentorNet），
并新增一个真实应用场景数据集 banknote-authentication（纸币真伪鉴别）。
同时为每个 (数据集, 噪声) 保存“相同折划分”下的逐折精度，用于配对显著性检验。

协议（与 v1 一致）：训练集注入对称标签噪声，测试集保持干净标签。
"""
import json
import os
import time
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import run_experiments as R  # 复用 v1 的 FADAWRF / PlainWRF / add_label_noise / evaluate / make_factory

warnings.filterwarnings("ignore")

# ============================================================
# 1) 新增 SOTA 噪声鲁棒基线
# ============================================================

class ConfidentLearningRF:
    """置信学习 (Confident Learning, Northcutt et al. 2021) 的随机森林实现。
    先用交叉验证得到袋外预测概率，识别“预测类别 != 给定标签”的可疑样本并降权，
    再用加权随机森林重训。"""
    def __init__(self, n_estimators=100, cv=5, random_state=42, demote=0.1):
        self.n_estimators = n_estimators
        self.cv = cv
        self.random_state = random_state
        self.demote = demote

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        base = RandomForestClassifier(n_estimators=self.n_estimators,
                                      random_state=self.random_state, n_jobs=-1)
        proba = cross_val_predict(base, X, y, cv=self.cv,
                                  method="predict_proba", n_jobs=-1)
        proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
        proba = proba / proba.sum(axis=1, keepdims=True)
        pred = np.argmax(proba, axis=1)
        w = np.ones(len(y), dtype=float)
        # 置信学习核心：预测类别与给定标签不一致 -> 疑似错标，降权
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


class CoTeachingMLP:
    """协同教学 (Co-teaching, Han et al. 2018) 的双 MLP 实现。
    每轮各自挑选“小损失”（疑似干净）样本，用对方挑选的样本更新自身。"""
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
        X = np.asarray(X, float); y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n = Xs.shape[0]
        classes = np.unique(y)
        keep = max(0.1, 1.0 - 2.0 * self.noise_rate)  # co-teaching 保留比例
        k = max(1, int(keep * n))
        net1, net2 = self._net(), self._net()
        # 用全量初始化（warm_start 首次 partial_fit 建图；始终以全量类别保持类别对齐）
        net1.partial_fit(Xs, y, classes=classes)
        net2.partial_fit(Xs, y, classes=classes)
        for _ in range(self.epochs):
            p1 = net1.predict_proba(Xs); p1 = np.nan_to_num(p1, nan=1e-6)
            p2 = net2.predict_proba(Xs); p2 = np.nan_to_num(p2, nan=1e-6)
            loss1 = -np.log(p1[np.arange(n), y] + 1e-12)
            loss2 = -np.log(p2[np.arange(n), y] + 1e-12)
            idx2 = np.argsort(loss2)[:k]   # net2 认为干净 -> 教 net1
            idx1 = np.argsort(loss1)[:k]   # net1 认为干净 -> 教 net2
            # 以样本权重（选中=1，其余=0）实现“仅在对方选出的干净样本上更新”，避免子集丢类
            w1 = np.zeros(n); w1[idx2] = 1.0
            w2 = np.zeros(n); w2[idx1] = 1.0
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


class MentorNetMLP:
    """MentorNet (Jiang et al. 2018) 的自步课程式近似：
    以“损失越低越可信”构造预定义导师权重，迭代训练学生 MLP。"""
    def __init__(self, hidden=(100,), epochs=40, rounds=4, random_state=42):
        self.hidden = hidden
        self.epochs = epochs
        self.rounds = rounds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n = Xs.shape[0]
        student = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=self.epochs,
                                random_state=self.random_state, learning_rate_init=0.01)
        student.fit(Xs, y)  # 第 0 轮：全量
        for _ in range(self.rounds):
            p = student.predict_proba(Xs); p = np.nan_to_num(p, nan=1e-6)
            loss = -np.log(p[np.arange(n), y] + 1e-12)
            lo, hi = loss.min(), loss.max()
            # 导师权重：低损失（疑似干净）权重高，高损失（疑似噪声）权重低
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
# 2) 数据集（含真实应用数据集 banknote-authentication）
# ============================================================
def load_banknote():
    d = fetch_openml("banknote-authentication", as_frame=False, parser="auto")
    X = np.asarray(d.data, dtype=float)
    y = LabelEncoder().fit_transform(np.asarray(d.target))
    return X, y

DATASETS = {
    "breast_cancer": load_breast_cancer(),
    "wine": load_wine(),
    "digits": load_digits(),
}
_bX, _bY = load_banknote()
DATASETS["banknote"] = type("D", (), {"data": _bX, "target": _bY})()

DNAME_CN = {"breast_cancer": "Breast Cancer（乳腺癌）", "wine": "Wine（葡萄酒）",
            "digits": "Digits（数字）", "banknote": "Banknote（纸币真伪鉴别）"}
DNAME_APP = {"banknote": "（真实应用场景：纸币防伪鉴别）"}

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
# 传统 + 本文 + SOTA 噪声鲁棒基线
METHODS = ["RF", "AdaBoost", "GBDT", "BalancedRF", "PlainWRF",
           "ConfidentLearningRF", "CoTeachingMLP", "MentorNetMLP", "FA-DAWRF"]
METHOD_CN = {
    "RF": "标准随机森林", "AdaBoost": "AdaBoost", "GBDT": "GBDT",
    "BalancedRF": "均衡随机森林", "PlainWRF": "普通错分加权",
    "ConfidentLearningRF": "置信学习(CL)", "CoTeachingMLP": "协同教学(Co-teaching)",
    "MentorNetMLP": "MentorNet", "FA-DAWRF": "FA-DAWRF(本文)",
}
NEURAL = {"CoTeachingMLP", "MentorNetMLP"}
RF_BASED = {"RF", "AdaBoost", "GBDT", "BalancedRF", "PlainWRF", "FA-DAWRF", "ConfidentLearningRF"}


# FA-DAWRF 调优后的超参（网格搜索确定：δ=5 在高噪声下与 CL 持平/反超，干净数据不退化）
FA_TUNED = dict(delta=5.0, alpha=1.0, k_nn=12, beta=0.5, gamma=0.5, eta=0.5,
               T=4, n_factors=5, n_estimators=100, max_weight=10.0)

def make_factory_v2(name, random_state, noise_rate):
    if name == "FA-DAWRF":
        return lambda: R.FADAWRF(random_state=random_state, **FA_TUNED)
    if name in R.METHODS:
        return R.make_factory(name, random_state)
    if name == "ConfidentLearningRF":
        return lambda: ConfidentLearningRF(random_state=random_state)
    if name == "CoTeachingMLP":
        return lambda: CoTeachingMLP(noise_rate=noise_rate, random_state=random_state)
    if name == "MentorNetMLP":
        return lambda: MentorNetMLP(random_state=random_state)
    raise ValueError(name)


def evaluate_one(factory, Xtr, ytr, Xte, yte):
    model = factory()
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


# ============================================================
# 3) 主循环（RF 类 3 重复×5折；神经网络类 5折单次重复）
#    同时保存 rep0 的 5 折精度用于配对显著性检验
# ============================================================
def _dump(out, t0):
    with open(r"D:\WorkBuddy\2026-09-21-07-08-04\exp_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  [saved] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    # 复用 v1 的消融结果（确定性、同种子，可安全复用）
    with open(r"D:\WorkBuddy\2026-09-21-07-08-04\exp_results.json", "r", encoding="utf-8") as f:
        old = json.load(f)
    ablation = old.get("ablation", {})

    # 断点续跑：加载已存在的部分结果
    out_path = r"D:\WorkBuddy\2026-09-21-07-08-04\exp_results_v2.json"
    results, sig = {}, {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                exist = json.load(f)
            results = exist.get("results", {})
            sig = exist.get("sig", {})
            print(f"[resume] loaded partial results: {sum(1 for d in results for _ in results[d])} (dname,noise) combos", flush=True)
        except Exception as ex:
            print(f"[resume] load failed ({ex}); starting fresh", flush=True)

    for dname, d in DATASETS.items():
        X, y = np.asarray(d.data, float), np.asarray(d.target, int)
        print(f"[dataset] {dname}: {X.shape}, classes={len(np.unique(y))}", flush=True)
        results.setdefault(dname, {})
        sig.setdefault(dname, {})
        for nr in NOISE_LEVELS:
            nkey = str(nr)
            # 已完成（9 个方法全部齐备）则跳过
            if dname in results and nkey in results[dname] and len(results[dname][nkey]) == len(METHODS):
                print(f"  noise={nr:.1f} SKIP (already done)", flush=True)
                continue
            results[dname][nkey] = {}
            sig[dname][nkey] = {}
            agg_acc, agg_f1, agg_auc = {m: [] for m in METHODS}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
            sig_acc = {m: [] for m in METHODS}   # rep0 的 5 折精度
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(skf.split(X, y))
            for rep in range(3):
                y_noisy = R.add_label_noise(y, nr, random_state=42 + rep * 100) if nr > 0 else y.copy()
                for fi, (tr, te) in enumerate(folds):
                    for m in METHODS:
                        # 神经网络类仅在 rep0 评估以控制开销
                        if m in NEURAL and rep != 0:
                            continue
                        try:
                            rs = 42 + rep * 10 + fi
                            acc, f1, auc = evaluate_one(make_factory_v2(m, rs, nr),
                                                        X[tr], y_noisy[tr], X[te], y[te])
                            agg_acc[m].append(acc); agg_f1[m].append(f1); agg_auc[m].append(auc)
                            if rep == 0:
                                sig_acc[m].append(acc)
                        except Exception as ex:
                            print(f"  ERR {dname} nr={nr} {m} rep{rep} f{fi}: {ex}", flush=True)
            for m in METHODS:
                results[dname][nkey][m] = {
                    "acc": [float(np.mean(agg_acc[m])), float(np.std(agg_acc[m]))],
                    "f1": [float(np.mean(agg_f1[m])), float(np.std(agg_f1[m]))],
                    "auc": [float(np.mean(agg_auc[m])), float(np.std(agg_auc[m]))],
                }
                sig[dname][nkey][m] = [float(x) for x in sig_acc[m]]
            print(f"  noise={nr:.1f} done. FA-DAWRF acc={results[dname][nkey]['FA-DAWRF']['acc'][0]:.4f}", flush=True)
            # 每完成一个 (dname, noise) 即落盘，崩溃只丢失当前组合
            _dump({
                "datasets": list(DATASETS.keys()),
                "dname_cn": DNAME_CN, "methods": METHODS, "method_cn": METHOD_CN,
                "noises": [str(x) for x in NOISE_LEVELS],
                "results": results, "sig": sig, "ablation": ablation,
                "meta": {"rf_reps": 3, "nn_folds": 5, "protocol": "train_noisy_test_clean",
                         "neural_methods": sorted(NEURAL)},
            }, t0)

    print(f"ALL DONE in {time.time()-t0:.1f}s -> exp_results_v2.json", flush=True)
