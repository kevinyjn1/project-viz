"""Precompute every number the Checkpoint Champion walkthrough shows.

Usage (from anywhere):
    python3 precompute.py path/to/League_of_Legends_Analysis/data/lol_2024.csv

It re-runs the final pipeline of code/data_analysis_and _processing.ipynb exactly
(same cleaning, features, ColumnTransformer, LogisticRegressionCV(cv=5, penalty='l1',
solver='liblinear', random_state=398), train_test_split(test_size=0.3, random_state=103)),
checks the accuracy against the notebook's saved output, and writes small JSON files to
../../lol-predictor/data/ (relative to this script).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "lol-predictor", "data"))
CSV = sys.argv[1] if len(sys.argv) > 1 else "lol_2024.csv"
GAME = "LOLTMNT05_90307"  # Worlds 2024 final, game 5: Bilibili Gaming (Blue) vs T1 (Red)

# Numbers saved in the notebook's last output table (cell 44) and README.
NOTEBOOK_TABLE = [
    {"name": "Baseline logistic regression", "nonzero": 5, "train": 0.809433, "test": 0.812473, "auc": 0.897624},
    {"name": "LASSO (L1) LogisticRegressionCV", "nonzero": 9, "train": 0.835888, "test": 0.8361, "auc": 0.921376, "final": True},
    {"name": "Ridge (L2) LogisticRegressionCV", "nonzero": 57, "train": 0.835979, "test": 0.836526, "auc": 0.921788},
    {"name": "RFECV backward elimination", "nonzero": 5, "train": 0.835523, "test": 0.833972, "auc": 0.920703},
]
NOTEBOOK_CM = [[1939, 398], [372, 1989]]  # labels [1, 0]; README


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV, low_memory=False)
    counts = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "games": int(df["gameid"].nunique())}

    # ---- notebook cells 12-19: keep team rows, drop rows missing the minute-25 snapshot
    data = df[df["position"] == "team"]
    counts["team_rows"] = int(len(data))
    cc = ['goldat25', 'xpat25', 'csat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25',
          'golddiffat25', 'xpdiffat25', 'csdiffat25',
          'killsat25', 'assistsat25', 'deathsat25',
          'opp_killsat25', 'opp_assistsat25', 'opp_deathsat25']
    missing = data[data[cc].isna().any(axis=1)]
    counts["missing_frac"] = r(data[cc].isna().mean().iloc[0], 5)
    counts["missing_rows"] = int(len(missing))
    counts["missing_by_league"] = {k: int(v) for k, v in missing["league"].value_counts().head(4).items()}
    data = data.dropna(subset=cc)
    counts["clean_rows"] = int(len(data))

    # histogram of gold difference at 25 by result (EDA, all clean rows)
    edges = np.arange(-20000, 20001, 1000)
    hist = {}
    for res in (0, 1):
        v = data.loc[data["result"] == res, "golddiffat25"].clip(-19999, 19999)
        hist[str(res)] = np.histogram(v, bins=edges)[0].tolist()
    ahead = data[data["golddiffat25"] > 0]
    counts["ahead_rows"] = int(len(ahead))
    counts["ahead_win_rate"] = r(ahead["result"].mean(), 4)
    means = data.groupby("result")[["goldat25", "xpat25", "csat25", "killsat25", "deathsat25"]].mean().round(2)

    # ---- baseline (cell 34-36), to check the reproduction
    base_cols = ['goldat25', 'xpat25', 'csat25', 'killsat25', 'deathsat25']
    Xtr, Xte, ytr, yte = train_test_split(data[base_cols], data["result"], test_size=0.3, random_state=103)
    bp = make_pipeline(StandardScaler(), LogisticRegression(random_state=398, solver='liblinear')).fit(Xtr, ytr)
    base_test = accuracy_score(yte, bp.predict(Xte))

    # ---- final model (cells 38-40)
    data = data.copy()
    data['side_binary'] = data['side'].map({'Blue': 0, 'Red': 1})
    feats = ['goldat25', 'xpat25', 'csat25', 'killsat25', 'deathsat25', 'opp_goldat25', 'opp_xpat25',
             'opp_csat25', 'side_binary', 'gamelength', 'league']
    X, y = data[feats], data['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=103)
    numeric = feats[:-1]
    pre = ColumnTransformer([('num', StandardScaler(), numeric),
                             ('cat', OneHotEncoder(handle_unknown='ignore'), ['league'])], remainder='drop')
    pipe = make_pipeline(pre, LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', random_state=398))
    pipe.fit(X_train, y_train)
    ct = pipe.named_steps['columntransformer']
    m = pipe.named_steps['logisticregressioncv']
    names = list(ct.get_feature_names_out())
    sc = ct.named_transformers_['num']
    leagues = list(ct.named_transformers_['cat'].categories_[0])

    ytr_pred = pipe.predict(X_train)
    yte_pred = pipe.predict(X_test)
    yte_score = pipe.decision_function(X_test)
    metrics = {
        "train_acc": r(accuracy_score(y_train, ytr_pred), 6),
        "test_acc": r(accuracy_score(y_test, yte_pred), 6),
        "auc": r(roc_auc_score(y_test, yte_score), 6),
        "cm": confusion_matrix(y_test, yte_pred, labels=[1, 0]).tolist(),
        "baseline_test_acc": r(base_test, 6),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    ok = (abs(metrics["test_acc"] - 0.8361) < 5e-5 and abs(metrics["train_acc"] - 0.835888) < 5e-6
          and metrics["cm"] == NOTEBOOK_CM and abs(metrics["baseline_test_acc"] - 0.812473) < 5e-6)
    metrics["matches_notebook"] = bool(ok)
    print("reproduced:", metrics, "matches notebook:", ok)

    coef = m.coef_[0]
    intercept = float(m.intercept_[0])
    C = float(m.C_[0])
    cv_acc = m.scores_[1].mean(axis=0)  # mean 5-fold accuracy for each C

    # ---- regularisation path: refit the same L1 model at many C values (real fits)
    Ztr = ct.transform(X_train)
    Ztr = Ztr.toarray() if hasattr(Ztr, "toarray") else np.asarray(Ztr)
    path_Cs = sorted(set(np.round(np.logspace(-4, 4, 41), 7).tolist() + [C]))
    path = []
    for c in path_Cs:
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, random_state=398).fit(Ztr, y_train)
        w = lr.coef_[0]
        path.append({"C": r(c, 7), "w": [r(v, 4) for v in w[:len(numeric)]],
                     "league_nonzero": int(np.count_nonzero(w[len(numeric):])),
                     "acc": r(lr.score(Ztr, y_train), 4)})

    # ---- "watch it learn": proximal gradient descent (ISTA) on liblinear's L1 objective
    # liblinear minimises ||w||_1 + C * sum log(1+exp(-y (w.x + b*1))) with the bias as a
    # penalised feature. Divided by C*n: lambda*||w||_1 + mean log-loss, lambda = 1/(C n).
    n = Ztr.shape[0]
    A = np.hstack([Ztr, np.ones((n, 1))])
    ys = np.where(y_train.values == 1, 1.0, -1.0)
    lam = 1.0 / (C * n)
    L = np.linalg.eigvalsh(A.T @ A / n).max() / 4.0
    step = 1.0 / L
    w = np.zeros(A.shape[1])
    keep = sorted(set(np.unique(np.round(np.logspace(0, np.log10(2000), 34)).astype(int)).tolist()))
    n_iter = 60000  # run long to check convergence; frames are only kept up to 2000
    frames = [{"it": 0, "w": [0.0] * len(numeric), "b": 0.0,
               "loss": r(np.log(2), 5), "acc": r(np.mean(y_train.values == 1), 4)}]
    def logloss(w):
        z = A @ w
        return np.mean(np.logaddexp(0, -ys * z))
    for it in range(1, n_iter + 1):
        z = A @ w
        g = A.T @ (-ys / (1 + np.exp(ys * z))) / n
        v = w - step * g
        w = np.sign(v) * np.maximum(np.abs(v) - step * lam, 0)
        if it in keep:
            acc = np.mean(((A @ w) > 0).astype(int) == y_train.values)
            frames.append({"it": it, "w": [r(x, 4) for x in w[:len(numeric)]], "b": r(w[-1], 4),
                           "loss": r(logloss(w), 5), "acc": r(acc, 4)})
    full = np.append(coef, intercept)
    gap = float(np.abs(w - full).max())
    print("ISTA endpoint vs sklearn coef, max abs diff:", gap)
    # final frame = the fitted model itself (the re-enactment ends exactly there)
    frames.append({"it": "fitted", "w": [r(x, 4) for x in coef[:len(numeric)]], "b": r(intercept, 4),
                   "loss": r(logloss(full), 5), "acc": r(metrics["train_acc"], 4)})

    # ---- how the row-level split treats the two rows of each game
    side_of = pd.Series(np.where(data.index.isin(X_test.index), 1, 0), index=data.index)
    per_game = side_of.groupby(data["gameid"]).sum()
    counts["games_clean"] = int(len(per_game))
    counts["games_both_train"] = int((per_game == 0).sum())
    counts["games_both_test"] = int((per_game == 2).sum())
    counts["games_split"] = int((per_game == 1).sum())

    # ---- test rows: logits and labels for the dot plot
    test_logit = [r(v, 3) for v in yte_score]

    # ---- the game we follow
    raw = df[df["gameid"] == GAME]
    raw_cols = ['position', 'side', 'teamname', 'playername', 'champion', 'result',
                'goldat25', 'xpat25', 'csat25', 'killsat25', 'deathsat25', 'assistsat25']
    raw12 = [[(None if pd.isna(v) else (v if isinstance(v, str) else r(v, 0))) for v in row]
             for row in raw[raw_cols].values.tolist()]
    teams = data[data["gameid"] == GAME]
    test_idx = set(X_test.index)
    rows = []
    for idx, t in teams.iterrows():
        z = ct.transform(t[feats].to_frame().T)
        z = (z.toarray() if hasattr(z, "toarray") else np.asarray(z))[0]
        prod = z * coef
        logit = float(prod.sum() + intercept)
        p_pipe = float(pipe.predict_proba(t[feats].to_frame().T)[0, 1])
        assert abs(1 / (1 + np.exp(-logit)) - p_pipe) < 1e-9
        rows.append({
            "team": t["teamname"], "side": t["side"], "result": int(t["result"]),
            "in_test": idx in test_idx,
            "at25": {c: r(t[c], 0) for c in cc},
            "gamelength": int(t["gamelength"]), "league": t["league"],
            "x": [r(t[c], 0) for c in numeric],
            "z": [r(v, 4) for v in z[:len(numeric)]],
            "league_index": leagues.index(t["league"]),
            "prod": [r(v, 4) for v in prod[:len(numeric)]],
            "logit": r(logit, 4), "p": r(p_pipe, 4),
        })
    meta = raw.iloc[0]
    dropped = missing.iloc[0]
    game = {"gameid": GAME, "league": meta["league"], "date": str(meta["date"]), "game": int(meta["game"]),
            "patch": str(meta["patch"]), "gamelength": int(meta["gamelength"]),
            "raw_cols": raw_cols, "raw": raw12, "teams": rows,
            "dropped_example": {"gameid": dropped["gameid"], "league": dropped["league"],
                                "team": dropped["teamname"], "side": dropped["side"],
                                "result": int(dropped["result"]), "completeness": dropped["datacompleteness"]}}

    model = {
        "numeric": numeric, "names": names, "leagues": leagues,
        "mean": [r(v, 3) for v in sc.mean_], "scale": [r(v, 3) for v in sc.scale_],
        "coef": [r(v, 6) for v in coef], "intercept": r(intercept, 6),
        "C": r(C, 6), "Cs": [r(v, 6) for v in m.Cs_], "cv_acc": [r(v, 5) for v in cv_acc],
        "lambda_readme": r(1 / C, 2),
        "path": path, "frames": frames, "ista_gap": r(gap, 5),
        "metrics": metrics, "notebook_table": NOTEBOOK_TABLE, "counts": counts,
        "hist": {"edges": edges.tolist(), "loss": hist["0"], "win": hist["1"]},
        "means_by_result": {str(k): {c: float(v) for c, v in row.items()} for k, row in means.iterrows()},
    }
    test = {"logit": test_logit, "y": [int(v) for v in y_test.values]}
    for fn, obj in (("model.json", model), ("game.json", game), ("test.json", test)):
        with open(os.path.join(OUT, fn), "w") as f:
            json.dump(obj, f, separators=(",", ":"), allow_nan=False)
        print("wrote", fn, os.path.getsize(os.path.join(OUT, fn)), "bytes")


if __name__ == "__main__":
    main()
