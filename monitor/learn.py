"""Walk-forward learner.

Every cycle it replays ~5 years of daily bars for the watchlist plus the latest scan's top names,
labels every day with its triple-barrier outcome (did the stock rise by its volatility-sized target
before it fell to its stop, within 40 sessions?)
and then:

1. Readiness model: refits the logistic regression on all labelled history. How far the
   coefficients may move away from the expert prior (lambda) is picked on a held-out final year
   that the fit never saw, and it only changes when the new choice beats the deployed one there.
2. Trigger rule: replays every variant in TRIGGER_GRID, picks the best on the training years, and
   swaps it in only if it also beats the deployed rule on the held-out year.

Structural changes need out-of-sample evidence; coefficients simply keep absorbing new data.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from .model import auc, fit_intercept, fit_logit, predict, readiness, sigmoid
from .signals import features, outcomes, trigger_events

TEST_SESSIONS = 250  # held-out final year
STEP = 3             # sample every 3rd session for the model (labels overlap heavily day to day)
READY_STEPS = [0, 50, 60, 70, 80]


def build_frames(bars: dict[str, pd.DataFrame], spy_close: pd.Series, label: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    for t, b in bars.items():
        if len(b) < 260:
            continue
        f = features(b, spy_close)
        frames[t] = f.join(outcomes(f["close"], label))
    return frames


def windows(sessions: pd.DatetimeIndex, horizon: int):
    """(train_end, test_start, label_end): an embargo of `horizon` sessions separates train and test."""
    label_end = sessions[-(horizon + 1)]
    test_start = sessions[-(horizon + 1) - TEST_SESSIONS]
    train_end = sessions[-(horizon + 1) - TEST_SESSIONS - horizon]
    return train_end, test_start, label_end


def model_rows(frames, sessions, label_end) -> pd.DataFrame:
    keep = sessions[::STEP]
    parts = [fr.loc[fr.index.isin(keep) & (fr.index <= label_end), C.FEATURES + ["y", "ret"]].assign(ticker=t)
             for t, fr in frames.items()]
    return pd.concat(parts).dropna(subset=C.FEATURES + ["y"])


def _std(train):
    mean = train[C.FEATURES].mean()
    std = train[C.FEATURES].std().replace(0, 1).fillna(1)
    return mean, std


def learn_model(rows, train_end, test_start, current_lam):
    train, test = rows[rows.index <= train_end], rows[rows.index >= test_start]
    prior = np.array([C.PRIOR[k] for k in C.FEATURES])
    mean, std = _std(train)
    Xtr = ((train[C.FEATURES] - mean) / std).to_numpy()
    Xte = ((test[C.FEATURES] - mean) / std).to_numpy()
    ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
    by_lam = {}
    for lam in C.LAMBDAS:
        w, b = fit_logit(Xtr, ytr, prior, lam)
        by_lam[lam] = auc(yte, sigmoid(Xte @ w + b))
    prior_auc = auc(yte, Xte @ prior + fit_intercept(Xtr, ytr, prior))
    best = max(by_lam, key=lambda k: by_lam[k])
    cur = current_lam if current_lam in by_lam else None
    adopt = cur is None or by_lam[best] >= by_lam[cur] + C.ADOPT_AUC_MARGIN
    lam = best if adopt else cur
    # the same spec fitted on the training years only: scores the held-out year without peeking
    w, b = fit_logit(Xtr, ytr, prior, lam)
    holdout = {"features": C.FEATURES, "coef": w.tolist(), "intercept": b, "mean": mean.tolist(),
               "std": std.tolist(), "quantiles": np.quantile(sigmoid(Xtr @ w + b), np.linspace(0, 1, 101)).tolist()}
    # production fit: every labelled row, with the chosen prior strength
    mean, std = _std(rows)
    X = ((rows[C.FEATURES] - mean) / std).to_numpy()
    w, b = fit_logit(X, rows["y"].to_numpy(), prior, lam)
    probs = sigmoid(X @ w + b)
    model = {"features": C.FEATURES, "coef": w.round(5).tolist(), "intercept": round(float(b), 5),
             "mean": mean.round(6).tolist(), "std": std.round(6).tolist(), "lam": lam,
             "quantiles": np.quantile(probs, np.linspace(0, 1, 101)).round(5).tolist()}
    report = {"lam": lam, "changed": lam != current_lam, "test_auc": round(by_lam[lam], 4),
              "test_auc_by_lam": {str(k): round(v, 4) for k, v in by_lam.items()},
              "prior_only_auc": round(prior_auc, 4), "n_train": len(train), "n_test": len(test),
              "base_rate": round(float(rows["y"].mean()), 4)}
    return model, report, holdout


def _event_stats(ev: pd.DataFrame, base: pd.DataFrame) -> dict:
    n = len(ev)
    exp = float(ev["ret"].mean()) if n else float("nan")
    base_exp = float(base["ret"].mean())
    return {"n": n, "hit_rate": round(float(ev["y"].mean()), 4) if n else None,
            "expectancy": round(exp, 5) if n else None,
            "base_hit_rate": round(float(base["y"].mean()), 4), "base_expectancy": round(base_exp, 5),
            # edge over buying a random day, shrunk toward zero when there are few events
            "score": round((exp - base_exp) * n / (n + 30), 5) if n else -1.0}


def learn_trigger(frames, train_end, test_start, label_end, current: dict):
    fixed = {k: current[k] for k in current if k not in C.TRIGGER_GRID}
    labelled = pd.concat([fr.loc[fr.index <= label_end, ["y", "ret"]] for fr in frames.values()]).dropna()
    base_tr, base_te = labelled[labelled.index <= train_end], labelled[labelled.index >= test_start]
    combos = list(itertools.product(*C.TRIGGER_GRID.values()))
    cur_combo = tuple(current[k] for k in C.TRIGGER_GRID)
    if cur_combo not in combos:  # a hand-edited rule outside the grid still gets scored
        combos.append(cur_combo)
    results = []
    for combo in combos:
        p = {**fixed, **dict(zip(C.TRIGGER_GRID, combo))}
        ev = pd.concat([fr.loc[trigger_events(fr, p) & (fr.index <= label_end), ["y", "ret"]]
                        for fr in frames.values()]).dropna()
        results.append((p, _event_stats(ev[ev.index <= train_end], base_tr),
                        _event_stats(ev[ev.index >= test_start], base_te), _event_stats(ev, labelled)))
    best = max(results, key=lambda r: r[1]["score"])
    cur = next(r for r in results if all(r[0][k] == current[k] for k in C.TRIGGER_GRID))
    adopt = (best[0] != cur[0] and best[2]["n"] >= 20
             and best[2]["score"] >= cur[2]["score"] + C.ADOPT_EXP_MARGIN)
    chosen = best if adopt else cur
    report = {"changed": bool(adopt), "rule": {k: chosen[0][k] for k in C.TRIGGER_GRID},
              "all": chosen[3], "test": chosen[2],
              "challenger": {"rule": {k: best[0][k] for k in C.TRIGGER_GRID}, "train": best[1], "test": best[2]},
              "champion_test": cur[2]}
    return chosen[0], report


def learn_min_ready(frames, rule: dict, holdout: dict, test_start, label_end, current: float) -> tuple[int, dict]:
    """Readiness a trigger needs before it counts as a strong signal, judged on the held-out year."""
    ev = []
    for fr in frames.values():
        hit = trigger_events(fr, rule) & (fr.index >= test_start) & (fr.index <= label_end)
        if hit.any():
            e = fr.loc[hit, C.FEATURES + ["y", "ret"]].dropna()
            ev.append(e.assign(ready=readiness(holdout, predict(holdout, e[C.FEATURES].to_numpy()))))
    ev = pd.concat(ev) if ev else pd.DataFrame(columns=["y", "ret", "ready"])
    base = pd.concat([fr.loc[(fr.index >= test_start) & (fr.index <= label_end), ["y", "ret"]]
                      for fr in frames.values()]).dropna()
    by = {t: _event_stats(ev[ev["ready"] >= t], base) for t in READY_STEPS}
    best = max((t for t in by if t == 0 or by[t]["n"] >= 30), key=lambda t: by[t]["score"])
    cur = current if current in by else 0
    chosen = best if by[best]["score"] >= by[cur]["score"] + C.ADOPT_EXP_MARGIN else cur
    return chosen, {"min_ready": chosen, "changed": chosen != current,
                    "by_threshold": {str(t): {k: by[t][k] for k in ("n", "hit_rate", "expectancy")} for t in by}}


def trigger_timing(frames, rule: dict, label_end) -> dict:
    """How long past triggers took to resolve: the basis for each open trade's target window."""
    ev = pd.concat([fr.loc[trigger_events(fr, rule) & (fr.index <= label_end), ["y", "ret", "bars", "stop"]]
                    for fr in frames.values()]).dropna()
    win = ev[ev["y"] == 1]
    stop = ev[(ev["y"] == 0) & (ev["ret"] <= -ev["stop"] + 1e-9)]
    if len(win) < 20:
        return {"n": len(ev), "wins": len(win)}
    q = win["bars"].quantile([0.25, 0.5, 0.75])
    share = lambda x: round(len(x) / len(ev), 3)
    return {"n": len(ev), "wins": len(win), "hit": share(win), "stopped": share(stop),
            "timed_out": round(1 - share(win) - share(stop), 3),
            "p25": int(q[0.25]), "median": int(q[0.5]), "p75": int(q[0.75]),
            "within10": round(float((win["bars"] <= 10).mean()), 3),
            "within20": round(float((win["bars"] <= 20).mean()), 3),
            "stop_median": int(stop["bars"].median()) if len(stop) else None,
            "avg_gain": round(float(win["ret"].mean()), 4)}


def learn(bars, spy_close, engine: dict | None, today: dt.date) -> dict:
    engine = engine or {}
    label = engine.get("label", C.LABEL)
    frames = build_frames(bars, spy_close, label)
    sessions = spy_close.index
    train_end, test_start, label_end = windows(sessions, label["horizon"])
    rows = model_rows(frames, sessions, label_end)
    cur_model = engine.get("model", {})
    lam = cur_model.get("lam") if cur_model.get("features") == C.FEATURES else None
    model, m_rep, holdout = learn_model(rows, train_end, test_start, lam)
    cur_trigger = engine.get("trigger", C.TRIGGER)
    trigger, t_rep = learn_trigger(frames, train_end, test_start, label_end, cur_trigger)
    trigger["min_ready"], r_rep = learn_min_ready(frames, trigger, holdout, test_start, label_end,
                                                  cur_trigger.get("min_ready", 0))
    t_rep["ready_filter"] = r_rep
    t_rep["timing"] = trigger_timing(frames, trigger, label_end)
    changed = m_rep["changed"] or t_rep["changed"] or r_rep["changed"]
    version = engine.get("version", 0) + (1 if changed or not engine else 0)
    entry = {"date": today.isoformat(), "version": version, "tickers": len(frames),
             "window": {"from": str(rows.index.min().date()), "train_end": str(train_end.date()),
                        "test_start": str(test_start.date()), "label_end": str(label_end.date())},
             "model": m_rep, "trigger": t_rep}
    C.ENGINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with C.ENGINE_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return {"version": version, "updated": today.isoformat(), "label": label, "trigger": trigger,
            "model": model, "metrics": {"model": m_rep, "trigger": t_rep, "tickers": len(frames)}}
