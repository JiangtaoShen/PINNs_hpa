#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kde_test.py
Load KDE dictionary 'kde_models_dict_ab_all.pkl' and evaluate 10 parameter sets.
Saves results to 'kde_test_output.csv' and prints to stdout.
"""

import os
import pickle
import numpy as np
import pandas as pd

KDE_PATH = "kde_models_dict_ab_all.pkl"
OUT_CSV = "kde_test_output.csv"

def _clip_and_cast(x):
    x = np.array(x, dtype=float).flatten()
    x[0] = int(np.clip(np.round(x[0]), 2, 10))
    x[1] = int(np.clip(np.round(x[1]), 5, 100))
    x[2] = int(np.clip(np.round(x[2]), 0, 4))
    x[3] = int(np.clip(np.round(x[3]), 5000, 100000))
    x[4] = int(np.clip(np.round(x[4]), 10, 200))
    x[5] = float(np.clip(x[5], 1e-5, 0.1))
    return x

def load_kdes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"KDE file not found: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    # support dict with 'all pdes' or direct dict of 6 KDEs
    if isinstance(d, dict) and "all pdes" in d:
        return d["all pdes"]
    return d

def eval_kde_for_params(kde_models, params):
    """
    kde_models: list-like of 6 KDE objects (each supports score_samples)
    params: raw parameter list/array [n_layers, n_nodes, activation, epochs, grid_size, lr]
    Returns dict with per-dim log densities, sum_log, and prior (exp(sum_log), may underflow to 0).
    """
    p = _clip_and_cast(params)
    per_log = []
    sum_log = 0.0
    for i in range(6):
        try:
            if kde_models is None:
                raise RuntimeError("kde_models is None")
            if i == 5:
                val = np.log10(p[5])
                logd = float(kde_models[i].score_samples([[val]])[0])
            else:
                logd = float(kde_models[i].score_samples([[p[i]]])[0])
        except Exception as e:
            # If any KDE missing or fail, set logd = nan
            logd = float("nan")
        per_log.append(logd)
        if not np.isnan(logd):
            sum_log += logd
        else:
            sum_log = float("nan")
            break

    # try to get prior = exp(sum_log) (may underflow to 0.0)
    prior = None
    if not np.isnan(sum_log):
        try:
            prior = float(np.exp(sum_log))
        except OverflowError:
            prior = float("inf")
        except Exception:
            prior = 0.0
    else:
        prior = float("nan")

    return {
        "params_clipped": p,
        "logd_0": per_log[0], "logd_1": per_log[1], "logd_2": per_log[2],
        "logd_3": per_log[3], "logd_4": per_log[4], "logd_5": per_log[5],
        "sum_log": sum_log, "prior": prior
    }

def main():
    # 10 example parameter sets (you can replace these with your own)
    samples = [
        [2, 5, 0, 5000, 10, 1e-5],
        [10, 100, 4, 100000, 200, 1e-1],
        [6, 50, 2, 50000, 100, 1e-3],
        [3, 20, 1, 10000, 20, 5e-5],
        [8, 80, 3, 80000, 150, 2e-4],
        [4, 10, 0, 7000, 30, 1e-4],
        [5, 60, 2, 20000, 120, 1e-2],
        [7, 30, 4, 90000, 60, 3e-5],
        [9, 40, 1, 30000, 80, 5e-3],
        [6, 70, 3, 60000, 180, 8e-4],
    ]

    print("Loading KDE models from:", KDE_PATH)
    try:
        kde_models = load_kdes(KDE_PATH)
    except Exception as e:
        print("Failed to load KDE models:", e)
        kde_models = None

    rows = []
    for idx, s in enumerate(samples, 1):
        out = eval_kde_for_params(kde_models, s)
        p = out["params_clipped"]
        row = {
            "sample_id": idx,
            "n_layers": int(p[0]), "n_nodes": int(p[1]), "activation": int(p[2]),
            "epochs": int(p[3]), "grid_size": int(p[4]), "learning_rate": float(p[5]),
            "logd_0": out["logd_0"], "logd_1": out["logd_1"], "logd_2": out["logd_2"],
            "logd_3": out["logd_3"], "logd_4": out["logd_4"], "logd_5": out["logd_5"],
            "sum_log": out["sum_log"], "prior": out["prior"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # pretty print
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print("\nKDE test results (per-sample):\n")
    print(df.to_string(index=False))
    # save csv
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")

if __name__ == "__main__":
    main()