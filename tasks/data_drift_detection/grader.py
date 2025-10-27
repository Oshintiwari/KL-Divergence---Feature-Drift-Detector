# tasks/data_drift_detection/grader.py
import numpy as np
import pandas as pd
import io, contextlib

def _kl(p, q, eps=1e-10):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def grade(seed: int = 42):
    """
    Returns: {"passed": bool, "reasons": [str]}
    """
    np.random.seed(seed)

    n = 2000
    old = pd.DataFrame({
        "age":    np.random.normal(40, 10, n),
        "income": np.random.normal(60000, 8000, n),
        "height": np.random.normal(170, 7, n),
    })
    new = old.copy()
    new["age"]    = new["age"] + np.random.normal(5, 2, n)
    new["income"] = new["income"] * 1.1

    namespace = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(open("tasks/data_drift_detection/detect_drift.py", "r", encoding="utf-8").read(), namespace)
    except Exception as e:
        return {"passed": False, "reasons": [f"import/exec failed: {e}"]}

    if "detect_drift" not in namespace:
        return {"passed": False, "reasons": ["detect_drift() not found"]}

    detect_drift = namespace["detect_drift"]

    try:
        result = detect_drift(old, new, threshold=0.1)
    except Exception as e:
        return {"passed": False, "reasons": [f"detect_drift crashed: {e}"]}

    if not isinstance(result, dict):
        return {"passed": False, "reasons": ["Return must be dict {col: float}"]}

    expected_cols = {"age", "income", "height"}
    if set(result.keys()) != expected_cols:
        return {"passed": False, "reasons": [f"Wrong keys: {set(result.keys())}"]}

    refs = {}
    for col in expected_cols:
        x, y = old[col].dropna().values, new[col].dropna().values
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        bins = np.linspace(lo, hi, 21)
        p, _ = np.histogram(x, bins=bins)
        q, _ = np.histogram(y, bins=bins)
        refs[col] = _kl(p, q)

    diffs = []
    for col, val in result.items():
        if not np.isfinite(val):
            return {"passed": False, "reasons": [f"Non-finite KL for {col}"]}
        diffs.append(abs(val - refs[col]))
    if max(diffs) > 1e-2:
        return {"passed": False, "reasons": ["KL mismatch (>1e-2)"]}

    drifted = [c for c, v in result.items() if v > 0.1]
    if "height" in drifted:
        return {"passed": False, "reasons": ["False positive: height drifted"]}
    if not {"age", "income"}.issubset(drifted):
        return {"passed": False, "reasons": ["Missed drift in age/income"]}

    return {"passed": True, "reasons": []}
