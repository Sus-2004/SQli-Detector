# backend_api/predictor_core.py
import os
import re
import joblib
from typing import Tuple, Dict, Any

# --- Paths (assumes this file is in backend_api/) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml_pipeline", "artifacts")
ARTIFACTS_DIR = os.path.normpath(ARTIFACTS_DIR)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
VECT_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")

# --- Load model & vectorizer (fail early with clear message) ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError(
        f"Model or vectorizer not found in {ARTIFACTS_DIR}. "
        "Run the training script to produce model.pkl and vectorizer.pkl"
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# --- Rule-based patterns (fast safety net) ---
RULE_PATTERNS = [
    r"\bor\b\s+1\s*=\s*1",                  # OR 1=1
    r"(--|#)",                              # SQL comment tokens
    r"\bunion\b\s+select\b",                # UNION SELECT
    r"\bselect\b.\bfrom\b.\bpassword\b",  # selecting password column
    r"\bdrop\s+table\b",                    # DROP TABLE
    r"\bdelete\s+from\b",                   # DELETE FROM
    r"\binsert\s+into\b",                   # INSERT INTO
    r"\bupdate\b\s+.*\bset\b",              # UPDATE ... SET
    r";\s*$",                               # trailing semicolon (stacked query end)
    r";\s*",                                # semicolon anywhere (stacked queries)
    r"xp_cmdshell|\bexec\b|\bexecute\b",    # execution calls
    r"\bsleep\s*\(",                        # time-based injection
    r"benchmark\s*\(",                      # mysql benchmark-based
    r"information_schema",                  # schema enumeration
    r"load_file\s*\(",                      # file read attempts
    r"concat\s*\(",                         # concat usage in some exfiltration payloads
]

COMPILED_RULES = [re.compile(p, re.IGNORECASE) for p in RULE_PATTERNS]


# --- Helpers ---
def rule_check(query: str) -> Tuple[bool, str]:
    """Return (True, pattern) if any rule pattern matches the query."""
    s = (query or "").strip()
    for pat in COMPILED_RULES:
        if pat.search(s):
            return True, pat.pattern
    return False, None


def interpret_label(label: Any) -> bool:
    """
    Interpret model output as boolean: True => sqli, False => safe.
    Supports numeric (0/1) or string labels like 'sqli'/'benign'.
    """
    if isinstance(label, (int, float)):
        return int(label) == 1
    txt = str(label).lower()
    return txt in ("1", "sqli", "malicious", "true", "yes")


# --- Main API used by server ---
def predict_query(query: str) -> Dict[str, Any]:
    """
    Predict whether the query is SQLi.

    Returns a dict:
      {
        "label": "sqli" | "safe",
        "confidence": float | None,
        "reason": "rule:<pattern>" | "ml" | "model_error"
      }
    """
    q = (query or "").strip()

    # 1) Quick rule-based check
    is_sus, matched = rule_check(q)
    if is_sus:
        # immediate block and high confidence
        return {"label": "sqli", "confidence": 1.0, "reason": f"rule:{matched}"}

    # 2) ML-based check (fallback)
    try:
        X = vectorizer.transform([q])
    except Exception as e:
        # If transformation fails, be conservative and mark as sqli
        # (you can change to 'safe' if you prefer permissive fallback)
        print("Vectorizer transform error:", e)
        return {"label": "sqli", "confidence": None, "reason": "vectorize_error"}

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        print("Model predict error:", e)
        # conservative fallback — block if model misbehaves
        return {"label": "sqli", "confidence": None, "reason": "model_error"}

    # get confidence if available
    confidence = None
    try:
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))
    except Exception:
        confidence = None

    is_sqli = interpret_label(pred)
    return {"label": "sqli" if is_sqli else "safe", "confidence": confidence, "reason": "ml"}