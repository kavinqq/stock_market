#!/usr/bin/env bash
# Daily self-tuning: retrain TW model -> honest OOS deploy gate -> promote iff it
# passes -> commit the deployed model as a version record. Called by launchd
# (com.kuanlin.stockmarket.selftune) each weekday 20:00. The scoped `git add`
# stages ONLY models/model_tw.json, so unrelated working-tree changes are never
# swept into these automated commits. Commits land on the current branch; no push.
set -uo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
ROOT="/Users/kuanlin/stock_market"
PY="$ROOT/.venv/bin/python"
cd "$ROOT" || exit 1

echo "=== self_tune $(date '+%F %T %Z') ==="
"$PY" train.py --market tw --out models/candidate_auto \
    --promote-if-passes --promote-dir models

# Version-record the deployed model only when it actually changed.
git add -- models/model_tw.json
if git diff --cached --quiet -- models/model_tw.json; then
    echo "self_tune: model unchanged — nothing to commit"
    exit 0
fi

SUMMARY=$("$PY" - <<'PY'
import json
try:
    m = json.load(open("models/candidate_auto/metrics_tw.json"))
    me = m["metrics"]
    b = m.get("baselines", {})
    # Quote the rolling-origin number first: it is what the deploy gate trusts.
    # The single held-out window swings several points with where the split
    # lands (0.468 fixed vs 0.441 rolling on the 2026-09-14 retrain).
    r = m.get("rolling_validation") or {}
    parts = []
    if r:
        rb = r.get("baselines", {})
        best = max([v for v in rb.values() if v is not None], default=0.0)
        sig = (r.get("significance") or {}).get("vs_always_neutral_mcnemar", {})
        parts.append(
            f"rolling hit={r['hit_rate']:.3f} (n={r['pooled_sample_size']}, "
            f"won {r.get('windows_won')}/{r.get('windows_total')}) "
            f"vs best-constant {best:.3f} p={sig.get('p_value', float('nan')):.3f}"
        )
    pc = {k: v.get("predicted_count") for k, v in me["classification"]["per_class"].items()}
    parts.append(
        f"fixed-split hit={me['hit_rate']:.3f} macroF1={me['macro_f1']:.3f} "
        f"brier={me.get('brier', float('nan')):.4f}/"
        f"{me.get('brier_baseline', float('nan')):.4f} "
        f"always_neutral={(b.get('always_neutral') or {}).get('hit_rate', 0):.3f} "
        f"pred_counts={pc} tested_on={m.get('tested_on')}"
    )
    print(" | ".join(parts))
except Exception as e:
    print(f"(metrics summary unavailable: {e})")
PY
)

# Pathspec on commit => ONLY models/model_tw.json is committed, even if the user
# happens to have other changes staged. Any such staged changes stay staged.
git commit -q \
    -m "chore(model): TW self-tune snapshot $(date +%F)" \
    -m "$SUMMARY" \
    -m "Automated retrain+gate+promote via com.kuanlin.stockmarket.selftune." \
    -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    -- models/model_tw.json
echo "self_tune: committed model snapshot — $SUMMARY"
