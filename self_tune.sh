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
    rp = (m.get("baselines", {}).get("random_prior") or {}).get("expected_hit_rate")
    pc = {k: v.get("predicted_count") for k, v in me["classification"]["per_class"].items()}
    print(f"hit={me['hit_rate']:.3f} macroF1={me['macro_f1']:.3f} "
          f"random_prior={(rp or 0):.3f} pred_counts={pc} tested_on={m.get('tested_on')}")
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
    -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
    -- models/model_tw.json
echo "self_tune: committed model snapshot — $SUMMARY"
