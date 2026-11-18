#!/bin/zsh
set -euo pipefail

cd "/Users/lianshuang/Documents/ChatGPT/最新论文 2"

PYTHON=".venv_ultrabert311/bin/python"
SCRIPT="news_polarity_experiment/run_ultrabert_quarter_eval.py"
DATA="news_polarity_experiment/artifacts/splits_ultrabert_clean_70_15_15"
BASE="news_polarity_experiment/artifacts/ultrabert_dev_seed42_len1024_epochs5_6/best_model"

while kill -0 21771 2>/dev/null; do
  print -r -- "{\"status\":\"waiting_for_existing_mps_job\",\"pid\":21771,\"time\":\"$(date -Iseconds)\"}"
  sleep 60
done

print -r -- "{\"status\":\"starting_sequence\",\"time\":\"$(date -Iseconds)\"}"

"$PYTHON" "$SCRIPT" \
  --data-dir "$DATA" \
  --output-dir news_polarity_experiment/artifacts/ultrabert_opt_e6_len2048_lr2e6 \
  --resume-from "$BASE" \
  --max-length 2048 \
  --learning-rate 2e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --quarter-patience 2 \
  --seed 42 \
  --log-every-steps 32

"$PYTHON" "$SCRIPT" \
  --data-dir "$DATA" \
  --output-dir news_polarity_experiment/artifacts/ultrabert_opt_e6_len2048_lr5e6 \
  --resume-from "$BASE" \
  --max-length 2048 \
  --learning-rate 5e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --quarter-patience 2 \
  --seed 42 \
  --log-every-steps 32

"$PYTHON" "$SCRIPT" \
  --data-dir "$DATA" \
  --output-dir news_polarity_experiment/artifacts/ultrabert_opt_e6_len2048_lr2e6_freeze12 \
  --resume-from "$BASE" \
  --max-length 2048 \
  --learning-rate 2e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --freeze-bottom-layers 12 \
  --quarter-patience 2 \
  --seed 42 \
  --log-every-steps 32

print -r -- "{\"status\":\"sequence_complete\",\"time\":\"$(date -Iseconds)\"}"
