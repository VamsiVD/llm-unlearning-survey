#!/bin/bash

# Automated benchmark runner for OpenUnlearning
# - Skips methods where checkpoint + eval already exist (resume-safe)
# - Isolates failures: one method failing won't abort the rest
# - Retries train once on failure (catches transient OOM)
# - Logs each method's stdout/stderr to logs/<METHOD>/
# - Prints pass/fail/skip summary at end

FORGET_SPLIT=forget10
RETAIN_SPLIT=retain90
HOLDOUT_SPLIT=holdout10
TASK_SUFFIX=tofu_1B
LOG_DIR=logs/benchmark
SAVES_DIR=saves
EVAL_BATCH_SIZE=8   # lower than default 32 to reduce VRAM pressure

# Each entry: "METHOD TRAINER EXPERIMENT [EXTRA_TRAIN_ARGS]"
METHODS_CONFIG=(
  "GradAscent GradAscent unlearn/tofu/default"
  "GradDiff   GradDiff   unlearn/tofu/default"
  "NPO        NPO        unlearn/tofu/default trainer.args.per_device_train_batch_size=1 trainer.args.gradient_accumulation_steps=4 trainer.args.eval_strategy=no trainer.args.optim=adamw_bnb_8bit"
  "SimNPO     SimNPO     unlearn/tofu/default"
  "IDK        DPO        unlearn/tofu/idk trainer.args.per_device_train_batch_size=1 trainer.args.gradient_accumulation_steps=4 trainer.args.eval_strategy=no trainer.args.optim=adamw_bnb_8bit"
  "RMU        RMU        unlearn/tofu/default"
)

mkdir -p "$LOG_DIR"

RESULTS=()  # "METHOD:STATUS" entries

run_step() {
  local label=$1
  local log=$2
  shift 2
  echo "  [CMD] $*" >> "$log"
  "$@" >> "$log" 2>&1
}

for ENTRY in "${METHODS_CONFIG[@]}"; do
  METHOD=$(echo $ENTRY | awk '{print $1}')
  TRAINER=$(echo $ENTRY | awk '{print $2}')
  EXPERIMENT=$(echo $ENTRY | awk '{print $3}')
  EXTRA_TRAIN_ARGS=$(echo $ENTRY | awk '{for(i=4;i<=NF;i++) printf $i" "; print ""}')

  TASK_NAME="${METHOD}_${TASK_SUFFIX}"
  CKPT_DIR="${SAVES_DIR}/unlearn/${TASK_NAME}"
  EVAL_DIR="${SAVES_DIR}/eval/${TASK_NAME}_eval"
  METHOD_LOG_DIR="${LOG_DIR}/${METHOD}"
  mkdir -p "$METHOD_LOG_DIR"
  TRAIN_LOG="${METHOD_LOG_DIR}/train.log"
  EVAL_LOG="${METHOD_LOG_DIR}/eval.log"

  echo "==============================="
  echo " $METHOD"
  echo "==============================="

  # ── TRAIN ──────────────────────────────────────────────
  if [ -d "$CKPT_DIR" ] && [ -f "${CKPT_DIR}/config.json" ]; then
    echo "  [SKIP] checkpoint exists: $CKPT_DIR"
    TRAIN_OK=true
  else
    echo "  [TRAIN] logging to $TRAIN_LOG"
    TRAIN_OK=false
    for attempt in 1 2; do
      echo "  [attempt $attempt]"
      if run_step "train" "$TRAIN_LOG" \
          python src/train.py --config-name=unlearn.yaml \
            experiment=${EXPERIMENT} \
            forget_split=${FORGET_SPLIT} retain_split=${RETAIN_SPLIT} \
            trainer=${TRAINER} task_name=${TASK_NAME} \
            +model.load_in_4bit=true \
            eval.tofu.batch_size=${EVAL_BATCH_SIZE} \
            trainer.args.gradient_checkpointing=true \
            ${EXTRA_TRAIN_ARGS}; then
        TRAIN_OK=true
        break
      else
        echo "  [FAIL] attempt $attempt failed — see $TRAIN_LOG"
        [ $attempt -lt 2 ] && echo "  [RETRY] waiting 10s..." && sleep 10
      fi
    done
    if ! $TRAIN_OK; then
      echo "  [ERROR] train failed after 2 attempts — skipping eval"
      RESULTS+=("${METHOD}:TRAIN_FAILED")
      continue
    fi
  fi

  # ── EVAL ───────────────────────────────────────────────
  if [ -f "${EVAL_DIR}/TOFU_EVAL.json" ]; then
    echo "  [SKIP] eval results exist: $EVAL_DIR"
    RESULTS+=("${METHOD}:SKIPPED")
  else
    echo "  [EVAL] logging to $EVAL_LOG"
    if run_step "eval" "$EVAL_LOG" \
        python src/eval.py --config-name=eval.yaml \
          experiment=eval/tofu/default \
          forget_split=${FORGET_SPLIT} \
          holdout_split=${HOLDOUT_SPLIT} \
          task_name=${TASK_NAME}_eval \
          model.model_args.pretrained_model_name_or_path=${CKPT_DIR} \
          eval.tofu.batch_size=${EVAL_BATCH_SIZE}; then
      echo "  [OK] eval done"
      RESULTS+=("${METHOD}:PASSED")
    else
      echo "  [ERROR] eval failed — see $EVAL_LOG"
      RESULTS+=("${METHOD}:EVAL_FAILED")
    fi
  fi

done

# ── SUMMARY ────────────────────────────────────────────────────────────────
echo ""
echo "==============================="
echo " BENCHMARK SUMMARY"
echo "==============================="
printf "  %-15s %s\n" "METHOD" "STATUS"
printf "  %-15s %s\n" "------" "------"
for ENTRY in "${RESULTS[@]}"; do
  METHOD=${ENTRY%%:*}
  STATUS=${ENTRY##*:}
  printf "  %-15s %s\n" "$METHOD" "$STATUS"
done
echo ""
