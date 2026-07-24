#!/usr/bin/env bash

# Iterate over all languages and models from new_results.txt,
# run main.py 3 times per (model, language), log outputs, and append to a shared CSV.

set -u

# Configurable parameters
N_REQUESTS=100
N_RESTARTS=5
N_PAR=5
REPEATS=3
RESULTS_CSV="all_runs_results.csv"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

# Initialize CSV with header if it doesn't exist
if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "timestamp,model,language,run_idx,n_requests,n_restarts,n_par,asr_gpt,asr_llama,asr_rules,json_path,status" > "$RESULTS_CSV"
fi

# Helper: trim whitespace
trim() {
  sed -e 's/^\s\+//' -e 's/\s\+$//'
}

MODELS=(
  "Qwen/Qwen3.5-9B"
  "gemma-4-31B-it"
  "Llama-3.3-70B-Instruct-Turbo"
)

LANGUAGES=(
  "english"
  "italian"
  "polish"
  "arabic"
  "japanese"
  "welsh"
  "chinese"
)

for current_model in "${MODELS[@]}"; do
  for language in "${LANGUAGES[@]}"; do
    for ((run=1; run<=REPEATS; run++)); do
      ts="$(date +%Y%m%d_%H%M%S)"
      log_file="$LOG_DIR/run_${ts}_model=$(echo "$current_model" | sed 's#[/ ]#_#g')_lang=${language}_run=${run}.log"

      # Build command
      if [[ "$language" == "english" ]]; then
        # English baseline: do not pass --lang
        cmd=(python main.py --target_model="$current_model" --n_requests "$N_REQUESTS" --n_restarts "$N_RESTARTS" --n_par "$N_PAR")
        json_lang="None"
      else
        cmd=(python main.py --target_model="$current_model" --n_requests "$N_REQUESTS" --n_restarts "$N_RESTARTS" --n_par "$N_PAR" --lang="$language")
        json_lang="$language"
      fi

      echo "[INFO] ${ts} Running model='$current_model' lang='$language' run=${run}" | tee -a "$log_file"

      # Run and capture exit code
      set +e
      "${cmd[@]}" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2)
      exit_code=$?
      set -e

      # Default metrics
      asr_gpt=""
      asr_llama=""
      asr_rules=""
      status="OK"

      # Parse the final ASR line from log: asr_gpt=..%, asr_llama=..%, asr_rules=..%
      final_line=$(grep -E 'asr_gpt=.*asr_llama=.*asr_rules=' "$log_file" | tail -n 1) || true
      if [[ -n "$final_line" ]]; then
        # Extract numeric percentages (drop % sign)
        asr_gpt=$(echo "$final_line"  | sed -E 's/.*asr_gpt=([0-9]+)%.*/\1/')
        asr_llama=$(echo "$final_line" | sed -E 's/.*asr_llama=([0-9]+)%.*/\1/')
        asr_rules=$(echo "$final_line" | sed -E 's/.*asr_rules=([0-9]+)%.*/\1/')
      fi

      # Locate the newest JSON artifact matching this run's params
      str_model="${current_model//\//_}"
      json_path=""
      candidate=$(ls -t jailbreak_artifacts/*-model="${str_model}"-lang="${json_lang}"-n_requests="${N_REQUESTS}"-n_restarts="${N_RESTARTS}".json 2>/dev/null | head -n1) || true
      if [[ -n "${candidate:-}" ]]; then
        json_path="$candidate"
      fi

      if [[ $exit_code -ne 0 ]]; then
        status="ERROR_${exit_code}"
      fi

      # Append CSV row
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${current_model//,/;},$language,$run,$N_REQUESTS,$N_RESTARTS,$N_PAR,$asr_gpt,$asr_llama,$asr_rules,${json_path},$status" >> "$RESULTS_CSV"

    done
  done
done

echo "Done. Aggregated results in: $RESULTS_CSV"
