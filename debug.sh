LOG_DIR=/localdisk/ssrivas9/multilingual-eleuther/logs; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S); LOG="$LOG_DIR/vie_run_${TS}.log"
ln -sfn "$LOG" "$LOG_DIR/vie_run_latest.log"

export goldfish=True
export PATH=/localdisk/ssrivas9/miniconda3/envs/multi/bin:$PATH
export LD_LIBRARY_PATH=/localdisk/ssrivas9/miniconda3/envs/multi/lib:${LD_LIBRARY_PATH:-}

python -c 'import transformers, torch; print("env ok")' || { echo "env check failed"; exit 1; }

stdbuf -oL -eL bash /localdisk/ssrivas9/multilingual-eleuther/launch_monolingual_vie.sh |& tee -a "$LOG"