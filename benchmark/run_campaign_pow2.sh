#!/bin/bash
# Power-of-two campaign: exp1 sweep 1024..16384, exp2/exp3 at 16384 (np 1..16).
# Writes to benchmark/results_pow2 so the existing benchmark/results is kept.
cd /home/mpiuser/para
HF=/home/mpiuser/para/myhosts
MCA="--mca btl_tcp_if_include 172.20.10.0/28 --mca oob_tcp_if_include 172.20.10.0/28"
BIN=/home/mpiuser/para/pdsdbscan
RES=/home/mpiuser/para/benchmark/results_pow2
mkdir -p "$RES"; LOG="$RES/campaign.log"; : > "$LOG"; rm -f "$RES/DONE"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local np=$1 N=$2 to=$3 out w c m
  out=$(timeout $to mpirun --hostfile "$HF" $MCA --oversubscribe --wdir /home/mpiuser/para -np $np "$BIN" --n $N --eps 0.5 --min-pts 5 2>/dev/null)
  w=$(echo "$out"|grep "wall time"|grep -oE '[0-9]+\.[0-9]+'); c=$(echo "$out"|grep "max compute"|grep -oE '[0-9]+\.[0-9]+'); m=$(echo "$out"|grep "max comm"|grep -oE '[0-9]+\.[0-9]+')
  echo "${w:-NA} ${c:-NA} ${m:-NA}"; }

log "=== EXP1: runtime vs N (np=8, 4 machines) ==="
echo "n_procs,n_points,wall_s,compute_s,comm_s" > "$RES/exp1_runtime_vs_n.csv"
for N in 1024 2048 4096 8192 16384; do
  read w c m <<< "$(run 8 $N 400)"; log "exp1 N=$N wall=$w comp=$c comm=$m"
  [ "$w" != "NA" ] && echo "8,$N,$w,$c,$m" >> "$RES/exp1_runtime_vs_n.csv"
done
NT=16384; log "N_target=$NT (2^14)"

log "=== EXP2: granularity (np=8 N=$NT --timing) ==="
TF="$RES/exp2_timing.txt"
timeout 400 mpirun --hostfile "$HF" $MCA --oversubscribe --wdir /home/mpiuser/para -np 8 "$BIN" --n $NT --eps 0.5 --min-pts 5 --timing > "$TF" 2>/dev/null
echo "n_procs,n_points,timing_file" > "$RES/exp2_granularity.csv"; echo "8,$NT,$TF" >> "$RES/exp2_granularity.csv"; log "exp2 saved"

log "=== EXP3: speedup (N=$NT, procs 1 2 4 8 16) ==="
echo "n_procs,n_points,wall_s,compute_s,comm_s" > "$RES/exp3_speedup.csv"
for np in 1 2 4 8 16; do
  read w c m <<< "$(run $np $NT 760)"; log "exp3 np=$np N=$NT wall=$w comp=$c comm=$m"
  [ "$w" != "NA" ] && echo "$np,$NT,$w,$c,$m" >> "$RES/exp3_speedup.csv"
done

log "=== correctness (ARI) ==="
echo "n_points,procs,seq_clusters,par_clusters,ari" > "$RES/correctness.csv"
for N in 2048 4096; do
  o=$(timeout 300 mpirun --hostfile "$HF" $MCA --oversubscribe --wdir /home/mpiuser/para -np 8 "$BIN" --n $N --eps 0.5 --min-pts 5 --verify 2>/dev/null)
  sc=$(echo "$o"|grep "seq clusters"|grep -oE '[0-9]+'); pc=$(echo "$o"|grep -E "^  clusters"|grep -oE '[0-9]+'|head -1); ari=$(echo "$o"|grep "ARI"|grep -oE '[0-9]+\.[0-9]+')
  log "correctness N=$N seq=$sc par=$pc ari=$ari"; echo "$N,8,$sc,$pc,$ari" >> "$RES/correctness.csv"
done

log "=== PLOTS (split comm/compute) ==="
python3 benchmark/plot_split.py --res "$RES" --procs "np=8 (4 machines, N=16384)" >> "$LOG" 2>&1
log "=== CAMPAIGN DONE ==="; touch "$RES/DONE"
