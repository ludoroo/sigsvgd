#!/usr/bin/env bash
set -euo pipefail
# Override REPO, OUT, ART, REF and NEW as needed. Run only on an idle host.
REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"
OUT=${OUT:-/tmp/sigsvgd-linux-validation}
ART=${ART:-$OUT/artifacts}
REF=${REF:-$HOME/.venvs/sigsvgd-reference/bin/python}
NEW=${NEW:-$HOME/.venvs/sigsvgd-candidate/bin/python}
mkdir -p "$OUT" "$ART"
run_case() {
    local stack=$1 device=$2 threads=$3 cores=$4 repeats=$5 suffix=${6:-}
    local py=$NEW name=$stack-$device-t$threads$suffix code
    local inputs=(--inputs "$OUT/legacy-cpu-t1.npz")
    if [[ $stack == legacy ]]; then py=$REF; fi
    if [[ $name == legacy-cpu-t1 ]]; then inputs=(); fi
    # Never reuse a stale output after failure.
    rm -f "$OUT/$name.npz" "$OUT/$name.json"
    if taskset -c "$cores" "$py" scripts/validate_path_signature.py \
        --backend signatory --device "$device" --threads "$threads" \
        --repeats "$repeats" "${inputs[@]}" --output "$OUT/$name.npz" \
        > "$ART/linux-$name-run.txt" 2>&1; then code=0; else code=$?; fi
    echo "exit_code=$code" >> "$ART/linux-$name-run.txt"
    if (( code != 0 )); then echo "Benchmark failed: $name" >&2; exit "$code"; fi
    cp "$OUT/$name.json" "$ART/linux-$name.json"
}
compare_pair() {
    local a=$1 b=$2 expected=$3 code log
    log=$ART/linux-compare-$a-$b.txt
    if "$NEW" scripts/validate_path_signature.py --compare "$OUT/$a.npz" "$OUT/$b.npz" \
        > "$log" 2>&1; then code=0; else code=$?; fi
    echo "exit_code=$code" >> "$log"
    local result='^[[:alnum:]_]+__(signature|vjp): (PASS|FAIL) max_abs=[0-9.eE+-]+ scaled_relative_norm=[0-9.eE+-]+ \(rtol=[0-9.eE+-]+, atol=[0-9.eE+-]+\)$'
    local stress='^high_channel_stress_8x10x7_f32__vjp: FAIL '
    local count passes fails bad
    count=$(grep -Ec "$result" "$log" || true)
    passes=$(grep -Ec '^[[:alnum:]_]+__(signature|vjp): PASS ' "$log" || true)
    fails=$(grep -Ec '^[[:alnum:]_]+__(signature|vjp): FAIL ' "$log" || true)
    bad=$(grep -Evc "$result|^exit_code=$code$" "$log" || true)
    if (( code != expected || count != 22 || bad != 0 )) ||
       { (( expected == 0 )) && (( passes != 22 || fails != 0 )); } ||
       { (( expected == 1 )) && { (( passes != 21 || fails != 1 )) || ! grep -Eq "$stress" "$log"; }; }; then
        echo "Unexpected comparison contents/exit: $a / $b ($code)" >&2
        exit 1
    fi
}
compare_suite() {
    local suffix=${1:-}
    compare_pair legacy-cpu-t1$suffix candidate-cpu-t1$suffix 0
    compare_pair legacy-cpu-t4$suffix candidate-cpu-t4$suffix 0
    compare_pair legacy-cpu-t1$suffix candidate-cuda-t1$suffix 1
    compare_pair candidate-cpu-t1$suffix candidate-cuda-t1$suffix 1
    compare_pair legacy-cpu-t1$suffix legacy-cuda-t1$suffix 1
    compare_pair legacy-cuda-t1$suffix candidate-cuda-t1$suffix 1
}
for spec in 'legacy cpu 1 2 50' 'candidate cpu 1 2 50' 'legacy cpu 4 2-5 50' 'candidate cpu 4 2-5 50' 'legacy cuda 1 2 200' 'candidate cuda 1 2 200'; do
    read -r stack device threads cores repeats <<< "$spec"
    run_case "$stack" "$device" "$threads" "$cores" "$repeats"
done
compare_suite
for round in 3 4; do
    order='candidate legacy'
    if [[ $round == 4 ]]; then order='legacy candidate'; fi
    for spec in 'cpu 1 2 50' 'cpu 4 2-5 50' 'cuda 1 2 200'; do
        read -r device threads cores repeats <<< "$spec"
        for stack in $order; do
            run_case "$stack" "$device" "$threads" "$cores" "$repeats" "-repeat$round"
        done
    done
    compare_suite "-repeat$round"
done
if MPLBACKEND=Agg "$NEW" -m pytest tests -q -rs > "$ART/linux-pytest.txt" 2>&1; then code=0; else code=$?; fi
echo "exit_code=$code" >> "$ART/linux-pytest.txt"
exit "$code"
