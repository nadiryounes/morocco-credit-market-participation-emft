#!/bin/bash
source .venv/bin/activate
rm -f tmp_*.json

SEEDS=(2026 2027 2028 2029 2030)
MODELS=("LR" "CB" "EBM")

for m in "${MODELS[@]}"; do
    for i in {0..4}; do
        python3 run_single.py cv "$m" "$i" "${SEEDS[$i]}" &
    done
done

wait
echo "All CVs done!"

for m in "${MODELS[@]}"; do
    python3 run_single.py final "$m" &
done

wait
echo "All Finals done!"

python3 -c "
import pandas as pd, glob, json
# Aggregate CV
res = []
for f in glob.glob('tmp_cv_*.json'):
    with open(f) as fp: res.append(json.load(fp))
df = pd.DataFrame(res)
df.to_csv('ML4_INTERNAL_CLASS_METRICS.csv', index=False)
summ = df.groupby('Model').agg(['mean', 'std', 'min', 'max']).reset_index()
summ.columns = ['_'.join(col).strip('_') for col in summ.columns.values]
summ.to_csv('ML3_INTERNAL_NESTED_CV_PERFORMANCE.csv', index=False)

# Aggregate params
res = []
for f in glob.glob('tmp_param_*.json'):
    with open(f) as fp: res.append(json.load(fp))
pd.DataFrame(res).to_csv('ML5_FINAL_HYPERPARAMETERS.csv', index=False)

# Aggregate external
res = []
for f in glob.glob('tmp_ext_*.json'):
    with open(f) as fp: res.append(json.load(fp))
pd.DataFrame(res).to_csv('ML6_EXTERNAL_2023_PERFORMANCE.csv', index=False)
"
echo "Aggregation complete."
