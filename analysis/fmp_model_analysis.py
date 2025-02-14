import subprocess
import json
import pandas as pd

TIMEOUT = 61

output_file = "results/fmp_model_analysis.csv"


def run_alloy(spec_path, cmd_index, output_file):
    cmd = f"java -jar alloy-metrics.jar alloyModelAnalysis {spec_path} {cmd_index} {output_file}"
    try:
        subprocess.run(cmd, shell=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"Timeout: {spec_path} {cmd_index}")


with open("data/json/fmp.json", "r") as f:
    data = [json.loads(line) for line in f]

# if output file exists, read it and skip the ones that are already done
try:
    existing_data = pd.read_csv(output_file)
    existing_data = existing_data.iloc[:, 0].tolist()
    existing_data = [x.split("/")[-1].split(".")[0] for x in existing_data]
except FileNotFoundError:
    existing_data = []

for d in data:
    if str(d["id"]) in existing_data:
        print(f"Skipping {d['id']}")
        continue
    spec_path = f'data/code/fmp/{d["id"]}.als'
    cmd = d["cmd"]
    if cmd is None:
        continue
    cmd = cmd - 1
    run_alloy(spec_path, cmd, output_file)
