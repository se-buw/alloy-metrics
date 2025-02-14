import ast
import os
import json
import subprocess
import pandas as pd
from numpy import log2
import tempfile


def calculate_halstead(metrices: list) -> float:
    n1, n2, N1, N2 = metrices
    if n1 == 0 or n2 == 0:
        return 0, 0
    N = n1 + n2  # Program length
    n = N1 + N2  # Program vocabulary
    V = N * log2(n)  # Volume
    D = (n1 / 2) * (N2 / n2)  # Difficulty
    E = D * V  # Effort

    return D, float(E)


input_files = os.listdir("data/json/a4f/")
input_files = [f"data/json/a4f/{input_file}" for input_file in input_files]
output_data = []

for input_file in input_files:
    oracle_id = input_file.split("/")[-1].split(".")[0]
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    data_dict = df.set_index("_id").to_dict(orient="index")
    oracle_code = data_dict[oracle_id]["code"]
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(oracle_code)
        temp_file_path = temp_file.name
    cmd = ["java", "-jar", "alloy-metrics.jar", "analyzeHalstead", temp_file_path]
    oracle_hal = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    oracle_hal = oracle_hal.stdout.decode()
    oracle_hal = ast.literal_eval(oracle_hal)
    oracle_hal = calculate_halstead(oracle_hal)
    os.remove(temp_file_path)

    for _, row in df.iterrows():
        sub_id = row["_id"]
        code = row["code"]
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        cmd = ["java", "-jar", "alloy-metrics.jar", "analyzeHalstead", temp_file_path]
        sub_hal = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        sub_hal = sub_hal.stdout.decode()
        sub_hal = ast.literal_eval(sub_hal)
        sub_hal = calculate_halstead(sub_hal)
        os.remove(temp_file_path)
        if sub_hal[0] > oracle_hal[0] or sub_hal[1] > oracle_hal[1]:
            print(f"Error: {sub_id} has higher halstead than {oracle_id}")
            output_data.append(
                [
                    oracle_id,
                    sub_id,
                    sub_hal[0] - oracle_hal[0],
                    sub_hal[1] - oracle_hal[1],
                ]
            )

output_df = pd.DataFrame(
    output_data,
    columns=[
        "oracle_id",
        "sub_id",
        "difficulty_diff",
        "effort_diff",
    ],
)
output_df.to_csv("results/a4f_halstead_oracle_vs_submission.csv", index=False)
