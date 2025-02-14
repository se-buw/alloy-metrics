import json
import os
import pandas as pd
import ast


def prepare_data():
    """Prepare the data for the analysis of the Alloy4Fun models."""
    input_files = os.listdir("data/json/a4f/")
    input_files = [f"data/json/a4f/{input_file}" for input_file in input_files]
    data = []
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    # find where there are no cmd_i
    df_no_cmd = df[df["cmd_i"].isnull()]
    df_cmd = df.dropna(subset=["cmd_i"])
    output = []

    base_spec_path = "data/alloy4fun/code/"
    for _, row in df_cmd.iterrows():
        spec_path = base_spec_path + row["_id"] + ".als"
        cmd = row["cmd_i"] - 1
        sat = row["sat"]
        if sat == 1:
            res = "SAT"
        elif sat == -1:
            res = "PARSEERROR"
        else:
            res = "UNSAT"
        output.append([spec_path, cmd, res])

    output_df = pd.DataFrame(output, columns=["spec_path", "cmd", "result"])
    output_df.to_csv("results/a4f_model_analysis.csv", index=False)


def a4f_per_task_status():
    """Compute the status of each task in the derivation chain."""
    ma_res_df = pd.read_csv("results/a4f_model_analysis.csv")
    chain_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    ma_res_df["spec_path"] = ma_res_df["spec_path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )

    spec_dict = dict(zip(ma_res_df["spec_path"], ma_res_df["result"]))
    chain_df["derivation_chain"] = chain_df["derivation_chain"].apply(ast.literal_eval)

    output_data = []

    for _, row in chain_df.iterrows():
        chain = row["derivation_chain"]
        id = row["id"]
        res_chain = []
        for i in range(len(chain)):
            spec_id = chain[i]
            if spec_id not in spec_dict:
                res_chain.append("UNKNOWN")
                continue
            res = spec_dict[spec_id]
            res_chain.append(res)
        output_data.append([id, res_chain])

    output_df = pd.DataFrame(output_data, columns=["id", "status_chain"])
    output_df.to_csv("results/a4f_individual_tasks_chain_status.csv", index=False)


def main():
    prepare_data()
    a4f_per_task_status()
