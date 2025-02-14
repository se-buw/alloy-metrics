import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta


def fmp_get_time_diffs_to_fix_parse_error():
    status_df = pd.read_csv("results/fmp_chain_longest_status.csv")
    edit_path_df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    fmp_data = []
    with open("data/json/fmp.json", "r") as f:
        for line in f:
            fmp_data.append(json.loads(line.strip()))

    # Convert JSON data into a dictionary indexed by ID
    fmp_data_by_id = {entry["id"]: entry for entry in fmp_data}

    results = []

    for _, row in status_df.iterrows():
        status_chain = eval(row["status_chain"])
        id_status_chain = int(row["id"])

        edit_path_row = edit_path_df[edit_path_df["id"] == id_status_chain]
        derivation_chain = eval(edit_path_row.iloc[0]["derivation_chain"])

        parse_error_indices = []
        for i, status in enumerate(status_chain):
            if status == "PARSEERROR":
                if not parse_error_indices or i != parse_error_indices[-1][-1] + 1:
                    parse_error_indices.append([i])
                else:
                    parse_error_indices[-1].append(i)

        time_diffs = []
        for indices in parse_error_indices:
            start_index = indices[0]
            end_index = indices[-1] + 1 if indices[-1] + 1 < len(status_chain) else -1
            if end_index == -1 or status_chain[end_index] == "PARSEERROR":
                continue

            start_id = int(derivation_chain[start_index])
            end_id = int(derivation_chain[end_index])

            start_time_str = fmp_data_by_id[start_id]["time"]
            end_time_str = fmp_data_by_id[end_id]["time"]

            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f")
            time_diff = end_time - start_time

            time_diffs.append(time_diff)

        results.append({"id": id_status_chain, "time_taken": time_diffs})

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/fmp_parse_error_fix_time_diffs.csv", index=False)

    all_time_diffs = [td for sublist in results_df["time_taken"] for td in sublist]
    # Convert to total seconds for statistical analysis
    all_time_seconds = [td.total_seconds() for td in all_time_diffs]

    # Calculate statistics
    min_time = np.min(all_time_seconds)  # Minimum
    max_time = np.max(all_time_seconds)  # Maximum
    mean_time = np.mean(all_time_seconds)  # Mean
    median_time = np.median(all_time_seconds)  # Median
    quartiles = np.percentile(
        all_time_seconds, [25, 75]
    )  # Quartiles (25th, 75th percentiles)

    min_timedelta = timedelta(seconds=min_time)
    max_timedelta = timedelta(seconds=max_time)
    mean_timedelta = timedelta(seconds=mean_time)
    median_timedelta = timedelta(seconds=median_time)
    quartiles_timedelta = [timedelta(seconds=q) for q in quartiles]

    overview = {
        "min": min_timedelta,
        "max": max_timedelta,
        "mean": mean_timedelta,
        "25th_percentile": quartiles_timedelta[0],
        "median": median_timedelta,
        "75th_percentile": quartiles_timedelta[1],
    }

    pd.DataFrame([overview]).to_csv(
        "results/tables/fmp_parse_error_fix_time_diffs_overview.csv", index=False
    )

    return all_time_seconds


def fmp_get_time_diffs_from_unsat_to_sat():
    status_df = pd.read_csv("results/fmp_chain_longest_status.csv")
    edit_path_df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    fmp_data = []
    with open("data/json/fmp.json", "r") as f:
        for line in f:
            fmp_data.append(json.loads(line.strip()))

    # Convert JSON data into a dictionary indexed by ID
    fmp_data_by_id = {entry["id"]: entry for entry in fmp_data}

    results = []
    for _, row in status_df.iterrows():
        status_chain = eval(row["status_chain"])
        id_status_chain = int(row["id"])

        edit_path_row = edit_path_df[edit_path_df["id"] == id_status_chain]
        derivation_chain = eval(edit_path_row.iloc[0]["derivation_chain"])
        unsat_indices = []
        for i, status in enumerate(status_chain):
            if status == "UNSAT":
                if not unsat_indices or i != unsat_indices[-1][-1] + 1:
                    unsat_indices.append([i])
                else:
                    unsat_indices[-1].append(i)

        time_diffs = []
        for indices in unsat_indices:
            start_index = indices[0]
            end_index = indices[-1] + 1 if indices[-1] + 1 < len(status_chain) else -1
            if end_index == -1 or status_chain[end_index] == "UNSAT":
                continue

            start_id = int(derivation_chain[start_index])
            end_id = int(derivation_chain[end_index])

            start_time_str = fmp_data_by_id[start_id]["time"]
            end_time_str = fmp_data_by_id[end_id]["time"]

            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f")

            time_diff = end_time - start_time
            time_diffs.append(time_diff)

        results.append({"id": id_status_chain, "time_taken": time_diffs})

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/fmp_unsat_to_sat_time_diffs.csv", index=False)

    all_time_diffs = [td for sublist in results_df["time_taken"] for td in sublist]
    # Convert to total seconds for statistical analysis
    all_time_seconds = [td.total_seconds() for td in all_time_diffs]

    # Calculate statistics
    min_time = np.min(all_time_seconds)  # Minimum
    max_time = np.max(all_time_seconds)  # Maximum
    mean_time = np.mean(all_time_seconds)  # Mean
    median_time = np.median(all_time_seconds)  # Median
    quartiles = np.percentile(
        all_time_seconds, [25, 75]
    )  # Quartiles (25th, 75th percentiles)

    min_timedelta = timedelta(seconds=min_time)
    max_timedelta = timedelta(seconds=max_time)
    mean_timedelta = timedelta(seconds=mean_time)
    median_timedelta = timedelta(seconds=median_time)
    quartiles_timedelta = [timedelta(seconds=q) for q in quartiles]

    overview = {
        "min": min_timedelta,
        "max": max_timedelta,
        "mean": mean_timedelta,
        "25th_percentile": quartiles_timedelta[0],
        "median": median_timedelta,
        "75th_percentile": quartiles_timedelta[1],
    }

    pd.DataFrame([overview]).to_csv(
        "results/tables/fmp_unsat_to_sat_time_diffs_overview.csv", index=False
    )

    return all_time_seconds

def main():
    fmp_get_time_diffs_from_unsat_to_sat()
    fmp_get_time_diffs_to_fix_parse_error()

