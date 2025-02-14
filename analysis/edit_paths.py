from collections import defaultdict
import json
import os
import pandas as pd
from numpy import log2
import pandas as pd
import ast
import Levenshtein
import numpy as np


def a4f_longest_chain_csv(input_file: str, output_file: str) -> None:
    """Compute the edit paths chain for the Alloy4Fun dataset."""
    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df = df[["_id", "derivationOf"]]
    df.columns = ["id", "parent"]

    # Build the derivation map
    derivation_map = dict(zip(df["id"], df["parent"]))

    def get_derivation_chain(start_id, derivation_map):
        chain = [start_id]
        current = start_id
        visited = set()
        while current in derivation_map and current not in visited:
            visited.add(current)
            next_id = derivation_map[current]
            if not next_id:  # Stop if the chain ends
                break
            chain.append(next_id)
            current = next_id
        return chain

    # Generate all chains
    all_chains = []
    for _, row in df.iterrows():
        chain = get_derivation_chain(row["id"], derivation_map)
        all_chains.append(chain)

    # Remove chains that are subsequences of longer chains
    longest_chains = []
    for chain in all_chains:
        if not any(
            set(chain).issubset(set(other_chain)) and chain != other_chain
            for other_chain in all_chains
        ):
            longest_chains.append(chain)

    # Write the longest chains to the output file
    with open(output_file, "w") as f:
        f.write("id,chain_length,derivation_chain\n")
        for chain in longest_chains:
            f.write(f"{chain[0]},{len(chain)},{' -> '.join(chain)}\n")


def fmp_longest_chain_csv(input_file: str, output_file: str):
    """Compute the edit paths chain for the FMP dataset."""
    data = []
    # Read and parse JSON lines
    with open(input_file) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df = df[["id", "parent"]]
    df.columns = ["id", "parent"]

    df["id"] = df["id"].apply(
        lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    )
    df["parent"] = df["parent"].apply(
        lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    )

    # Build the derivation map
    derivation_map = dict(zip(df["id"], df["parent"]))

    def get_derivation_chain(start_id, derivation_map):
        chain = [start_id]
        current = start_id
        visited = set()
        while current in derivation_map and current not in visited:
            visited.add(current)
            next_id = derivation_map[current]
            if not next_id:  # Stop if the chain ends
                break
            chain.append(next_id)
            current = next_id
        return chain

    # Generate all chains
    all_chains = []
    for _, row in df.iterrows():
        chain = get_derivation_chain(row["id"], derivation_map)
        all_chains.append(chain)

    # Remove chains that are subsequences of longer chains
    longest_chains = []
    for chain in all_chains:
        if not any(
            set(chain).issubset(set(other_chain)) and chain != other_chain
            for other_chain in all_chains
        ):
            longest_chains.append(chain)

    # Write the longest chains to the output file
    with open(output_file, "w") as f:
        f.write("id,chain_length,derivation_chain\n")
        for chain in longest_chains:
            f.write(f"{chain[0]},{len(chain)},{' -> '.join(chain)}\n")


def fmp_chain_to_list(input_file: str, output_file: str):
    """Convert the FMP dataset edit paths chain to a list."""
    df = pd.read_csv(input_file)
    df["id"] = df["id"].apply(
        lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    )
    for i, row in df.iterrows():
        chain = row["derivation_chain"].split(" -> ")
        chain.reverse()
        df.at[i, "derivation_chain"] = list(chain)
    df.to_csv(output_file, index=False)


def a4f_chain_to_list(input_file: str, output_file: str):
    """Convert the Alloy4Fun dataset edit paths chain to a list."""
    df = pd.read_csv(input_file)
    df["id"] = df["id"].apply(
        lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    )
    for i, row in df.iterrows():
        chain = row["derivation_chain"].split(" -> ")
        chain.reverse()
        df.loc[i, "chain_length"] = len(chain)
        df.at[i, "derivation_chain"] = list(chain)

    df.to_csv(output_file, index=False)


def save_edit_path_chain_overview():
    a4f_df = pd.read_csv(
        "results/a4f_individual_tasks_edit_paths_chain_list.csv",
    )
    fmp_df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    a4f_df["derivation_chain"] = a4f_df["derivation_chain"].apply(ast.literal_eval)
    a4f_df["chain_length"] = a4f_df["derivation_chain"].apply(len)
    a4f_chain_length = a4f_df["chain_length"]
    fmp_chain_length = fmp_df["chain_length"]
    a4f_status_df = pd.read_csv("results/a4f_individual_tasks_chain_status.csv")
    fmp_status_df = pd.read_csv("results/fmp_chain_longest_status.csv")

    a4f_status_df["status_chain"] = a4f_status_df["status_chain"].apply(
        ast.literal_eval
    )
    fmp_status_df["status_chain"] = fmp_status_df["status_chain"].apply(
        ast.literal_eval
    )

    a4f_status_df["filtered_status_chain"] = a4f_status_df["status_chain"].apply(
        lambda statuses: [status for status in statuses if status != "UNKNOWN"]
    )
    fmp_status_df["filtered_status_chain"] = fmp_status_df["status_chain"].apply(
        lambda statuses: [status for status in statuses if status != "UNKNOWN"]
    )

    a4f_status_df["has_parse_error"] = a4f_status_df["filtered_status_chain"].apply(
        lambda statuses: "PARSEERROR" in statuses
    )
    fmp_status_df["has_parse_error"] = fmp_status_df["filtered_status_chain"].apply(
        lambda statuses: "PARSEERROR" in statuses
    )

    a4f_status_df["all_parse_error"] = a4f_status_df["filtered_status_chain"].apply(
        lambda statuses: all(status == "PARSEERROR" for status in statuses)
    )
    fmp_status_df["all_parse_error"] = fmp_status_df["filtered_status_chain"].apply(
        lambda statuses: all(status == "PARSEERROR" for status in statuses)
    )

    a4f_status_df["has_unsat"] = a4f_status_df["filtered_status_chain"].apply(
        lambda statuses: "UNSAT" in statuses
    )
    fmp_status_df["has_unsat"] = fmp_status_df["filtered_status_chain"].apply(
        lambda statuses: "UNSAT" in statuses
    )

    a4f_status_df["all_unsat"] = a4f_status_df["filtered_status_chain"].apply(
        lambda statuses: all(status == "UNSAT" for status in statuses)
    )
    fmp_status_df["all_unsat"] = fmp_status_df["filtered_status_chain"].apply(
        lambda statuses: all(status == "UNSAT" for status in statuses)
    )

    a4f_overview = {
        "min": a4f_chain_length.min(),
        "max": a4f_chain_length.max(),
        "mean": a4f_chain_length.mean(),
        "25th percentile": a4f_chain_length.quantile(0.25),
        "median": a4f_chain_length.median(),
        "75th percentile": a4f_chain_length.quantile(0.75),
        "std": a4f_chain_length.std(),
        "count": a4f_chain_length.count(),
        ">=5%": (a4f_chain_length >= 5).sum() / a4f_chain_length.count(),
        "has_parse_error": a4f_status_df["has_parse_error"].sum(),
        "has_parse_error(%)": a4f_status_df["has_parse_error"].sum()
        / a4f_chain_length.count(),
        "all_parse_error": a4f_status_df["all_parse_error"].sum(),
        "all_parse_error(%)": a4f_status_df["all_parse_error"].sum()
        / a4f_chain_length.count(),
        "has_unsat": a4f_status_df["has_unsat"].sum(),
        "has_unsat(%)": a4f_status_df["has_unsat"].sum() / a4f_chain_length.count(),
        "all_unsat": a4f_status_df["all_unsat"].sum(),
        "all_unsat(%)": a4f_status_df["all_unsat"].sum() / a4f_chain_length.count(),
    }

    fmp_overview = {
        "min": fmp_chain_length.min(),
        "max": fmp_chain_length.max(),
        "mean": fmp_chain_length.mean(),
        "25th percentile": fmp_chain_length.quantile(0.25),
        "median": fmp_chain_length.median(),
        "75th percentile": fmp_chain_length.quantile(0.75),
        "std": fmp_chain_length.std(),
        "count": fmp_chain_length.count(),
        ">=5%": (fmp_chain_length >= 5).sum() / fmp_chain_length.count(),
        "has_parse_error": fmp_status_df["has_parse_error"].sum(),
        "has_parse_error(%)": fmp_status_df["has_parse_error"].sum()
        / fmp_chain_length.count(),
        "all_parse_error": fmp_status_df["all_parse_error"].sum(),
        "all_parse_error(%)": fmp_status_df["all_parse_error"].sum()
        / fmp_chain_length.count(),
        "has_unsat": fmp_status_df["has_unsat"].sum(),
        "has_unsat(%)": fmp_status_df["has_unsat"].sum() / fmp_chain_length.count(),
        "all_unsat": fmp_status_df["all_unsat"].sum(),
        "all_unsat(%)": fmp_status_df["all_unsat"].sum() / fmp_chain_length.count(),
    }

    overview = pd.DataFrame([a4f_overview, fmp_overview], index=["a4f", "fmp"]).T
    overview.to_csv("results/tables/edit_path_overview.csv", index=True)


def calculate_distance(file_1: str, file_2: str) -> float:
    """Calculate the Levenshtein distance between two files."""
    try:
        with open(file_1, encoding="utf-8") as f:
            s1 = f.read()
    except FileNotFoundError:
        # File not found means either the user starts with a clean editor or the previous run on a different tool selected
        # We consider this as a clean spec
        s1 = ""

    try:
        with open(file_2, encoding="utf-8") as f:
            s2 = f.read()
    except FileNotFoundError:
        # File not found means either the user starts with a clean editor or the previous run on a different tool selected
        s2 = ""

    distance = Levenshtein.distance(s1, s2)
    return distance


def calculate_halstead(metrices: list) -> float:
    """Calculate the Halstead difficulty and effort."""
    n1, n2, N1, N2 = metrices
    if n1 == 0 or n2 == 0:
        return 0, 0
    N = n1 + n2  # Program length
    n = N1 + N2  # Program vocabulary
    V = N * log2(n)  # Volume
    D = (n1 / 2) * (N2 / n2)  # Difficulty
    E = D * V  # Effort

    return D, float(E)


def fmp_chain_distance_halstead() -> None:
    analysis_file = "results/fmp_spec_analysis.csv"
    input_file = "results/fmp_edit_paths_chain_list.csv"
    analysis = pd.read_csv(analysis_file)
    if "\\" in analysis["spec"].iloc[0]:
        analysis["id"] = analysis["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        analysis["id"] = analysis["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    analysis = analysis[["id", "parseable", "eloc", "comments", "halstead"]]
    analysis["halstead"] = analysis["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    halstead_dict = dict(zip(analysis["id"], analysis["halstead"]))

    df = pd.read_csv(input_file)
    df["parsed_chain"] = df["derivation_chain"].apply(ast.literal_eval)

    data_dir = "data/code/fmp/"
    output_data = []
    for _, row in df.iterrows():
        id = row["id"]
        parsed_chain = row["parsed_chain"]
        chain_len = len(parsed_chain)
        # Skip rows with chains of length less than 3
        if chain_len < 3:
            continue
        distances = []
        halstead_difficulty = []
        halstead_effort = []
        for i in range(chain_len - 1):
            # Calculate distance between two files
            s1 = data_dir + str(parsed_chain[i]) + ".als"
            s2 = data_dir + str(parsed_chain[i + 1]) + ".als"
            distance = calculate_distance(s1, s2)
            distances.append(distance)
            # Calculate halstead difficulty and effort
            halstead = calculate_halstead(
                halstead_dict.get(str(parsed_chain[i]), [0, 0, 0, 0])
            )
            halstead_difficulty.append(halstead[0])
            halstead_effort.append(halstead[1])
        # Calculate halstead difficulty and effort for the last file in the chain
        halstead = calculate_halstead(
            halstead_dict.get(str(parsed_chain[-1]), [0, 0, 0, 0])
        )
        halstead_difficulty.append(halstead[0])
        halstead_effort.append(halstead[1])

        output_row = [
            id,
            chain_len,
            str(distances),
            str(halstead_difficulty),
            str(halstead_effort),
        ]
        output_data.append(output_row)

    output_df = pd.DataFrame(
        output_data,
        columns=[
            "id",
            "chain_len",
            "distances",
            "halstead_difficulty",
            "halstead_effort",
        ],
    )

    output_file = "results/fmp_edit_paths_chain_levenshtein_halstead.csv"
    output_df.to_csv(output_file, index=False)


def a4f_chain_distance_halstead():
    analysis_file = "results/a4f_spec_analysis.csv"
    input_file = "results/a4f_edit_paths_chain_list.csv"
    analysis = pd.read_csv(analysis_file)

    if "\\" in analysis["spec"].iloc[0]:
        analysis["id"] = analysis["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        analysis["id"] = analysis["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    analysis = analysis[["id", "parseable", "eloc", "comments", "halstead"]]
    analysis["halstead"] = analysis["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    halstead_dict = dict(zip(analysis["id"], analysis["halstead"]))

    df = pd.read_csv(input_file)
    df["parsed_chain"] = df["derivation_chain"].apply(ast.literal_eval)
    data_dir = "data/code/a4f/"
    output_data = []
    for _, row in df.iterrows():
        id = row["id"]
        parsed_chain = row["parsed_chain"]
        chain_len = len(parsed_chain)
        # Skip rows with chains of length less than 3
        if chain_len < 3:
            continue
        distances = []
        halstead_difficulty = []
        halstead_effort = []
        for i in range(chain_len - 1):
            # Calculate distance between two files
            s1 = data_dir + str(parsed_chain[i]) + ".als"
            s2 = data_dir + str(parsed_chain[i + 1]) + ".als"
            distance = calculate_distance(s1, s2)
            distances.append(distance)
            # Calculate halstead difficulty and effort
            halstead = calculate_halstead(
                halstead_dict.get(str(parsed_chain[i]), [0, 0, 0, 0])
            )
            halstead_difficulty.append(halstead[0])
            halstead_effort.append(halstead[1])
        # Calculate halstead difficulty and effort for the last file in the chain
        halstead = calculate_halstead(
            halstead_dict.get(str(parsed_chain[-1]), [0, 0, 0, 0])
        )
        halstead_difficulty.append(halstead[0])
        halstead_effort.append(halstead[1])
        output_row = [
            id,
            chain_len,
            str(distances),
            str(halstead_difficulty),
            str(halstead_effort),
        ]
        output_data.append(output_row)

    output_df = pd.DataFrame(
        output_data,
        columns=[
            "id",
            "chain_len",
            "distances",
            "halstead_difficulty",
            "halstead_effort",
        ],
    )

    output_file = "results/a4f_edit_paths_chain_levenshtein_halstead.csv"
    output_df.to_csv(output_file, index=False)


def a4f_per_task_chain_distance_halstead():
    analysis_file = "results/a4f_spec_analysis.csv"
    input_file = "results/a4f_individual_tasks_edit_paths_chain_list.csv"
    analysis = pd.read_csv(analysis_file)

    if "\\" in analysis["spec"].iloc[0]:
        analysis["id"] = analysis["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        analysis["id"] = analysis["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    analysis = analysis[["id", "parseable", "eloc", "comments", "halstead"]]
    analysis["halstead"] = analysis["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    halstead_dict = dict(zip(analysis["id"], analysis["halstead"]))

    df = pd.read_csv(input_file)
    df["parsed_chain"] = df["derivation_chain"].apply(ast.literal_eval)
    data_dir = "data/code/a4f/"
    output_data = []
    for _, row in df.iterrows():
        id = row["id"]
        parsed_chain = row["parsed_chain"]
        chain_len = len(parsed_chain)
        # Skip rows with chains of length less than 3
        if chain_len < 3:
            continue
        distances = []
        halstead_difficulty = []
        halstead_effort = []
        for i in range(chain_len - 1):
            # Calculate distance between two files
            s1 = data_dir + str(parsed_chain[i]) + ".als"
            s2 = data_dir + str(parsed_chain[i + 1]) + ".als"
            distance = calculate_distance(s1, s2)
            distances.append(distance)
            # Calculate halstead difficulty and effort
            halstead = calculate_halstead(
                halstead_dict.get(str(parsed_chain[i]), [0, 0, 0, 0])
            )
            halstead_difficulty.append(halstead[0])
            halstead_effort.append(halstead[1])
        # Calculate halstead difficulty and effort for the last file in the chain
        halstead = calculate_halstead(
            halstead_dict.get(str(parsed_chain[-1]), [0, 0, 0, 0])
        )
        halstead_difficulty.append(halstead[0])
        halstead_effort.append(halstead[1])
        output_row = [
            id,
            chain_len,
            str(distances),
            str(halstead_difficulty),
            str(halstead_effort),
        ]
        output_data.append(output_row)

    output_df = pd.DataFrame(
        output_data,
        columns=[
            "id",
            "chain_len",
            "distances",
            "halstead_difficulty",
            "halstead_effort",
        ],
    )

    output_file = "a4f_edit_paths_chain_levenshtein_halstead.csv"
    output_df.to_csv(output_file, index=False)


def a4f_chain_sat_unsat_status():
    a4f_json = "data/json/a4f.json"
    with open(a4f_json) as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df_cmd = df.dropna(subset=["cmd_i"])
    output = []
    base_spec_path = "data/code/a4f/"
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

    ma_res_df = pd.read_csv("results/a4f_model_analysis.csv")
    chain_df = pd.read_csv("results/a4f_edit_paths_chain_list.csv")

    if "\\" in ma_res_df["spec_path"].iloc[0]:
        ma_res_df["spec_path"] = ma_res_df["spec_path"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
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
    output_df.to_csv("results/a4f_chain_longest_status.csv", index=False)


def fmp_chain_sat_unsat_status():
    ma_res_df = pd.read_csv(
        "results/fmp_model_analysis.csv",
        header=None,
        names=["spec_path", "cmd", "result"],
    )
    chain_df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")

    if "\\" in ma_res_df["spec_path"].iloc[0]:
        ma_res_df["spec_path"] = ma_res_df["spec_path"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
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
    output_df.to_csv("results/fmp_chain_longest_status.csv", index=False)


def calculate_parseerror_fix_steps(status_chain):
    steps = []
    i = 0
    while i < len(status_chain):
        if status_chain[i] == "PARSEERROR":
            # Record the first occurrence of consecutive PARSEERRORs
            start = i
            while i + 1 < len(status_chain) and status_chain[i + 1] == "PARSEERROR":
                i += 1
            # Find the next non-PARSEERROR value
            for j in range(i + 1, len(status_chain)):
                if status_chain[j] != "PARSEERROR":
                    steps.append(j - start)
                    break
        i += 1
    return steps


def calculate_unsat_to_sat_steps(status_chain):
    steps = []
    i = 0
    while i < len(status_chain):
        if status_chain[i] == "UNSAT":
            # Skip to the last consecutive UNSAT
            while i + 1 < len(status_chain) and status_chain[i + 1] == "UNSAT":
                i += 1
            # Find the next SAT
            for j in range(i + 1, len(status_chain)):
                if status_chain[j] == "SAT":
                    steps.append(j - i)
                    break
        i += 1
    return steps


def save_steps_to_fix():
    # ------- Alloy4Fun -------
    a4f_status_df = pd.read_csv("results/a4f_chain_longest_status.csv")
    a4f_status_df["status_chain"] = a4f_status_df["status_chain"].apply(
        ast.literal_eval
    )
    a4f_status_df["parseerror_fix_steps"] = a4f_status_df["status_chain"].apply(
        calculate_parseerror_fix_steps
    )
    a4f_status_df["unsat_to_sat_steps"] = a4f_status_df["status_chain"].apply(
        calculate_unsat_to_sat_steps
    )
    a4f_status_df.to_csv(
        "results/a4f_steps_to_fix.csv",
        index=False,
        columns=["id", "parseerror_fix_steps", "unsat_to_sat_steps"],
    )
    a4f_parseerror_fix_steps = [
        step for steps in a4f_status_df["parseerror_fix_steps"] for step in steps
    ]
    a4f_unsat_to_sat_steps = [
        step for steps in a4f_status_df["unsat_to_sat_steps"] for step in steps
    ]

    # ------- FMP -------
    fmp_status_df = pd.read_csv("results/fmp_chain_longest_status.csv")
    fmp_status_df["status_chain"] = fmp_status_df["status_chain"].apply(
        ast.literal_eval
    )
    fmp_status_df["parseerror_fix_steps"] = fmp_status_df["status_chain"].apply(
        calculate_parseerror_fix_steps
    )
    fmp_status_df["unsat_to_sat_steps"] = fmp_status_df["status_chain"].apply(
        calculate_unsat_to_sat_steps
    )
    fmp_status_df.to_csv(
        "results/fmp_steps_to_fix.csv",
        index=False,
        columns=["id", "parseerror_fix_steps", "unsat_to_sat_steps"],
    )

    fmp_parseerror_fix_steps = [
        step for steps in fmp_status_df["parseerror_fix_steps"] for step in steps
    ]
    fmp_unsat_to_sat_steps = [
        step for steps in fmp_status_df["unsat_to_sat_steps"] for step in steps
    ]

    # ------- Overview -------
    a4f_parseerror_fix_overview = {
        "min": min(a4f_parseerror_fix_steps),
        "max": max(a4f_parseerror_fix_steps),
        "mean": sum(a4f_parseerror_fix_steps) / len(a4f_parseerror_fix_steps),
        "25th percentile": np.percentile(a4f_parseerror_fix_steps, 25),
        "median": np.median(a4f_parseerror_fix_steps),
        "75th percentile": np.percentile(a4f_parseerror_fix_steps, 75),
        "std": np.std(a4f_parseerror_fix_steps),
        "count": len(a4f_parseerror_fix_steps),
        ">2%": sum(1 for step in a4f_parseerror_fix_steps if step > 2)
        / len(a4f_parseerror_fix_steps),
    }

    a4f_unsat_to_sat_overview = {
        "min": min(a4f_unsat_to_sat_steps),
        "max": max(a4f_unsat_to_sat_steps),
        "mean": sum(a4f_unsat_to_sat_steps) / len(a4f_unsat_to_sat_steps),
        "25th percentile": np.percentile(a4f_unsat_to_sat_steps, 25),
        "median": np.median(a4f_unsat_to_sat_steps),
        "75th percentile": np.percentile(a4f_unsat_to_sat_steps, 75),
        "std": np.std(a4f_unsat_to_sat_steps),
        "count": len(a4f_unsat_to_sat_steps),
        ">2%": sum(1 for step in a4f_unsat_to_sat_steps if step > 2)
        / len(a4f_unsat_to_sat_steps),
    }

    fmp_parseerror_fix_overview = {
        "min": min(fmp_parseerror_fix_steps),
        "max": max(fmp_parseerror_fix_steps),
        "mean": sum(fmp_parseerror_fix_steps) / len(fmp_parseerror_fix_steps),
        "25th percentile": np.percentile(fmp_parseerror_fix_steps, 25),
        "median": np.median(fmp_parseerror_fix_steps),
        "75th percentile": np.percentile(fmp_parseerror_fix_steps, 75),
        "std": np.std(fmp_parseerror_fix_steps),
        "count": len(fmp_parseerror_fix_steps),
        ">3%": sum(1 for step in fmp_parseerror_fix_steps if step > 3)
        / len(fmp_parseerror_fix_steps),
    }

    fmp_unsat_to_sat_overview = {
        "min": min(fmp_unsat_to_sat_steps),
        "max": max(fmp_unsat_to_sat_steps),
        "mean": sum(fmp_unsat_to_sat_steps) / len(fmp_unsat_to_sat_steps),
        "25th percentile": np.percentile(fmp_unsat_to_sat_steps, 25),
        "median": np.median(fmp_unsat_to_sat_steps),
        "75th percentile": np.percentile(fmp_unsat_to_sat_steps, 75),
        "std": np.std(fmp_unsat_to_sat_steps),
        "count": len(fmp_unsat_to_sat_steps),
        ">3%": sum(1 for step in fmp_unsat_to_sat_steps if step > 3)
        / len(fmp_unsat_to_sat_steps),
    }

    overview = pd.DataFrame(
        [
            a4f_parseerror_fix_overview,
            a4f_unsat_to_sat_overview,
            fmp_parseerror_fix_overview,
            fmp_unsat_to_sat_overview,
        ],
        index=[
            "A4F Parseerror Fix",
            "A4F UNSAT to SAT",
            "FMP Parseerror Fix",
            "FMP UNSAT to SAT",
        ],
    )

    overview.to_csv("results/tables/parseerror_unsat_to_sat_overview.csv", index=True)


def find_parseerror_pairs(derivation_chain, status_chain):
    parseerror_pairs = []
    for i in range(1, len(status_chain)):
        if status_chain[i] == "PARSEERROR" and status_chain[i - 1] == "PARSEERROR":
            parseerror_pairs.append((derivation_chain[i - 1], derivation_chain[i]))

    return parseerror_pairs


def not_parseerror_pairs(derivation_chain, status_chain):
    not_parseerror_pairs = []
    for i in range(1, len(status_chain)):
        if (
            status_chain[i] != "PARSEERROR"
            and status_chain[i - 1] != "PARSEERROR"
            and status_chain[i] != "UNKNOWN"
            and status_chain[i - 1] != "UNKNOWN"
        ):
            not_parseerror_pairs.append((derivation_chain[i - 1], derivation_chain[i]))

    return not_parseerror_pairs


def a4f_parse_error_distance_vs_not_parse_error():
    df1 = pd.read_csv("results/a4f_edit_paths_chain_list.csv")
    df2 = pd.read_csv("results/a4f_chain_longest_status.csv")
    df = pd.merge(df1, df2, on="id")

    def calculate_distances_for_pairs(pairs):
        base_spec_path = "data/code/a4f/"
        distances = []
        for pair in pairs:
            file_1 = base_spec_path + pair[0] + ".als"
            file_2 = base_spec_path + pair[1] + ".als"
            distance = calculate_distance(file_1, file_2)
            distances.append(distance)
        return distances

    df["parseerror_pairs"] = df.apply(
        lambda row: find_parseerror_pairs(
            ast.literal_eval(row["derivation_chain"]),
            ast.literal_eval(row["status_chain"]),
        ),
        axis=1,
    )

    df["not_parseerror_pairs"] = df.apply(
        lambda row: not_parseerror_pairs(
            ast.literal_eval(row["derivation_chain"]),
            ast.literal_eval(row["status_chain"]),
        ),
        axis=1,
    )

    df["parseerror_distances"] = df["parseerror_pairs"].apply(
        calculate_distances_for_pairs
    )
    df["not_parseerror_distances"] = df["not_parseerror_pairs"].apply(
        calculate_distances_for_pairs
    )
    df = df[["id", "parseerror_distances", "not_parseerror_distances"]]
    df.to_csv("results/a4f_parseerror_pairs.csv", index=False)


def fmp_parse_error_distance_vs_not_parse_error():
    df1 = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    df2 = pd.read_csv("results/fmp_chain_longest_status.csv")
    df = pd.merge(df1, df2, on="id")

    def calculate_distances_for_pairs(pairs):
        base_spec_path = "data/code/fmp/"
        distances = []
        for pair in pairs:
            file_1 = base_spec_path + pair[0] + ".als"
            file_2 = base_spec_path + pair[1] + ".als"
            distance = calculate_distance(file_1, file_2)
            distances.append(distance)
        return distances

    df["parseerror_pairs"] = df.apply(
        lambda row: find_parseerror_pairs(
            ast.literal_eval(row["derivation_chain"]),
            ast.literal_eval(row["status_chain"]),
        ),
        axis=1,
    )

    df["not_parseerror_pairs"] = df.apply(
        lambda row: not_parseerror_pairs(
            ast.literal_eval(row["derivation_chain"]),
            ast.literal_eval(row["status_chain"]),
        ),
        axis=1,
    )

    df["parseerror_distances"] = df["parseerror_pairs"].apply(
        calculate_distances_for_pairs
    )
    df["not_parseerror_distances"] = df["not_parseerror_pairs"].apply(
        calculate_distances_for_pairs
    )
    df = df[["id", "parseerror_distances", "not_parseerror_distances"]]
    df.to_csv("results/fmp_parseerror_pairs.csv", index=False)


def a4f_individual_tasks_edit_paths_chain_list():
    input_files = os.listdir("data/json/a4f/")
    input_files = [f"data/json/a4f/{input_file}" for input_file in input_files]
    data = []
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))

    data_dict = {item["_id"]: item for item in data}

    chain_df = pd.read_csv("results/a4f_edit_paths_chain_list.csv")
    chain_df = chain_df[["derivation_chain"]]
    chain_df["derivation_chain"] = chain_df["derivation_chain"].apply(ast.literal_eval)

    df_rows = []
    for _, row in chain_df.iterrows():
        chain_len = len(row["derivation_chain"])
        root_node = row["derivation_chain"][0]
        grouped_chains = defaultdict(list)

        for i in range(1, chain_len):
            node = row["derivation_chain"][i]
            try:
                node_cmd_i = data_dict[node]["cmd_i"]
            except KeyError:
                continue
            # node_cmd_n = data_dict[node]["cmd_n"]
            node_original = data_dict[node]["_id"]
            grouped_chains[node_cmd_i].append(node_original)

        for cmd_i, nodes in grouped_chains.items():
            df_rows.append({"id": root_node, "cmd_i": cmd_i, "derivation_chain": nodes})

    # Create a DataFrame from the list of rows
    grouped_chains_df = pd.DataFrame(df_rows)
    grouped_chains_df.to_csv(
        "results/a4f_individual_tasks_edit_paths_chain_list.csv", index=False
    )


def fmp_unique_starting_nodes():
    df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    df["derivation_chain"] = df["derivation_chain"].apply(ast.literal_eval)
    res = set()
    for _, row in df.iterrows():
        chain = row["derivation_chain"][1]  # first one is
        res.add(chain)
    print(f"Number of unique starting nodes: {len(res)}")


def a4f_unique_starting_nodes():
    df = pd.read_csv("results/a4f_edit_paths_chain_list.csv")
    df["derivation_chain"] = df["derivation_chain"].apply(ast.literal_eval)
    res = set()
    for _, row in df.iterrows():
        chain = row["derivation_chain"][0]
        res.add(chain)
    print(f"Number of unique starting nodes: {len(res)}")


def main():
    # ------- Alloy4Fun -------
    input_file = "data/json/a4f.json"
    output_file = "results/a4f_edit_paths_chain_overview.csv"
    a4f_longest_chain_csv(input_file, output_file)
    a4f_chain_to_list(
        "results/a4f_edit_paths_chain_overview.csv",
        "results/a4f_edit_paths_chain_list.csv",
    )
    a4f_chain_sat_unsat_status()
    a4f_chain_distance_halstead()

    # ------- FMP -------
    input_file = "data/json/fmp.json"
    output_file = "results/fmp_edit_paths_chain_overview.csv"
    fmp_longest_chain_csv(input_file, output_file)
    fmp_chain_to_list(
        "results/fmp_edit_paths_chain_overview.csv",
        "results/fmp_edit_paths_chain_list.csv",
    )
    fmp_chain_sat_unsat_status()
    fmp_chain_distance_halstead()

    # ------- Statistics -------
    save_edit_path_chain_overview()
    save_steps_to_fix()

    a4f_individual_tasks_edit_paths_chain_list()
