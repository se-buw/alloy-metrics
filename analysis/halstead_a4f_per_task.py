"""
This script calculates Halstead metrics for the partitioned edit paths of the A4F dataset.
"""

import json
import ast
import os
import subprocess
import tempfile
import pandas as pd
import re
from edit_paths import calculate_halstead

ignored_predicates = ["bla", "x", "test", "check$1", "run$1"]

map_9jPK8KBWzjFmBx4Hb = {
    7: "prop8",
    9: "prop10",
    6: "prop7",
    18: "prop19",
    10: "prop11",
    14: "prop15",
    13: "prop14",
    4: "prop5",
    1: "prop2",
    19: "prop20",
    3: "prop4",
    17: "prop18",
    12: "prop13",
    5: "prop6",
    8: "prop9",
    11: "prop12",
    16: "prop17",
    2: "prop3",
    15: "prop16",
    0: "prop1",
}

map_aTwuoJgesSd8hXXEP = {
    8: "inv9",
    6: "inv7",
    7: "inv8",
    9: "inv10",
    5: "inv6",
    4: "inv5",
    3: "inv4",
    1: "inv2",
    0: "inv1",
    2: "inv3",
}

map_bNCCf9FMRZoxqobfX = {
    6: "inv7",
    3: "inv4",
    7: "inv8",
    5: "inv6",
    4: "inv5",
    1: "inv2",
    0: "inv1",
    8: "inv9",
    2: "inv3",
    9: "inv10",
    10: "inv10",
}

map_dkZH6HJNQNLLDX6Aj = {
    7: "inv8",
    5: "inv6",
    6: "inv7",
    2: "inv3",
    0: "inv1",
    4: "inv5",
    1: "inv2",
    3: "inv4",
    8: "inv8",
}

map_FwCGymHmbqcziisH5 = {
    5: "prop6",
    8: "prop9",
    16: "prop17",
    10: "prop11",
    2: "prop3",
    4: "prop5",
    3: "prop4",
    1: "prop2",
    0: "prop1",
    7: "prop8",
    13: "prop14",
    6: "prop7",
    9: "prop10",
    15: "prop16",
    11: "prop12",
    12: "prop13",
    14: "prop15",
    17: "prop18",
}

map_gAeD3MTGCCv8YNTaK = {
    4: "noLoops",
    2: "acyclic",
    5: "weaklyConnected",
    3: "complete",
    6: "stonglyConnected",
    1: "oriented",
    0: "undirected",
    7: "transitive",
}

map_JC8Tij8o8GZb99gEJ = {
    0: "Inv1",
    3: "Inv4",
    1: "Inv2",
    2: "Inv3",
}

map_JDKw8yJZF5fiP3jv3 = {
    8: "inv9",
    0: "inv1",
    9: "inv10",
    11: "inv12",
    4: "inv5",
    1: "inv2",
    2: "inv3",
    12: "inv13",
    5: "inv6",
    6: "inv7",
    14: "inv15",
    10: "inv11",
    13: "inv14",
    3: "inv4",
    7: "inv8",
}

map_jyS8Bmceejj9pLbTW = {
    0: "Inv1",
    1: "Inv2",
    2: "Inv3",
    3: "Inv4",
}

map_PQAJE67kz8w5NWJuM = {
    6: "inv7",
    9: "inv10",
    8: "inv9",
    7: "inv8",
    5: "inv6",
    4: "inv5",
    2: "inv3",
    0: "inv1",
    3: "inv4",
    1: "inv2",
}

map_PSqwzYAfW9dFAa9im = {
    4: "inv5",
    6: "inv7",
    8: "inv9",
    14: "inv15",
    9: "inv10",
    13: "inv14",
    11: "inv12",
    5: "inv6",
    1: "inv2",
    3: "inv4",
    7: "inv8",
    0: "inv1",
    2: "inv3",
    12: "inv13",
    10: "inv11",
}

map_QxGnrFQnXPGh2Lh8C = {
    9: "inv10",
    2: "inv3",
    0: "inv1",
    4: "inv5",
    1: "inv2",
    8: "inv9",
    7: "inv8",
    3: "inv4",
    5: "inv6",
    6: "inv7",
}

map_sDLK7uBCbgZon3znd = {
    4: "inv5",
    5: "inv6",
    0: "inv1",
    3: "inv4",
    8: "inv9",
    1: "inv2",
    6: "inv7",
    7: "inv8",
    2: "inv3",
    9: "inv10",
    10: "inv10",
}

map_WGdhwKZnCu7aKhXq9 = {
    0: "Inv1",
    3: "Inv4",
    2: "Inv3",
    1: "Inv2",
}

map_YH3ANm7Y5Qe5dSYem = {
    8: "inv9",
    10: "inv11",
    14: "inv15",
    13: "inv14",
    5: "inv6",
    6: "inv7",
    9: "inv10",
    0: "inv1",
    7: "inv8",
    11: "inv12",
    2: "inv3",
    4: "inv5",
    3: "inv4",
    12: "inv13",
    1: "inv2",
}

map_zoEADeCW2b2suJB2k = {
    3: "inv4",
    6: "inv7",
    4: "inv5",
    0: "inv1",
    2: "inv3",
    1: "inv2",
    7: "inv7",
    5: "inv6",
}

map_zRAn69AocpkmxXZnW = {
    2: "inv3",
    1: "inv2",
    12: "inv13",
    11: "inv12",
    5: "inv6",
    14: "inv15",
    13: "inv14",
    10: "inv11",
    7: "inv8",
    0: "inv1",
    6: "inv7",
    8: "inv9",
    3: "inv4",
    9: "inv10",
    4: "inv5",
}


def filter_predicates(code, preserve_pred_name, predicate_map):
    pred_pattern = re.compile(r"(pred\s+(\w+)\s*\{[\s\S]*?\})", re.MULTILINE)

    filtered_code = []
    last_index = 0

    for match in pred_pattern.finditer(code):
        full_match = match.group(1)  # Full predicate block
        pred_name = match.group(2)  # Predicate name
        start, end = match.span()  # Start and end of the match

        # Append everything before the current match
        filtered_code.append(code[last_index:start])

        # Remove predicates in the map unless they match preserve_pred_name
        if pred_name in predicate_map.values() and pred_name != preserve_pred_name:
            pass  # Skip this predicate
        else:
            filtered_code.append(full_match)

        # Update last index
        last_index = end

    # Append remaining code after the last match
    filtered_code.append(code[last_index:])

    return "".join(filtered_code)


def filter_cmd_n(cmd_n):
    return re.sub(r"(?<=\d)Ok$", "", cmd_n, flags=re.IGNORECASE)


def a4f_individual_tasks_edit_paths_halstead():
    input_files = os.listdir("data/json/a4f/")
    input_files = [f"data/json/a4f/{input_file}" for input_file in input_files]
    data = []
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    data_dict = {item["_id"]: item for item in data}

    chain_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    chain_df["derivation_chain"] = chain_df["derivation_chain"].apply(ast.literal_eval)

    output_data = []
    for _, row in chain_df.iterrows():
        root_node = row["id"]
        chain = row["derivation_chain"]
        chain_len = len(chain)
        halstead_difficulty = []
        halstead_effort = []
        for i in range(chain_len):
            node = chain[i]
            code = data_dict[node]["code"]
            map_name = f'map_{data_dict[node]["original"]}'
            cmd_n = data_dict[node]["cmd_n"] if "cmd_n" in data_dict[node] else None
            if cmd_n:
                if cmd_n in ignored_predicates:
                    continue
                cmd_n = filter_cmd_n(cmd_n)
                code = filter_predicates(code, cmd_n, globals()[map_name])
            else:
                cmd_i = data_dict[node]["cmd_i"] if "cmd_i" in data_dict[node] else None
                if cmd_i is None:
                    continue
                try:
                    cmd_n = globals()[map_name][cmd_i]
                except KeyError:
                    # e.g., dQCxpGCWhXrwZ5Ruj
                    continue
                code = filter_predicates(code, cmd_n, globals()[map_name])
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            cmd = f"java -jar alloy-metrics.jar analyzeHalstead {temp_file_path}"
            hal_res = subprocess.run(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            hal_res = hal_res.stdout.decode()
            hal_res = ast.literal_eval(hal_res)
            hal_res = calculate_halstead(hal_res)
            halstead_difficulty.append(hal_res[0])
            halstead_effort.append(hal_res[1])
            os.remove(temp_file_path)
        print(f"{root_node}")
        output_row = [root_node, chain_len, halstead_difficulty, halstead_effort]
        output_data.append(output_row)

    output_df = pd.DataFrame(
        output_data,
        columns=["root_node", "chain_len", "halstead_difficulty", "halstead_effort"],
    )
    output_df.to_csv("a4f_individual_tasks_edit_paths_halstead.csv", index=False)
