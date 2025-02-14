import os
import json


def a4f_prepare_json(output_file: str) -> None:
    """Prepare the Alloy4Fun dataset for the analysis.
    Concatenate all the json files into a single file."""
    input_files = os.listdir("data/alloy4fun/json/")
    input_files = [f"data/alloy4fun/json/{input_file}" for input_file in input_files]
    data = []
    count_cmd_i = 0
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                if "cmd_i" not in line:
                    count_cmd_i += 1
                    continue
                data.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print(f"Number of entries without cmd_i: {count_cmd_i}")


def a4f_parse_dataset(input_file: str) -> None:
    """Parse the Alloy4Fun dataset and save the model files to disk."""
    os.makedirs(f"data/code/a4f", exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            file_name = data["_id"]
            code = data["code"]
            with open(
                f"data/code/a4f/{file_name}.als", "w", encoding="utf-8"
            ) as code_file:
                code_file.write(code)


def fmp_prepare_json(input_file: str, output_file: str) -> None:
    """Prepare the FMP dataset for the analysis."""
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            id = data["id"]
            if id <= 17409:  # Migration point
                continue
            parent = data["parent"]
            code = data["code"]
            permalink = data["permalink"]
            time = data["time"]
            cmd = json.loads(data["metadata"]).get("cmd", None)
            fmp_data = {
                "id": id,
                "parent": parent,
                "code": code,
                "permalink": permalink,
                "time": time,
                "cmd": cmd,
            }
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(fmp_data) + "\n")


def fmp_prepare_json_before_migration(input_file: str, output_file: str) -> None:
    """Prepare the FMP dataset for the analysis before the migration point."""
    with open(input_file, "r", encoding="utf-8") as f:
        loaded_data = [json.loads(line) for line in f]
    for data in loaded_data:
        id = data["id"]
        parent = data["parent"]
        code = data["code"]
        permalink = data["permalink"]
        time = data["time"]
        metadata_str = data.get("metadata")
        if metadata_str is not None:
            fixed_metadata_str = (
                metadata_str.replace("{", '{"')
                .replace(":", '":')
                .replace(", ", ", ")
                .replace("}", "}")
            )
            metadata = json.loads(fixed_metadata_str)
            cmd = metadata.get("cmd_i", metadata.get("cmd", None))
        else:
            cmd = None

        fmp_data = {
            "id": id,
            "parent": parent,
            "code": code,
            "permalink": permalink,
            "time": time,
            "cmd": cmd,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(fmp_data) + "\n")


def fmp_parse_dataset(input_file: str) -> None:
    """Parse the FMP dataset and save the model files to disk."""
    os.makedirs(f"data/code/fmp", exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            file_name = data["id"]
            code = data["code"]
            with open(
                f"data/code/fmp/{file_name}.als", "w", encoding="utf-8"
            ) as code_file:
                code_file.write(code)


def main():
    # ----- Alloy4Fun ---------------
    output_file = "data/json/a4f.json"
    a4f_prepare_json(output_file)
    a4f_parse_dataset(output_file)

    # --------- FMP -----------------
    input_file = "data/json/fmp-1.json"
    output_file = "data/json/fmp.json"
    fmp_prepare_json_before_migration(input_file, output_file)
    input_file = "data/json/fmp-2.json"
    fmp_prepare_json(input_file, output_file)
    fmp_parse_dataset(output_file)
