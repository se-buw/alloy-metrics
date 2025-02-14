import hashlib
import ast
import pandas as pd
from collections import defaultdict


def compute_file_hash(file_path):
    def normalize(content):
        return "".join(
            content.encode("utf-8").decode("utf-8-sig").replace("\r\n", "\n").split()
        )

    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = normalize(f.read())
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def a4f_syntactic_equivalences():
    df = pd.read_csv("results/a4f_spec_analysis.csv")
    spec_path = df["spec"]
    file_hashes = {}
    for spec in spec_path:
        # print(f"Calculating hash for file: {spec}")
        file_hashes[spec] = compute_file_hash(spec)
    syntactic_equivalences = []
    for i, spec1 in enumerate(spec_path):
        for j in range(i + 1, len(spec_path)):
            spec2 = spec_path[j]
            if file_hashes[spec1] == file_hashes[spec2]:
                print(f"Files {spec1} and {spec2} are syntactically equivalent")
                syntactic_equivalences.append((spec1, spec2))
    pd.DataFrame(syntactic_equivalences, columns=["file1", "file2"]).to_csv(
        "results/a4f_syntactic_equivalences.csv", index=False
    )


def fmp_syntactic_equivalences():
    df = pd.read_csv("results/fmp_spec_analysis.csv")
    spec_path = df["spec"]
    file_hashes = {}
    for spec in spec_path:
        # print(f"Calculating hash for file: {spec}")
        file_hashes[spec] = compute_file_hash(spec)
    syntactic_equivalences = []
    for i, spec1 in enumerate(spec_path):
        for j in range(i + 1, len(spec_path)):
            spec2 = spec_path[j]
            if file_hashes[spec1] == file_hashes[spec2]:
                print(f"Files {spec1} and {spec2} are syntactically equivalent")
                syntactic_equivalences.append((spec1, spec2))
    pd.DataFrame(syntactic_equivalences, columns=["file1", "file2"]).to_csv(
        "results/fmp_syntactic_equivalences.csv", index=False
    )


def save_systactic_uniqueness():
    a4f_df = pd.read_csv("results/a4f_syntactic_equivalences.csv")
    fmp_df = pd.read_csv("results/fmp_syntactic_equivalences.csv")
    a4f_unique = pd.concat([a4f_df["file1"], a4f_df["file2"]]).unique()
    fmp_unique = pd.concat([fmp_df["file1"], fmp_df["file2"]]).unique()

    a4f_overview = {
        "total duplicate": len(a4f_unique),
        "% duplicates": len(a4f_unique) / len(a4f_df),
        "% unique": (1 - len(a4f_unique) / len(a4f_df)),
    }
    fmp_overview = {
        "total duplicate": len(fmp_unique),
        "% duplicates": len(fmp_unique) / len(fmp_df),
        "% unique": (1 - len(fmp_unique) / len(fmp_df)),
    }
    overview = pd.DataFrame([a4f_overview, fmp_overview], index=["a4f", "fmp"])
    overview.to_csv("results/tables/syntactic_uniqueness_overview.csv")


def systactic_uniqueness():
    """Compute the syntactic uniqueness of the models for Alloy4Fun and FMP."""
    a4f_model_df = pd.read_csv("results/a4f_model_analysis.csv")
    fmp_model_df = pd.read_csv(
        "results/fmp_model_analysis.csv", names=["spec_path", "cmd", "result"]
    )
    a4f_model_df["spec_path"] = a4f_model_df["spec_path"].str.replace(
        r"\\", "/", regex=True
    )
    fmp_model_df["spec_path"] = fmp_model_df["spec_path"].str.replace(
        r"\\", "/", regex=True
    )

    a4f_syntax_eq = pd.read_csv("results/a4f_syntactic_equivalences.csv")
    a4f_syntax_eq = pd.unique(a4f_syntax_eq.values.flatten())
    a4f_syntax_eq = [x.replace("\\", "/") for x in a4f_syntax_eq]
    fmp_syntax_eq = pd.read_csv("results/fmp_syntactic_equivalences.csv")
    fmp_syntax_eq = pd.unique(fmp_syntax_eq.values.flatten())
    fmp_syntax_eq = [x.replace("\\", "/") for x in fmp_syntax_eq]

    a4f_df_filtered = a4f_model_df[~a4f_model_df["spec_path"].isin(a4f_syntax_eq)]
    fmp_df_filtered = fmp_model_df[~fmp_model_df["spec_path"].isin(fmp_syntax_eq)]

    result_mapping = {
        "SAT": "Correct",
        "UNSAT": "Correct",
        "UNKNOWN": "Correct",
        "TIMEOUT": "Correct",
        "PARSEERROR": "Syntax Error",
    }

    # Ensure you're working with a copy of the filtered DataFrame
    a4f_df_filtered = a4f_df_filtered.copy()
    fmp_df_filtered = fmp_df_filtered.copy()

    # Now use .loc to safely assign the new column
    a4f_df_filtered.loc[:, "result_category"] = a4f_df_filtered["result"].map(
        result_mapping
    )
    fmp_df_filtered.loc[:, "result_category"] = fmp_df_filtered["result"].map(
        result_mapping
    )

    a4f_result_counts = a4f_df_filtered["result_category"].value_counts()
    fmp_result_counts = fmp_df_filtered["result_category"].value_counts()
    categories = ["Correct", "Syntax Error"]
    a4f_result_counts = a4f_result_counts.reindex(categories, fill_value=0)
    fmp_result_counts = fmp_result_counts.reindex(categories, fill_value=0)

    # Check for any uncategorized results
    uncategorized_a4f = a4f_df_filtered[
        ~a4f_df_filtered["result_category"].isin(result_mapping.values())
    ]
    uncategorized_fmp = fmp_df_filtered[
        ~fmp_df_filtered["result_category"].isin(result_mapping.values())
    ]

    #########################
    print("==============A4F==============")
    print(f"Total Models: {len(a4f_model_df)}")
    print("Unique Models: ", len(a4f_df_filtered))
    print("Unique Models (%): ", len(a4f_df_filtered) / len(a4f_model_df) * 100)

    print(
        "Unique Correct Models: ",
        len(a4f_df_filtered[a4f_df_filtered["result_category"] == "Correct"]),
    )
    print(
        "Unique Correct Models (%): ",
        len(a4f_df_filtered[a4f_df_filtered["result_category"] == "Correct"])
        / len(a4f_df_filtered)
        * 100,
    )
    print(
        "Unique Syntax Errors: ",
        len(a4f_df_filtered[a4f_df_filtered["result_category"] == "Syntax Error"]),
    )
    print(
        "Unique Syntax Errors (%): ",
        len(a4f_df_filtered[a4f_df_filtered["result_category"] == "Syntax Error"])
        / len(a4f_df_filtered)
        * 100,
    )
    print(
        "Uncategorized Models: ",
        len(uncategorized_a4f),
    )
    print(
        "Uncategorized Models (%): ",
        len(uncategorized_a4f) / len(a4f_df_filtered) * 100,
    )
    print("Uncategorized Rows:")
    print(uncategorized_a4f)

    print("==============FMP==============")
    print(f"Total Models: {len(fmp_model_df)}")
    print("Unique Models: ", len(fmp_df_filtered))
    print("Unique Models (%): ", len(fmp_df_filtered) / len(fmp_model_df) * 100)
    print(
        "Unique Correct Models: ",
        len(fmp_df_filtered[fmp_df_filtered["result_category"] == "Correct"]),
    )
    print(
        "Unique Correct Models (%): ",
        len(fmp_df_filtered[fmp_df_filtered["result_category"] == "Correct"])
        / len(fmp_df_filtered)
        * 100,
    )

    print(
        "Unique Syntax Errors: ",
        len(fmp_df_filtered[fmp_df_filtered["result_category"] == "Syntax Error"]),
    )
    print(
        "Unique Syntax Errors (%): ",
        len(fmp_df_filtered[fmp_df_filtered["result_category"] == "Syntax Error"])
        / len(fmp_df_filtered)
        * 100,
    )
    print(
        "Uncategorized Models: ",
        len(uncategorized_fmp),
    )
    print(
        "Uncategorized Models (%): ",
        len(uncategorized_fmp) / len(fmp_df_filtered) * 100,
    )
    print("Uncategorized Rows:")
    print(uncategorized_fmp)


def compute_file_hash(file_path):
    def normalize(content):
        return "".join(
            content.encode("utf-8").decode("utf-8-sig").replace("\r\n", "\n").split()
        )

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            content = normalize(f.read())
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    except FileNotFoundError:
        return None


def identical_files(edit_path_file: str):
    df = pd.read_csv(edit_path_file)

    df["derivation_chain"] = df["derivation_chain"].apply(ast.literal_eval)

    file_hashes = {}
    all_files = set()

    # Compute hashes for all files
    for chain in df["derivation_chain"]:
        for file in chain:
            if file not in file_hashes:
                file_path = f"data/code/fmp/{str(file)}.als"
                file_hashes[file] = compute_file_hash(file_path)
            all_files.add(file)

    # Mapping of hashes to file names
    hash_to_files = defaultdict(set)
    for file, h in file_hashes.items():
        if h:
            hash_to_files[h].add(file)

    # Identify identical files
    identical_files = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    # Identify consecutive identical files
    consecutive_identical = []
    non_consecutive_identical = []

    for chain in df["derivation_chain"]:
        previous_hash = None
        seen_hashes = set()
        for file in chain:
            file_hash = file_hashes.get(file)
            if not file_hash:
                continue

            # Check for consecutive identical files
            if file_hash == previous_hash:
                consecutive_identical.append(file)

            # Check for non-consecutive identical files
            if file_hash in seen_hashes:
                non_consecutive_identical.append(file)

            seen_hashes.add(file_hash)
            previous_hash = file_hash

    print(f"Total files: {len(all_files)}")
    print(f"Identical files: {len(identical_files)}")
    print(f"Consecutive identical files: {len(consecutive_identical)}")
    print(f"Identical but non-consecutive files: {len(non_consecutive_identical)}")

    # Deduplicate the lists
    consecutive_identical = set(consecutive_identical)
    non_consecutive_identical = set(non_consecutive_identical) - consecutive_identical
    print(f"Consecutive identical files: {len(consecutive_identical)}")
    print(f"Identical but non-consecutive files: {len(non_consecutive_identical)}")


def main():
    a4f_syntactic_equivalences()
    fmp_syntactic_equivalences()
