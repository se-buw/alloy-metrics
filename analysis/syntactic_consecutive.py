import pandas as pd
import ast
import hashlib
from collections import defaultdict


# Function to compute hash of a file
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
