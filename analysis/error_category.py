import subprocess
import pandas as pd


def a4f_error_category():
    a4f_model_df = pd.read_csv("results/a4f_model_analysis.csv")
    a4f_model_df = a4f_model_df[a4f_model_df["result"] == "PARSEERROR"]
    spec_paths = a4f_model_df["spec_path"].tolist()
    output_file = "a4f_error_category.csv"
    for spec_path in spec_paths:
        cmd = [
            "java",
            "-jar",
            "alloy-metrics.jar",
            "analyzeErrors",
            spec_path,
            output_file,
        ]
        subprocess.run(cmd)


def fmp_error_category():
    fmp_model_df = pd.read_csv(
        "results/fmp_model_analysis.csv", names=["spec_path", "cmd", "result"]
    )
    fmp_model_df = fmp_model_df[fmp_model_df["result"] == "PARSEERROR"]
    spec_paths = fmp_model_df["spec_path"].tolist()
    output_file = "fmp_error_category.csv"
    for spec_path in spec_paths:
        cmd = [
            "java",
            "-jar",
            "alloy-metrics.jar",
            "analyzeErrors",
            spec_path,
            output_file,
        ]
        subprocess.run(cmd)


a4f_error_df = pd.read_csv("a4f_error_category.csv")
print(a4f_error_df["errorCategory"].value_counts())
print(a4f_error_df["errorCategory"].value_counts(normalize=True))
print(a4f_error_df["errorLocation"].value_counts())
print(a4f_error_df["errorLocation"].value_counts(normalize=True))

fmp_error_df = pd.read_csv("fmp_error_category.csv")
print(fmp_error_df["errorCategory"].value_counts())
print(fmp_error_df["errorCategory"].value_counts(normalize=True))
print(fmp_error_df["errorLocation"].value_counts())
print(fmp_error_df["errorLocation"].value_counts(normalize=True))
