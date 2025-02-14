import json
import os
import ast
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import pointbiserialr
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from time_diffs import (
    fmp_get_time_diffs_from_unsat_to_sat,
    fmp_get_time_diffs_to_fix_parse_error,
)
from edit_paths import calculate_halstead
from utils import convert_fmp_to_datetime, convert_to_datetime, parse_list_column


##############################################
######## Edit Paths / Revision Chains ########
##############################################
def individual_chain_length_distribution_hist_plot(df: pd.DataFrame, output_file: str):
    chain_length = df["chain_length"]
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(chain_length, bins=range(1, 25), kde=False)
    for container in ax.containers:
        labels = [f"{int(v)}" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="edge", fontsize=9, padding=3)
    plt.xlabel("Chain length")
    plt.ylabel("Branches Count")
    plt.title("Distribution of derivation chain length")
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def individual_chain_length_distribution_box_plot(df: pd.DataFrame, output_file: str):
    chain_length = df["chain_length"]
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(chain_length)
    Q1 = chain_length.quantile(0.25)
    median_length = chain_length.median()
    Q3 = chain_length.quantile(0.75)
    mean_length = chain_length.mean()
    stats_label = (
        f"Q1: {Q1:.2f}\n"
        f"Median: {median_length:.2f}\n"
        f"Q3: {Q3:.2f}\n"
        f"Mean: {mean_length:.2f}\n"
    )

    plt.legend([stats_label], loc="upper right")
    plt.xlabel("Chain length")
    plt.title("Distribution of derivation chain length")
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def combined_chain_length_distribution_hist_plot(output_file: str):
    a4f_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    fmp_df = pd.read_csv("results/fmp_edit_paths_chain_overview.csv")
    a4f_df["derivation_chain"] = a4f_df["derivation_chain"].apply(ast.literal_eval)
    a4f_df["chain_length"] = a4f_df["derivation_chain"].apply(len)
    fmp_chain_length = fmp_df["chain_length"]
    a4f_chain_length = a4f_df["chain_length"]

    data = pd.DataFrame(
        {
            "chain_length": pd.concat([a4f_chain_length, fmp_chain_length]),
            "Dataset": ["A4F"] * len(a4f_chain_length)
            + ["FMP"] * len(fmp_chain_length),
        }
    )

    # Plot the histograms
    plt.figure(figsize=(15, 6))
    ax = sns.histplot(
        data=data,
        x="chain_length",
        hue="Dataset",
        multiple="dodge",
        bins=15,
        kde=False,
        alpha=0.8,
    )

    for container in ax.containers:
        labels = [f"{int(v)}" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="edge", fontsize=9, padding=3)

    plt.xlabel("Chain length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Chain Lengths")
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def combined_chain_length_distribution_box_plot():
    """Box plot of the rivision chain length for A4F and FMP datasets.
    A4F chain lengths are for individual tasks, while FMP chain lengths are for the entire dataset.
    """
    a4f_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    fmp_df = pd.read_csv("results/fmp_edit_paths_chain_overview.csv")
    a4f_df["derivation_chain"] = a4f_df["derivation_chain"].apply(ast.literal_eval)
    a4f_df["chain_length"] = a4f_df["derivation_chain"].apply(len)
    fmp_chain_length = fmp_df["chain_length"]
    a4f_chain_length = a4f_df["chain_length"]

    data = pd.DataFrame(
        {
            "chain_length": pd.concat([a4f_chain_length, fmp_chain_length]),
            "Dataset": [r"$\text{A4FpT}$"] * len(a4f_chain_length)
            + [r"$\text{FMP}_{\text{als}}$"] * len(fmp_chain_length),
        }
    )

    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(
        x="Dataset",
        y="chain_length",
        data=data,
        palette="Set2",
        hue="Dataset",
        showfliers=False,
        width=0.7,
    )
    plt.ylabel("Edit Count", fontsize=14)
    plt.title("Edit Chain Length Distribution", fontsize=14)
    plt.legend([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)
    plt.xticks(fontsize=12)
    for dataset in data["Dataset"].unique():
        dataset_values = data[data["Dataset"] == dataset]["chain_length"]
        q1, median, q3 = np.percentile(dataset_values, [25, 50, 75])

        # Annotate quartiles on the box plot
        y_offset = 0.5
        x_pos = {r"$\text{A4FpT}$": 0, r"$\text{FMP}_{\text{als}}$": 1}[
            dataset
        ]  # Position for each dataset
        ax.text(
            x_pos - 0.3,
            q1 - 1.7,
            f"Q1: {q1:.1f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax.text(
            x_pos,
            median + y_offset,
            f"Med: {median:.1f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax.text(
            x_pos + 0.2,
            q3 + y_offset,
            f"Q3: {q3:.1f}",
            color="black",
            ha="center",
            fontsize=12,
        )
    plt.tight_layout()
    plt.savefig(
        "results/plots/combined_chain_distribution_box.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def combined_chain_distribution_hist_with_inset_box_plot(output_file: str):
    a4f_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    fmp_df = pd.read_csv("results/fmp_edit_paths_chain_overview.csv")
    a4f_df["derivation_chain"] = a4f_df["derivation_chain"].apply(ast.literal_eval)
    a4f_df["chain_length"] = a4f_df["derivation_chain"].apply(len)
    a4f_chain_length = a4f_df["chain_length"]
    fmp_chain_length = fmp_df["chain_length"]

    data = pd.DataFrame(
        {
            "chain_length": pd.concat([a4f_chain_length, fmp_chain_length]),
            "Dataset": ["A4F"] * len(a4f_chain_length)
            + ["FMP"] * len(fmp_chain_length),
        }
    )

    # Main histogram
    plt.figure(figsize=(15, 6))
    ax = sns.histplot(
        data=data,
        x="chain_length",
        hue="Dataset",
        palette="Set2",
        multiple="dodge",
        bins=5,
        kde=False,
        alpha=0.8,
    )

    for container in ax.containers:
        labels = [f"{int(v)}" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="edge", fontsize=9, padding=3)

    # Inset box plot
    inset_ax = inset_axes(
        ax,  # Parent axis
        width="30%",  # Width of the inset
        height="75%",  # Height of the inset
        loc="upper right",  # Location of the inset
    )
    sns.boxplot(
        x="Dataset",
        y="chain_length",
        data=data,
        ax=inset_ax,
        palette="Set2",
        hue="Dataset",
        showfliers=False,
    )
    inset_ax.set_title("")
    inset_ax.set_ylabel("Chain length", fontsize=12)
    inset_ax.set_xlabel("Dataset", fontsize=12)
    inset_ax.tick_params(axis="both", labelsize=10)  # Adjust tick label size
    for dataset in data["Dataset"].unique():
        dataset_values = data[data["Dataset"] == dataset]["chain_length"]
        q1, median, q3 = np.percentile(dataset_values, [25, 50, 75])

        # Annotate quartiles on the box plot
        y_offset = 0.5
        x_pos = {"A4F": 0, "FMP": 1}[dataset]  # Position for each dataset
        inset_ax.text(
            x_pos - 0.2,
            q1 + y_offset,
            f"Q1: {q1:.1f}",
            color="black",
            ha="center",
            fontsize=10,
        )
        inset_ax.text(
            x_pos,
            median + y_offset,
            f"Median: {median:.1f}",
            color="black",
            ha="center",
            fontsize=10,
        )
        inset_ax.text(
            x_pos + 0.2,
            q3 + y_offset,
            f"Q3: {q3:.1f}",
            color="black",
            ha="center",
            fontsize=10,
        )
    # Finalize plot
    ax.set_xlabel("Chain length", fontsize=12)
    ax.set_ylabel("Frequency" if len(data) > 1 else "Count", fontsize=12)
    ax.set_title("Distribution of edit paths length", fontsize=14)
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


##################################
######## Halstead Metrics ########
##################################
def halstead_difficulty_line_plot(df: pd.DataFrame, output_file: str):
    def parse_difficulty(difficulty_str):
        try:
            return ast.literal_eval(difficulty_str)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string: {difficulty_str}")
            return []

    df["halstead_difficulty"] = df["halstead_difficulty"].apply(parse_difficulty)

    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        spec_id = row["id"]
        difficulty_values = row["halstead_difficulty"]
        if difficulty_values:
            sns.lineplot(
                x=range(1, len(difficulty_values) + 1),
                y=difficulty_values,
                label=spec_id,
            )

    plt.title("Halstead Difficulty for Each Row")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Halstead Difficulty")
    plt.legend(title="Leaf Spec ID", loc="upper right")
    plt.show()


def halstead_plot_cluster(df: pd.DataFrame, output_file: str, n_clusters: int = 5):
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(ast.literal_eval)
    difficulty_expanded = pd.DataFrame(df["halstead_difficulty"].tolist())

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(difficulty_expanded.fillna(0))

    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        cluster_data = difficulty_expanded[df["cluster"] == cluster]
        cluster_mean = cluster_data.mean(axis=0)
        sns.lineplot(
            x=range(1, len(cluster_mean) + 1),
            y=cluster_mean,
            label=f"Cluster {cluster+1}",
        )

    plt.title("Clustered Halstead Difficulty")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Halstead Difficulty")
    plt.legend()
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def halstead_plot_summary(df: pd.DataFrame, output_file: str):
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    difficulty_expanded = pd.DataFrame(df["halstead_difficulty"].tolist())
    mean_values = difficulty_expanded.mean(axis=0)
    std_values = difficulty_expanded.std(axis=0)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=range(1, len(mean_values) + 1),
        y=mean_values,
        label="Mean Difficulty",
        color="blue",
    )
    plt.fill_between(
        range(1, len(mean_values) + 1),
        # If the mean value is close to 0 and the standard deviation is large,
        # this fill can result in negative, even though our original data does not contain negatives.
        # Clip if needed: np.maximum(mean_values - std_values, 0)
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.3,
        color="blue",
        label="Standard Deviation",
    )

    plt.title("Aggregated Halstead Difficulty")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Halstead Difficulty")
    plt.legend()
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def halstead_cumulative_trend(df: pd.DataFrame, output_file: str):
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    difficulty_flat = [
        value for row in df["halstead_difficulty"] for value in row if value != 0
    ]
    cumulative_sum = pd.Series(difficulty_flat).cumsum()

    plt.figure(figsize=(12, 8))
    sns.lineplot(x=range(len(cumulative_sum)), y=cumulative_sum)
    plt.title("Cumulative Trend of Halstead Difficulty")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Cumulative Sum of Halstead Difficulty")
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def halstead_plot_regression_trend(df: pd.DataFrame, output_file: str):
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    difficulty_flat = [
        value for row in df["halstead_difficulty"] for value in row if value != 0
    ]
    x = np.arange(len(difficulty_flat)).reshape(-1, 1)
    y = np.array(difficulty_flat)

    model = LinearRegression()
    model.fit(x, y)
    trend_line = model.predict(x)

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        x=range(len(difficulty_flat)),
        y=difficulty_flat,
        label="Original Data",
        alpha=0.5,
    )
    sns.lineplot(
        x=range(len(trend_line)),
        y=trend_line,
        label="Trend (Linear Regression)",
        color="red",
    )
    plt.title("Regression Analysis of Halstead Difficulty")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Halstead Difficulty")
    plt.legend()
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def halstead_plot_cluster_with_std_dev(
    df: pd.DataFrame, output_file: str, n_clusters: int = 5
):
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    difficulty_expanded = pd.DataFrame(df["halstead_difficulty"].tolist())

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(difficulty_expanded.fillna(0))

    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        cluster_data = difficulty_expanded[df["cluster"] == cluster]
        cluster_mean = cluster_data.mean(axis=0)
        cluster_std = cluster_data.std(axis=0)
        cluster_size = len(cluster_data)
        # Plot the mean line
        sns.lineplot(
            x=range(1, len(cluster_mean) + 1),
            y=cluster_mean,
            label=f"Cluster {cluster + 1}  (n={cluster_size})",
        )

        # Add the shaded area for standard deviation
        plt.fill_between(
            range(1, len(cluster_mean) + 1),
            cluster_mean - cluster_std,
            cluster_mean + cluster_std,
            alpha=0.2,
            # label=f"Cluster {cluster + 1} Std Dev",
        )

    plt.title("Halstead Difficulty with Standard Deviation", fontsize=14)
    plt.xlabel("Edit Chain Steps", fontsize=12)
    plt.ylabel("Halstead Difficulty", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    df.to_csv(output_file.replace(".pdf", ".csv"), index=False)


def halstead_plot_cluster_percentile(
    df: pd.DataFrame, output_file: str, n_clusters: int = 5
):
    # Convert the halstead_difficulty strings into lists
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    difficulty_expanded = pd.DataFrame(df["halstead_difficulty"].tolist())

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(difficulty_expanded.fillna(0))

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot each cluster
    for cluster in range(n_clusters):
        # Select data for the current cluster
        cluster_data = difficulty_expanded[df["cluster"] == cluster]

        # Calculate mean, 25th, and 75th percentiles
        cluster_mean = cluster_data.mean(axis=0)
        cluster_25th = np.percentile(cluster_data, 25, axis=0)
        cluster_75th = np.percentile(cluster_data, 75, axis=0)

        # Plot the mean line
        sns.lineplot(
            x=range(1, len(cluster_mean) + 1),
            y=cluster_mean,
            label=f"Cluster {cluster + 1}",
        )

        # Add the shaded area for the interquartile range (IQR)
        plt.fill_between(
            range(1, len(cluster_mean) + 1),
            cluster_25th,
            cluster_75th,
            alpha=0.2,
            label=f"Cluster {cluster + 1} IQR (25%-75%)",
        )

    # Add labels, title, and legend
    plt.title("Clustered Halstead Difficulty with Interquartile Range (25%-75%)")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Halstead Difficulty")
    plt.legend()
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def halstead_difficulty_and_effort_on_alloy_examples():
    analysis_file = "results/alloyEx_spec_analysis.csv"
    analysis = pd.read_csv(analysis_file)

    analysis["halstead"] = analysis["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    analysis["halstead_difficulty"] = analysis["halstead"].apply(
        lambda x: calculate_halstead(x)[0]
    )
    analysis["halstead_effort"] = analysis["halstead"].apply(
        lambda x: calculate_halstead(x)[1]
    )
    analysis.to_csv(analysis_file, index=False)

    analysis["spec"] = analysis["spec"].str.replace(r"\\", "/", regex=True)
    analysis["category"] = analysis["spec"].str.extract(
        r"Examples/(book|algorithms|case_studies|temporal)"
    )

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=analysis,
        ax=ax1,
        x="category",
        y="halstead_difficulty",
        palette="Set2",
        linewidth=1,
        width=0.5,
        hue="category",
    )
    ax1.set_title("Halstead Difficulty of Typical Alloy Models", fontsize=14)
    ax1.set_ylabel("Halstead Difficulty", fontsize=14)
    ax1.set_xlabel("Category", fontsize=14)
    ax1.legend(fontsize=14)

    # drop nan
    analysis = analysis.dropna(subset=["category"])
    for i, category in enumerate(analysis["category"].unique()):
        q1, median, q3 = np.percentile(
            analysis[analysis["category"] == category]["halstead_difficulty"],
            [25, 50, 75],
        )
        ax1.text(
            i - 0.2,
            q1 - 7.2,
            f"Q1: {q1:.1f}",
            color="black",
            ha="center",
            fontsize=11,
        )
        ax1.text(
            i,
            median + 1,
            f"Med: {median:.1f}",
            color="black",
            ha="center",
            fontsize=11,
        )
        ax1.text(
            i + 0.23,
            q3 + 2,
            f"Q3: {q3:.1f}",
            color="black",
            ha="center",
            fontsize=11,
        )

    # ax2 = fig.add_subplot(gs[0, 1])
    # sns.boxplot(
    #     data=analysis,
    #     ax=ax2,
    #     x="category",
    #     y="halstead_effort",
    #     palette="Set2",
    #     linewidth=1,
    #     width=0.5,
    #     hue="category",
    #     showfliers=False,
    # )
    # ax2.set_title("Halstead Effort on Alloy Examples Dataset", fontsize=14)
    # ax2.set_ylabel("Halstead Effort", fontsize=12)
    # ax2.set_xlabel("Category", fontsize=12)
    # ax2.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        "results/plots/halstead_difficulty_effort_alloy_examples.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def halstead_difficulty_last_spec_of_the_chain_boxplot():
    """Box plot of the Halstead difficulty and effort for the last specification in the derivation chain.
    For A4F, the chain is for individual tasks, while for FMP, the chain is for the entire dataset.
    """
    a4f_chain_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    a4f_chain_df["parsed_chain"] = a4f_chain_df["derivation_chain"].apply(
        ast.literal_eval
    )
    fmp_chain_df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    fmp_chain_df["parsed_chain"] = fmp_chain_df["derivation_chain"].apply(
        ast.literal_eval
    )
    a4f_halstead_df = pd.read_csv("results/a4f_spec_analysis.csv")
    fmp_halstead_df = pd.read_csv("results/fmp_spec_analysis.csv")

    if "\\" in a4f_halstead_df["spec"].iloc[0]:
        a4f_halstead_df["id"] = a4f_halstead_df["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        a4f_halstead_df["id"] = a4f_halstead_df["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    a4f_halstead_df["halstead"] = a4f_halstead_df["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    a4f_halstead_dict = dict(zip(a4f_halstead_df["id"], a4f_halstead_df["halstead"]))

    if "\\" in fmp_halstead_df["spec"].iloc[0]:
        fmp_halstead_df["id"] = fmp_halstead_df["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        fmp_halstead_df["id"] = fmp_halstead_df["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    fmp_halstead_df["halstead"] = fmp_halstead_df["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    fmp_halstead_dict = dict(zip(fmp_halstead_df["id"], fmp_halstead_df["halstead"]))

    a4f_halstead_difficulty = []
    a4f_halstead_effort = []
    fmp_halstead_difficulty = []
    fmp_halstead_effort = []

    for _, row in a4f_chain_df.iterrows():
        last_spec = row["parsed_chain"][-1]
        halstead = a4f_halstead_dict.get(last_spec)
        if halstead:
            D, E = calculate_halstead(halstead)
            a4f_halstead_difficulty.append(D)
            a4f_halstead_effort.append(E)

    for _, row in fmp_chain_df.iterrows():
        last_spec = row["parsed_chain"][-1]
        halstead = fmp_halstead_dict.get(last_spec)
        if halstead:
            D, E = calculate_halstead(halstead)
            fmp_halstead_difficulty.append(D)
            fmp_halstead_effort.append(E)

    fig = plt.figure(figsize=(6, 5))
    gs = GridSpec(1, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=[a4f_halstead_difficulty, fmp_halstead_difficulty],
        ax=ax1,
        palette="Set2",
        linewidth=1,
        width=0.5,
        showfliers=False,
    )
    ax1.set_title(
        "Halstead Difficulty of the \nlast model in the revision chain", fontsize=14
    )
    ax1.set_ylabel("Difficulty", fontsize=14)
    ax1.set_xticks(
        [0, 1], [r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=14
    )
    for i, dataset in enumerate([a4f_halstead_difficulty, fmp_halstead_difficulty]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        ax1.text(i, q1 + 0.35, f"Q1: {q1:.2f}", color="black", ha="center", fontsize=12)
        ax1.text(
            i,
            median + 0.35,
            f"Median: {median:.2f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax1.text(
            i + 0.2, q3 + 0.5, f"Q3: {q3:.2f}", color="black", ha="center", fontsize=12
        )

    # ax2 = fig.add_subplot(gs[0, 1])
    # sns.boxplot(
    #     data=[a4f_halstead_effort, fmp_halstead_effort],
    #     ax=ax2,
    #     palette="Set2",
    #     linewidth=1,
    #     width=0.5,
    #     showfliers=False,
    # )
    # ax2.set_title("Halstead Effort")
    # ax2.set_ylabel("Effort")
    # ax2.set_xticks([0, 1], ["A4F", "FMP"])
    # y_offset = 400
    # for i, dataset in enumerate([a4f_halstead_effort, fmp_halstead_effort]):
    #     q1, median, q3 = np.percentile(dataset, [25, 50, 75])
    #     ax2.text(
    #         i, q1 + y_offset, f"Q1: {q1:.0f}", color="black", ha="center", fontsize=10
    #     )
    #     ax2.text(
    #         i,
    #         median + y_offset,
    #         f"Median: {median:.0f}",
    #         color="black",
    #         ha="center",
    #         fontsize=10,
    #     )
    #     ax2.text(
    #         i + 0.2,
    #         q3 + y_offset,
    #         f"Q3: {q3:.0f}",
    #         color="black",
    #         ha="center",
    #         fontsize=10,
    #     )
    plt.tight_layout()
    plt.savefig(
        "results/plots/halstead_difficulty_effort_last_spec_chain.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def halstead_difficulty_alloy_examples_and_fmp_a4f():
    analysis_file = "results/alloyEx_spec_analysis.csv"
    analysis = pd.read_csv(analysis_file)

    analysis["halstead"] = analysis["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    analysis["halstead_difficulty"] = analysis["halstead"].apply(
        lambda x: calculate_halstead(x)[0]
    )
    analysis["halstead_effort"] = analysis["halstead"].apply(
        lambda x: calculate_halstead(x)[1]
    )
    analysis.to_csv(analysis_file, index=False)

    analysis["spec"] = analysis["spec"].str.replace(r"\\", "/", regex=True)
    analysis["category"] = analysis["spec"].str.extract(
        r"Examples/(book|algorithms|case_studies|temporal)"
    )

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=analysis,
        ax=ax1,
        x="category",
        y="halstead_difficulty",
        palette="Set2",
        linewidth=1,
        width=0.5,
        hue="category",
    )
    # ax1.set_title("Halstead Difficulty of Typical Alloy Models", fontsize=14)
    ax1.set_ylabel("Halstead Difficulty", fontsize=14)
    ax1.set_xticklabels(["Algorithms", "Book", "Case Studies", "Temporal"], fontsize=14)
    ax1.set_xlabel("(a)", fontsize=14)
    ax1.legend(fontsize=14)

    # drop nan
    analysis = analysis.dropna(subset=["category"])
    for i, category in enumerate(analysis["category"].unique()):
        q1, median, q3 = np.percentile(
            analysis[analysis["category"] == category]["halstead_difficulty"],
            [25, 50, 75],
        )
        ax1.text(
            i - 0.22,
            q1 - 7.2,
            f"Q1: {q1:.1f}",
            color="black",
            ha="center",
            fontsize=11,
        )
        ax1.text(
            i,
            median + 1,
            f"M: {median:.1f}",
            color="black",
            ha="center",
            fontsize=11,
        )
        ax1.text(
            i + 0.25,
            q3 + 2,
            f"Q3: {q3:.1f}",
            color="black",
            ha="center",
            fontsize=11,
        )

    a4f_chain_df = pd.read_csv("results/a4f_individual_tasks_edit_paths_chain_list.csv")
    a4f_chain_df["parsed_chain"] = a4f_chain_df["derivation_chain"].apply(
        ast.literal_eval
    )
    fmp_chain_df = pd.read_csv("results/fmp_edit_paths_chain_list.csv")
    fmp_chain_df["parsed_chain"] = fmp_chain_df["derivation_chain"].apply(
        ast.literal_eval
    )
    a4f_halstead_df = pd.read_csv("results/a4f_spec_analysis.csv")
    fmp_halstead_df = pd.read_csv("results/fmp_spec_analysis.csv")

    if "\\" in a4f_halstead_df["spec"].iloc[0]:
        a4f_halstead_df["id"] = a4f_halstead_df["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        a4f_halstead_df["id"] = a4f_halstead_df["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    a4f_halstead_df["halstead"] = a4f_halstead_df["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    a4f_halstead_dict = dict(zip(a4f_halstead_df["id"], a4f_halstead_df["halstead"]))

    if "\\" in fmp_halstead_df["spec"].iloc[0]:
        fmp_halstead_df["id"] = fmp_halstead_df["spec"].apply(
            lambda x: x.split("\\")[-1].split(".")[0]
        )
    else:
        fmp_halstead_df["id"] = fmp_halstead_df["spec"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    fmp_halstead_df["halstead"] = fmp_halstead_df["halstead"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    fmp_halstead_dict = dict(zip(fmp_halstead_df["id"], fmp_halstead_df["halstead"]))

    a4f_halstead_difficulty = []
    a4f_halstead_effort = []
    fmp_halstead_difficulty = []
    fmp_halstead_effort = []

    for _, row in a4f_chain_df.iterrows():
        last_spec = row["parsed_chain"][-1]
        halstead = a4f_halstead_dict.get(last_spec)
        if halstead:
            D, E = calculate_halstead(halstead)
            a4f_halstead_difficulty.append(D)
            a4f_halstead_effort.append(E)

    for _, row in fmp_chain_df.iterrows():
        last_spec = row["parsed_chain"][-1]
        halstead = fmp_halstead_dict.get(last_spec)
        if halstead:
            D, E = calculate_halstead(halstead)
            fmp_halstead_difficulty.append(D)
            fmp_halstead_effort.append(E)

    ax2 = fig.add_subplot(gs[0, 1])
    set2_palette = sns.color_palette("Set2")
    sns.boxplot(
        data=[a4f_halstead_difficulty, fmp_halstead_difficulty],
        ax=ax2,
        palette=[set2_palette[4], set2_palette[5]],
        linewidth=1,
        width=0.5,
        showfliers=False,
    )
    # ax2.set_title("Halstead Difficulty of the \nlast model in the revision chain", fontsize=14)
    ax2.set_ylabel("Difficulty", fontsize=14)
    ax2.set_xlabel("(b)", fontsize=14)
    ax2.legend([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=14)
    ax2.set_xticks(
        [0, 1], [r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=14
    )
    for i, dataset in enumerate([a4f_halstead_difficulty, fmp_halstead_difficulty]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        ax2.text(i, q1 + 0.35, f"Q1: {q1:.1f}", color="black", ha="center", fontsize=12)
        ax2.text(
            i,
            median + 0.35,
            f"M: {median:.1f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax2.text(
            i + 0.25, q3 + 0.5, f"Q3: {q3:.1f}", color="black", ha="center", fontsize=12
        )
    plt.tight_layout()
    plt.savefig(
        "results/plots/halstead_difficulty_effort_alloy_examples_fmp_a4f.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def halstead_difficulty_delta():
    a4f_df = pd.read_csv(
        "results/a4f_individual_edit_paths_chain_levenshtein_halstead.csv"
    )
    fmp_df = pd.read_csv("results/fmp_edit_paths_chain_levenshtein_halstead.csv")

    a4f_df["halstead_difficulty"] = a4f_df["halstead_difficulty"].apply(
        ast.literal_eval
    )
    fmp_df["halstead_difficulty"] = fmp_df["halstead_difficulty"].apply(
        ast.literal_eval
    )

    a4f_all_diffs = []
    fmp_all_diffs = []
    for difficulties in a4f_df["halstead_difficulty"]:
        diffs = [
            difficulties[i + 1] - difficulties[i] for i in range(len(difficulties) - 1)
        ]
        a4f_all_diffs.extend(diffs)

    for difficulties in fmp_df["halstead_difficulty"]:
        diffs = [
            difficulties[i + 1] - difficulties[i] for i in range(len(difficulties) - 1)
        ]
        fmp_all_diffs.extend(diffs)

    # Drop difficulty differences of 0
    a4f_all_diffs = [diff for diff in a4f_all_diffs if diff != 0]
    fmp_all_diffs = [diff for diff in fmp_all_diffs if diff != 0]

    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data=[a4f_all_diffs, fmp_all_diffs],
        palette="Set2",
        showfliers=False,
        width=0.7,
    )
    for i, dataset in enumerate([a4f_all_diffs, fmp_all_diffs]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        plt.text(
            i - 0.2, q1 - 0.5, f"Q1: {q1:.2f}", color="black", ha="center", fontsize=12
        )
        plt.text(
            i,
            median + 0.2,
            f"Med: {median:.2f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        plt.text(
            i - 0.2, q3 + 0.2, f"Q3: {q3:.2f}", color="black", ha="center", fontsize=12
        )
    plt.title("Halstead Difficulty Delta", fontsize=14)
    plt.ylabel("Difficulty Delta", fontsize=14)
    plt.xticks([0, 1], [r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)
    plt.legend([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)
    plt.tight_layout()
    plt.savefig(
        "results/plots/halstead_difficulty_delta.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


###############################
######## Edit Distance ########
###############################
def combined_boxplot_edit_distance_and_fix_steps():
    a4f_distance_path = (
        "results/a4f_individual_edit_paths_chain_levenshtein_halstead.csv"
    )
    a4f_distance_df = pd.read_csv(a4f_distance_path)
    fmp_distance_path = "results/fmp_edit_paths_chain_levenshtein_halstead.csv"
    fmp_distance_df = pd.read_csv(fmp_distance_path)
    a4f_fix_steps_path = "results/a4f_steps_to_fix.csv"
    a4f_fix_steps_df = pd.read_csv(a4f_fix_steps_path)
    fmp_fix_steps_path = "results/fmp_steps_to_fix.csv"
    fmp_fix_steps_df = pd.read_csv(fmp_fix_steps_path)

    a4f_distance_df["distances"] = a4f_distance_df["distances"].apply(ast.literal_eval)
    fmp_distance_df["distances"] = fmp_distance_df["distances"].apply(ast.literal_eval)

    a4f_all_distances = [
        distance for sublist in a4f_distance_df["distances"] for distance in sublist
    ]
    # Drop edit distance 0
    a4f_all_distances = [distance for distance in a4f_all_distances if distance != 0]
    fmp_all_distances = [
        distance for sublist in fmp_distance_df["distances"] for distance in sublist
    ]
    # Drop edit distance 0
    fmp_all_distances = [distance for distance in fmp_all_distances if distance != 0]

    a4f_fix_steps_df["parseerror_fix_steps"] = a4f_fix_steps_df[
        "parseerror_fix_steps"
    ].apply(ast.literal_eval)
    a4f_fix_steps_df["unsat_to_sat_steps"] = a4f_fix_steps_df[
        "unsat_to_sat_steps"
    ].apply(ast.literal_eval)
    fmp_fix_steps_df["parseerror_fix_steps"] = fmp_fix_steps_df[
        "parseerror_fix_steps"
    ].apply(ast.literal_eval)
    fmp_fix_steps_df["unsat_to_sat_steps"] = fmp_fix_steps_df[
        "unsat_to_sat_steps"
    ].apply(ast.literal_eval)

    a4f_parseerror_fix_steps = [
        step for steps in a4f_fix_steps_df["parseerror_fix_steps"] for step in steps
    ]
    fmp_parseerror_fix_steps = [
        step for steps in fmp_fix_steps_df["parseerror_fix_steps"] for step in steps
    ]

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=[a4f_all_distances, fmp_all_distances],
        ax=ax1,
        palette="Set2",
        showfliers=False,
        width=0.5,
    )

    for i, dataset in enumerate([a4f_all_distances, fmp_all_distances]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        ax1.text(
            i - 0.2, q1 - 16, f"Q1: {q1:.0f}", color="black", ha="center", fontsize=12
        )
        ax1.text(
            i,
            median + 2 + 0.5,
            f"Med: {median:.0f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax1.text(
            i - 0.2, q3 + 1.5, f"Q3: {q3:.0f}", color="black", ha="center", fontsize=12
        )

    ax1.set_title("Edit Distances between interactions", fontsize=14)
    ax1.set_ylabel("Distances", fontsize=13)
    ax1.set_xticklabels([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=[a4f_parseerror_fix_steps, fmp_parseerror_fix_steps],
        ax=ax2,
        palette="Set2",
        showfliers=False,
        width=0.5,
    )

    for i, dataset in enumerate([a4f_parseerror_fix_steps, fmp_parseerror_fix_steps]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        ax2.text(
            i - 0.2, q1 - 0.2, f"Q1: {q1:.0f}", color="black", ha="center", fontsize=12
        )
        ax2.text(
            i,
            median + 0.2,
            f"Med: {median:.0f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax2.text(
            i - 0.2, q3 + 0.1, f"Q3: {q3:.0f}", color="black", ha="center", fontsize=12
        )
    ax2.set_title("Interactions to Fix Errors", fontsize=14)
    ax2.set_ylabel("Interactions", fontsize=13)
    ax2.set_xticklabels([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)

    plt.tight_layout()
    plt.savefig(
        "results/plots/combined_boxplot_edit_distance_fix_steps.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


##################################
######## Time Differences ########
##################################
def time_diff_boxplot_of_parse_error_fix_unsat_sat():
    parse_error_fix_all_seconds = fmp_get_time_diffs_to_fix_parse_error()
    unsat_sat_all_seconds = fmp_get_time_diffs_from_unsat_to_sat()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=[parse_error_fix_all_seconds, unsat_sat_all_seconds],
        showfliers=False,
        width=0.5,
        linewidth=1.5,
        flierprops=dict(marker="o", markersize=5),
        palette="Set2",
    )
    plt.title("Time Differences to Fix Parse Error and from UNSAT to SAT")
    plt.ylabel("Time (seconds)")
    plt.legend(["Parse Error Fix", "UNSAT to SAT"], loc="upper right")
    plt.xticks([0, 1], ["Parse Error Fix", "UNSAT to SAT"])
    plt.savefig(
        "results/plots/time_diff_boxplot.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()


def boxplot_parse_error_vs_not_parse_error():

    a4f_df = pd.read_csv("results/a4f_parseerror_pairs.csv")
    a4f_df["parseerror_distances"] = a4f_df["parseerror_distances"].apply(
        ast.literal_eval
    )
    a4f_df["not_parseerror_distances"] = a4f_df["not_parseerror_distances"].apply(
        ast.literal_eval
    )

    a4f_parseerror_data = [
        item for sublist in a4f_df["parseerror_distances"] for item in sublist
    ]
    a4f_not_parseerror_data = [
        item for sublist in a4f_df["not_parseerror_distances"] for item in sublist
    ]

    fmp_df = pd.read_csv("results/fmp_parseerror_pairs.csv")
    fmp_df["parseerror_distances"] = fmp_df["parseerror_distances"].apply(
        ast.literal_eval
    )
    fmp_df["not_parseerror_distances"] = fmp_df["not_parseerror_distances"].apply(
        ast.literal_eval
    )

    fmp_parseerror_data = [
        item for sublist in fmp_df["parseerror_distances"] for item in sublist
    ]
    fmp_not_parseerror_data = [
        item for sublist in fmp_df["not_parseerror_distances"] for item in sublist
    ]

    fig = plt.figure(figsize=(11, 5))
    gs = GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=[a4f_parseerror_data, a4f_not_parseerror_data],
        ax=ax1,
        palette="Set2",
        linewidth=1,
        width=0.5,
        showfliers=False,
    )
    ax1.set_title("Alloy4Fun")
    ax1.set_ylabel("Edit Distance")
    ax1.set_xticks(
        [0, 1], ["Distance with parse error", "Distance without parse error"]
    )

    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=[fmp_parseerror_data, fmp_not_parseerror_data],
        ax=ax2,
        palette="Set2",
        linewidth=1,
        width=0.5,
        showfliers=False,
    )
    ax2.set_title("FMP")
    ax2.set_ylabel("Edit Distance")
    ax2.set_xticks(
        [0, 1], ["Distance with parse error", "Distance without parse error"]
    )
    plt.tight_layout()
    # plt.savefig(
    #     "results/plots/parsed_not_parsed_edit_distance.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    plt.show()


def rq1_syntax_error_or_not_pie_chart(output_file: str):
    a4f_model_df = pd.read_csv("results/a4f_model_analysis.csv")
    fmp_model_df = pd.read_csv(
        "results/fmp_model_analysis.csv", names=["sepc", "cmd", "result"]
    )
    result_mapping = {
        "SAT": "Correct",
        "UNSAT": "Correct",
        "PARSEERROR": "Syntax Error",
    }
    a4f_model_df["result_category"] = a4f_model_df["result"].map(result_mapping)
    fmp_model_df["result_category"] = fmp_model_df["result"].map(result_mapping)

    a4f_result_counts = a4f_model_df["result_category"].value_counts()
    fmp_result_counts = fmp_model_df["result_category"].value_counts()
    categories = ["Correct", "Syntax Error"]
    a4f_result_counts = a4f_result_counts.reindex(categories, fill_value=0)
    fmp_result_counts = fmp_result_counts.reindex(categories, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    set2_colors = sns.color_palette("Set2", n_colors=2)
    wedges, texts, autotexts = axes[0].pie(
        a4f_result_counts,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=set2_colors,
    )

    for i, autotext in enumerate(autotexts):
        autotext.set_text(a4f_result_counts.index[i] + "\n" + autotext.get_text())
        autotext.set_fontsize(12)
    axes[0].text(
        0,
        -1.2,
        r"A4F",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=18,
    )

    wedges, texts, autotexts = axes[1].pie(
        fmp_result_counts,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=set2_colors,
    )

    for i, autotext in enumerate(autotexts):
        autotext.set_text(fmp_result_counts.index[i] + "\n" + autotext.get_text())
        autotext.set_fontsize(12)
    axes[1].text(
        0,
        -1.2,
        r"$\text{FMP}_{\text{als}}$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=18,
    )

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def a4f_logistic_regression_correlation():
    df = pd.read_csv("results/a4f_correlation_data.csv")
    df["derivation_chain"] = parse_list_column(df["derivation_chain"])
    df["status_chain"] = parse_list_column(df["status_chain"])
    df["halstead_difficulty"] = parse_list_column(df["halstead_difficulty"])

    flattened_data = []
    for _, row in df.iterrows():
        for status, difficulty in zip(row["status_chain"], row["halstead_difficulty"]):
            error_label = (
                1 if status == "PARSEERROR" else 0
            )  # 1 if PARSEERROR, 0 otherwise
            flattened_data.append([difficulty, error_label])

    flat_df = pd.DataFrame(flattened_data, columns=["halstead_difficulty", "error"])

    flat_df["intercept"] = 1

    X = flat_df[["intercept", "halstead_difficulty"]]
    y = flat_df["error"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    corr, p_value = pointbiserialr(flat_df["halstead_difficulty"], flat_df["error"])

    print(result.summary())
    print(f"\nPoint-Biserial Correlation: {corr:.4f}, p-value: {p_value:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(
        flat_df["halstead_difficulty"],
        flat_df["error"],
        alpha=0.1,
        label="Actual Data",
        color="blue",
    )

    plt.plot(
        flat_df["halstead_difficulty"],
        result.predict(X),
        label="Logistic Regression",
        color="red",
    )
    plt.xlabel("Halstead Difficulty")
    plt.ylabel("Probability of Syntax Error (PARSEERROR)")
    plt.title("Logistic Regression: Syntax Errors vs. Halstead Difficulty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "results/plots/a4f_logistic_regression_error_no_error.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def fmp_logistic_regression_correlation():
    df = pd.read_csv("results/fmp_correlation_data.csv")
    df["derivation_chain"] = parse_list_column(df["derivation_chain"])
    df["status_chain"] = parse_list_column(df["status_chain"])
    df["halstead_difficulty"] = parse_list_column(df["halstead_difficulty"])

    flattened_data = []
    for _, row in df.iterrows():
        if len(row["status_chain"]) > 1:
            for status, difficulty in zip(
                row["status_chain"][1:], row["halstead_difficulty"][1:]
            ):  # Skip first element
                error_label = 1 if status == "PARSEERROR" else 0
                flattened_data.append([difficulty, error_label])

    flat_df = pd.DataFrame(flattened_data, columns=["halstead_difficulty", "error"])

    flat_df["intercept"] = 1

    X = flat_df[["intercept", "halstead_difficulty"]]
    y = flat_df["error"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    corr, p_value = pointbiserialr(flat_df["halstead_difficulty"], flat_df["error"])

    print(result.summary())
    print(f"\nPoint-Biserial Correlation: {corr:.4f}, p-value: {p_value:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(
        flat_df["halstead_difficulty"],
        flat_df["error"],
        alpha=0.1,
        label="Actual Data",
        color="blue",
    )

    plt.plot(
        flat_df["halstead_difficulty"],
        result.predict(X),
        label="Logistic Regression",
        color="red",
    )
    plt.xlabel("Halstead Difficulty")
    plt.ylabel("Probability of Syntax Error (PARSEERROR)")
    plt.title("Logistic Regression: Syntax Errors vs. Halstead Difficulty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "results/plots/fmp_logistic_regression_error_no_error.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def a4f_spearman_correlation_halstead_time_to_fix():
    input_files = os.listdir("data/json/a4f/")
    input_files = [f"data/json/a4f/{input_file}" for input_file in input_files]
    data = []
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    data_dict = {item["_id"]: item for item in data}

    df = pd.read_csv("results/a4f_parseerror_extracted.csv")
    df["derivation_chain"] = df["derivation_chain"].apply(ast.literal_eval)
    df["status_chain"] = df["status_chain"].apply(ast.literal_eval)
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(ast.literal_eval)

    halstead_difficulty_diffs = []
    time_diffs = []

    for _, row in df.iterrows():
        derivation_chain = row["derivation_chain"]
        halstead_difficulty = row["halstead_difficulty"]

        if len(derivation_chain) < 2:
            continue

        first_id = derivation_chain[0]
        last_id = derivation_chain[-1]

        first_time = convert_to_datetime(data_dict[first_id]["time"])
        last_time = convert_to_datetime(data_dict[last_id]["time"])

        time_diff = abs((last_time - first_time).total_seconds())
        if time_diff > 600:
            continue

        halstead_difficulty_diff = halstead_difficulty[0]

        halstead_difficulty_diffs.append(halstead_difficulty_diff)
        time_diffs.append(time_diff)

    corr, p_value = spearmanr(halstead_difficulty_diffs, time_diffs)

    print(f"Spearman Correlation: {corr}, p-value: {p_value}")

    df = pd.DataFrame(
        {
            "Halstead Difficulty Difference": halstead_difficulty_diffs,
            "Time Difference (seconds)": time_diffs,
        }
    )
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Halstead Difficulty Difference",
        y="Time Difference (seconds)",
        color="blue",
    )
    sns.regplot(
        data=df,
        x="Halstead Difficulty Difference",
        y="Time Difference (seconds)",
        scatter=False,
        color="red",
    )
    plt.xlabel("Halstead Difficulty Difference")
    plt.ylabel("Time Difference (seconds)")
    plt.title("Correlation Between Halstead Difficulty Difference and Time to Fix")
    plt.tight_layout()
    plt.savefig(
        "results/plots/a4f_spearman_correlation_halstead_time_to_fix.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def fmp_spearman_correlation_halstead_time_to_fix():
    data = []
    with open("data/json/fmp.json", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    data_dict = {item["id"]: item for item in data}

    df = pd.read_csv("results/fmp_parseerror_extracted.csv")
    df["derivation_chain"] = df["derivation_chain"].apply(ast.literal_eval)
    df["status_chain"] = df["status_chain"].apply(ast.literal_eval)
    df["halstead_difficulty"] = df["halstead_difficulty"].apply(ast.literal_eval)

    halstead_difficulty_diffs = []
    time_diffs = []

    for _, row in df.iterrows():
        derivation_chain = row["derivation_chain"]
        halstead_difficulty = row["halstead_difficulty"]

        if len(derivation_chain) < 2:
            continue

        first_id = int(derivation_chain[0])
        last_id = int(derivation_chain[-1])

        first_time = convert_fmp_to_datetime(data_dict[first_id]["time"])
        last_time = convert_fmp_to_datetime(data_dict[last_id]["time"])

        time_diff = abs((last_time - first_time).total_seconds())
        if time_diff > 600:
            continue

        # halstead_difficulty_diff = abs(halstead_difficulty[-1] - halstead_difficulty[0])
        halstead_difficulty_diff = halstead_difficulty[0]

        halstead_difficulty_diffs.append(halstead_difficulty_diff)
        time_diffs.append(time_diff)

    corr, p_value = spearmanr(halstead_difficulty_diffs, time_diffs)

    print(f"Spearman Correlation: {corr}, p-value: {p_value}")

    df = pd.DataFrame(
        {
            "Halstead Difficulty Difference": halstead_difficulty_diffs,
            "Time Difference (seconds)": time_diffs,
        }
    )
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Halstead Difficulty Difference",
        y="Time Difference (seconds)",
        color="blue",
    )

    sns.regplot(
        data=df,
        x="Halstead Difficulty Difference",
        y="Time Difference (seconds)",
        scatter=False,
        color="red",
    )
    plt.xlabel("Halstead Difficulty Difference")
    plt.ylabel("Time Difference (seconds)")
    plt.title("Correlation Between Halstead Difficulty Difference and Time to Fix")
    plt.tight_layout()
    plt.savefig(
        "results/plots/fmp_spearman_correlation_halstead_time_to_fix.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def box_plot_edit_distance():
    a4f_distance_path = (
        "results/a4f_individual_edit_paths_chain_levenshtein_halstead.csv"
    )
    a4f_distance_df = pd.read_csv(a4f_distance_path)
    fmp_distance_path = "results/fmp_edit_paths_chain_levenshtein_halstead.csv"
    fmp_distance_df = pd.read_csv(fmp_distance_path)
    a4f_distance_df["distances"] = a4f_distance_df["distances"].apply(ast.literal_eval)
    fmp_distance_df["distances"] = fmp_distance_df["distances"].apply(ast.literal_eval)

    a4f_all_distances = [
        distance for sublist in a4f_distance_df["distances"] for distance in sublist
    ]
    # Drop edit distance 0
    a4f_all_distances = [distance for distance in a4f_all_distances if distance != 0]
    fmp_all_distances = [
        distance for sublist in fmp_distance_df["distances"] for distance in sublist
    ]
    # Drop edit distance 0
    fmp_all_distances = [distance for distance in fmp_all_distances if distance != 0]

    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data=[a4f_all_distances, fmp_all_distances],
        palette="Set2",
        showfliers=False,
        width=0.5,
    )
    for i, dataset in enumerate([a4f_all_distances, fmp_all_distances]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        plt.text(
            i - 0.2, q1 - 16, f"Q1: {q1:.0f}", color="black", ha="center", fontsize=12
        )
        plt.text(
            i,
            median + 2 + 0.5,
            f"Med: {median:.0f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        plt.text(
            i - 0.2, q3 + 1.5, f"Q3: {q3:.0f}", color="black", ha="center", fontsize=12
        )

    plt.title("Edit Distances between interactions", fontsize=14)
    plt.xlabel("Dataset", fontsize=14)
    plt.ylabel("Distances", fontsize=14)
    plt.xticks([0, 1], [r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=14)
    plt.legend([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=14)
    plt.tight_layout()
    plt.savefig(
        "results/plots/edit_distance_boxplot.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def box_plot_fix_error_steps():
    a4f_fix_steps_path = "results/a4f_steps_to_fix.csv"
    a4f_fix_steps_df = pd.read_csv(a4f_fix_steps_path)
    fmp_fix_steps_path = "results/fmp_steps_to_fix.csv"
    fmp_fix_steps_df = pd.read_csv(fmp_fix_steps_path)

    a4f_fix_steps_df["parseerror_fix_steps"] = a4f_fix_steps_df[
        "parseerror_fix_steps"
    ].apply(ast.literal_eval)
    a4f_fix_steps_df["unsat_to_sat_steps"] = a4f_fix_steps_df[
        "unsat_to_sat_steps"
    ].apply(ast.literal_eval)
    fmp_fix_steps_df["parseerror_fix_steps"] = fmp_fix_steps_df[
        "parseerror_fix_steps"
    ].apply(ast.literal_eval)
    fmp_fix_steps_df["unsat_to_sat_steps"] = fmp_fix_steps_df[
        "unsat_to_sat_steps"
    ].apply(ast.literal_eval)

    a4f_parseerror_fix_steps = [
        step for steps in a4f_fix_steps_df["parseerror_fix_steps"] for step in steps
    ]
    fmp_parseerror_fix_steps = [
        step for steps in fmp_fix_steps_df["parseerror_fix_steps"] for step in steps
    ]

    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data=[a4f_parseerror_fix_steps, fmp_parseerror_fix_steps],
        palette="Set2",
        showfliers=False,
        width=0.5,
    )
    for i, dataset in enumerate([a4f_parseerror_fix_steps, fmp_parseerror_fix_steps]):
        q1, median, q3 = np.percentile(dataset, [25, 50, 75])
        plt.text(
            i - 0.2, q1 - 0.2, f"Q1: {q1:.0f}", color="black", ha="center", fontsize=12
        )
        plt.text(
            i,
            median + 0.2,
            f"Med: {median:.0f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        plt.text(
            i - 0.2, q3 + 0.1, f"Q3: {q3:.0f}", color="black", ha="center", fontsize=12
        )
    plt.title("Interactions to Fix Errors", fontsize=14)
    plt.ylabel("Interactions", fontsize=13)
    plt.xlabel("Dataset", fontsize=13)
    plt.xticks([0, 1], [r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)
    plt.legend([r"$\text{A4FpT}$", r"$\text{FMP}_{\text{als}}$"], fontsize=13)

    plt.tight_layout()
    plt.savefig(
        "results/plots/fix_error_steps_boxplot.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


######## Main ########
def main():
    combined_boxplot_edit_distance_and_fix_steps()
    combined_chain_length_distribution_box_plot()
    box_plot_edit_distance()
    box_plot_fix_error_steps()
    halstead_difficulty_delta()
    halstead_difficulty_alloy_examples_and_fmp_a4f()
    rq1_syntax_error_or_not_pie_chart(
        "results/plots/rq1_syntax_error_or_not_pie_chart.pdf"
    )

    a4f_chain_distance_halstead_df = pd.read_csv(
        "results/a4f_individual_tasks_edit_paths_halstead.csv"
    )

    halstead_plot_cluster_with_std_dev(
        a4f_chain_distance_halstead_df,
        "results/plots/a4f_halstead_clustered_std_dev.pdf",
        8,
    )

    fmp_chain_distance_halstead_df = pd.read_csv(
        "results/fmp_edit_paths_chain_levenshtein_halstead.csv"
    )
    halstead_plot_cluster_with_std_dev(
        fmp_chain_distance_halstead_df,
        "results/plots/fmp_halstead_clustered_std_dev.pdf",
        3,
    )
