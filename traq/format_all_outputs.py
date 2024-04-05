"""Format all outputs/visualizations and report final numbers for the final paper."""

import glob
import os

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_jitter,
    geom_line,
    geom_point,
    geom_smooth,
    ggplot,
    xlab,
    ylab,
    ylim,
)

valid_trial_names = [
    "poise",
    "hipattack",
    "compass",
    "manage",
    "rely",
    "tips3",
    "hope3",
]
algo = "ECOD"


def main(args):  # noqa: C901
    # sensitivity analysis visualizations
    results_dfs, performances_dfs = [], []
    for trial_name in os.listdir(args.input_granularity_dir):
        results_df_filename = os.path.join(
            args.input_granularity_dir, trial_name, "results.csv"
        )
        performances_df_filename = os.path.join(
            args.input_granularity_dir, trial_name, "performances.csv"
        )

        results_df = pd.read_csv(results_df_filename, index_col=0)
        results_df["trial"] = trial_name
        performances_df = pd.read_csv(performances_df_filename, index_col=0)
        performances_df["trial"] = trial_name

        results_dfs.append(results_df)
        performances_dfs.append(performances_df)
    results_df = pd.concat(results_dfs, axis=0)
    performances_df = pd.concat(performances_dfs, axis=0)

    results_df = results_df.rename({"num_samples": "# entries"}, axis=1)
    plot_df = results_df.melt(id_vars=["trial", "snapshot", "name", "# entries"])
    plot_df["threshold"] = pd.Categorical(
        plot_df["variable"].apply(lambda s: float(s.split("_")[-1]))
    )
    plot_df = plot_df.rename({"value": "ascertained irregularity rate"}, axis=1)
    median_plot_df = (
        plot_df.groupby("threshold")["ascertained irregularity rate"]
        .median()
        .to_frame()
        .reset_index()
    )
    plot_filename = os.path.join(args.output_directory, "anomaly_prop.jpg")
    anomaly_prop_plt = (
        ggplot(plot_df)
        + geom_jitter(
            aes(
                x="threshold",
                y="ascertained irregularity rate",
                size="# entries",
                color="trial",
            ),
            width=0.2,
            height=0.0,
        )
        + geom_line(
            median_plot_df,
            aes(x="threshold", y="ascertained irregularity rate", group=1),
            color="black",
            linetype="dashed",
        )
        + geom_point(
            median_plot_df,
            aes(x="threshold", y="ascertained irregularity rate"),
            color="black",
        )
        + xlab("threshold")
        + ylim([0.0, 1.0])
    )
    anomaly_prop_plt.save(plot_filename, height=8, width=8, dpi=300)

    plot_df = performances_df[performances_df["algorithm"].str.contains("ECOD")]
    plot_filename = os.path.join(args.output_directory, "performances.jpg")
    plot_df["record_threshold"] = pd.Categorical(plot_df["record_threshold"])
    plot_df["# entries"] = plot_df["num_samples"]
    median_plot_df = (
        plot_df.groupby("record_threshold")["auroc"].median().to_frame().reset_index()
    )
    performances_plt = (
        ggplot(
            plot_df,
        )
        + geom_jitter(
            aes(x="record_threshold", y="auroc", size="# entries", color="trial"),
            width=0.2,
            height=0.0,
        )
        + geom_line(
            median_plot_df,
            aes(x="record_threshold", y="auroc", group=1),
            color="black",
            linetype="dashed",
        )
        + geom_point(
            median_plot_df, aes(x="record_threshold", y="auroc"), color="black"
        )
        + xlab("threshold")
        + ylab("AUROC")
        + ylim([0.0, 1.0])
    )
    performances_plt.save(plot_filename, height=8, width=8, dpi=300)

    dfs = []
    for comparison_fn in glob.glob(os.path.join(args.input_comparisons_dir, "*.csv")):
        trial_name = os.path.basename(comparison_fn).split(".")[0]
        if trial_name in valid_trial_names:
            df = pd.read_csv(comparison_fn)
            df["trial"] = trial_name
            dfs.append(df)
        else:
            print("skipping ", comparison_fn)

    df_raw = pd.concat(dfs, axis=0)
    df_algo = df_raw[df_raw["algorithm"] == algo]

    print("Total number of tables pumped through the pipeline: ", len(df_algo))
    df = df_algo.drop("error", axis=1).dropna()
    print("Number of tables afte removing tables with no anomalies: ", len(df))
    print("Total number of entries: ", df["num_samples"].sum())
    print("Total number of irregularities: ", df["num_anomalies"].sum())
    print("Median irregularity rate: ", df["anomaly_proportion"].median())

    fmt_float = "{:.2f}".format

    # Compute overview statistics
    num_forms = df.groupby("trial")["plate"].nunique()
    num_entries_desc = df.groupby("trial")["num_samples"].describe()
    num_entries = (
        num_entries_desc["50%"].astype(str)
        + " ("
        + num_entries_desc["25%"].astype(str)
        + ", "
        + num_entries_desc["75%"].astype(str)
        + ")"
    )
    num_columns_desc = df.groupby("trial")["num_columns"].describe()
    num_columns = (
        num_columns_desc["50%"].astype(str)
        + " ("
        + num_columns_desc["25%"].astype(str)
        + ", "
        + num_columns_desc["75%"].astype(str)
        + ")"
    )
    num_anomalies = df.groupby("trial")["num_anomalies"].sum()
    prop_anomalies = num_anomalies / df.groupby("trial")["num_samples"].sum() * 100
    anomalies = (
        num_anomalies.map(fmt_float) + " (" + prop_anomalies.map(fmt_float) + "%)"
    )
    df_summary = pd.DataFrame([num_forms, num_entries, num_columns, anomalies]).T
    df_summary.columns = [
        "num_forms",
        "num_entries",
        "num_fields",
        "num_irregularities",
    ]
    df_summary.to_csv("df_summary.csv")

    # Compute median AUROC by trial and snapshot with IQR
    df_auroc = (
        df.groupby(["trial", "snapshot"])["auroc"]
        .describe()[["25%", "50%", "75%"]]
        .reset_index(1)
    )
    df_auroc["snapshot_idx"] = np.arange(len(df_auroc)).astype(int)
    df_auroc["snapshot_idx"] = (
        df_auroc["snapshot_idx"] - df_auroc["snapshot_idx"].groupby("trial").min()
    )
    df = (
        df.set_index(["trial", "snapshot"])
        .join(df_auroc.reset_index().set_index(["trial", "snapshot"])[["snapshot_idx"]])
        .reset_index()
    )
    df_auroc = df_auroc.drop("snapshot", axis=1)
    df_trial_overall = df.groupby("trial")["auroc"].describe()[["25%", "50%", "75%"]]
    df_trial_overall["snapshot_idx"] = "overall"

    df_snapshot_overall = (
        df.groupby("snapshot_idx")["auroc"]
        .describe()[["25%", "50%", "75%"]]
        .reset_index()
    )
    df_snapshot_overall["trial"] = "overall"
    df_snapshot_overall = df_snapshot_overall.set_index("trial")

    df_overall_overall = df["auroc"].describe()[["25%", "50%", "75%"]]
    df_overall_overall["trial"] = "overall"
    df_overall_overall["snapshot_idx"] = "overall"
    df_overall_overall = df_overall_overall.to_frame().T.set_index("trial")

    df_auroc = pd.concat(
        (df_auroc, df_trial_overall, df_snapshot_overall, df_overall_overall), axis=0
    )
    df_auroc["auroc_fmt"] = (
        df_auroc["50%"].map(fmt_float)
        + " ("
        + df_auroc["25%"].map(fmt_float)
        + ", "
        + df_auroc["75%"].map(fmt_float)
        + ")"
    )
    df_auroc_fmt = df_auroc.pivot(columns="snapshot_idx", values="auroc_fmt")

    # Compute overall precisions
    def bootstrap_ci(f, df, nboot=1000, alpha=0.05):
        stats = []
        for _ in range(nboot):
            idx = np.random.choice(len(df), size=len(df), replace=True)
            df_shuffle = df.iloc[idx, :]
            stats.append(f(df_shuffle))

        _df = pd.DataFrame(stats).quantile([alpha / 2, 1 - alpha / 2]).T
        _df.columns = ["ci.lower", "ci.upper"]

        return _df

    # Baselines
    def anomaly_rate(df):
        return (
            df.groupby("trial")["num_anomalies"].sum()
            / df.groupby("trial")["num_samples"].sum()
        )

    df_anomaly_rate_ci = bootstrap_ci(anomaly_rate, df)
    df_anomaly_rate = (
        anomaly_rate(df).map(fmt_float)
        + " ("
        + df_anomaly_rate_ci["ci.lower"].map(fmt_float)
        + ", "
        + df_anomaly_rate_ci["ci.upper"].map(fmt_float)
        + ")"
    )
    df_anomaly_rate = df_anomaly_rate.to_frame()
    df_anomaly_rate.columns = pd.MultiIndex.from_tuples(
        [("complete SDV baseline", "precision")]
    )

    df["true_positives_at_k"] = df["precision_at_k"] * df["num_anomalies"]

    def tpk(df):
        return (
            df.groupby("trial")["true_positives_at_k"].sum()
            / df.groupby("trial")["num_anomalies"].sum()
        )

    df_precision_at_k_ci = bootstrap_ci(tpk, df)
    df_precision_at_k = (
        tpk(df).map(fmt_float)
        + " ("
        + df_precision_at_k_ci["ci.lower"].map(fmt_float)
        + ", "
        + df_precision_at_k_ci["ci.upper"].map(fmt_float)
        + ")"
    )
    df_precision_at_k = df_precision_at_k.to_frame()
    df_precision_at_k.columns = pd.MultiIndex.from_tuples([("true k", "precision")])

    df["true_positives_at_10"] = df["precision_at_10"] * 10

    def pat10(df):
        return df.groupby("trial")["true_positives_at_10"].sum() / (
            10 * df.groupby("trial")["true_positives_at_10"].count()
        )

    def rat10(df):
        return (
            df.groupby("trial")["true_positives_at_10"].sum()
            / df.groupby("trial")["num_anomalies"].sum()
        )

    df_precision_at_10_ci = bootstrap_ci(pat10, df)
    df_precision_at_10 = (
        pat10(df).map(fmt_float)
        + " ("
        + df_precision_at_10_ci["ci.lower"].map(fmt_float)
        + ", "
        + df_precision_at_10_ci["ci.upper"].map(fmt_float)
        + ")"
    )
    df_recall_at_10_ci = bootstrap_ci(rat10, df)
    df_recall_at_10 = (
        rat10(df).map(fmt_float)
        + " ("
        + df_recall_at_10_ci["ci.lower"].map(fmt_float)
        + ", "
        + df_recall_at_10_ci["ci.upper"].map(fmt_float)
        + ")"
    )
    df_pr_at_10 = pd.concat((df_precision_at_10, df_recall_at_10), axis=1)
    df_pr_at_10.columns = pd.MultiIndex.from_tuples(
        [("at 10", "precision"), ("at 10", "recall")]
    )

    df["true_positives_at_5pct"] = df["precision_at_5pct"] * 0.05 * df["num_samples"]

    def pat5pct(df):
        return df.groupby("trial")["true_positives_at_5pct"].sum() / (
            df.groupby("trial")["num_samples"].sum() * 0.05
        )

    def rat5pct(df):
        return (
            df.groupby("trial")["true_positives_at_5pct"].sum()
            / df.groupby("trial")["num_anomalies"].sum()
        )

    df_precision_at_5pct_ci = bootstrap_ci(pat5pct, df)
    df_precision_at_5pct = (
        pat5pct(df).map(fmt_float)
        + " ("
        + df_precision_at_5pct_ci["ci.lower"].map(fmt_float)
        + ", "
        + df_precision_at_5pct_ci["ci.upper"].map(fmt_float)
        + ")"
    )
    df_recall_at_5pct_ci = bootstrap_ci(rat5pct, df)
    df_recall_at_5pct = rat5pct(df)
    df_recall_at_5pct = (
        rat5pct(df).map(fmt_float)
        + " ("
        + df_recall_at_5pct_ci["ci.lower"].map(fmt_float)
        + ", "
        + df_recall_at_5pct_ci["ci.upper"].map(fmt_float)
        + ")"
    )
    df_pr_at_5pct = pd.concat((df_precision_at_5pct, df_recall_at_5pct), axis=1)
    df_pr_at_5pct.columns = pd.MultiIndex.from_tuples(
        [("at 5%", "precision"), ("at 5%", "recall")]
    )

    df_pr_fmt = pd.concat(
        (df_anomaly_rate, df_precision_at_k, df_pr_at_10, df_pr_at_5pct), axis=1
    )
    # for col in df_pr_fmt: df_pr_fmt[col] = df_pr_fmt[col].map(fmt_float)

    # Visualize relationship between Prec@k and dataset meta-features (per-form)
    df_plot = df[
        ["num_samples", "anomaly_proportion", "precision_at_k", "recall_at_5pct"]
    ]
    plt = (
        ggplot(
            df_plot, aes(x="anomaly_proportion", y="recall_at_5pct", size="num_samples")
        )
        + geom_point()
        + geom_smooth(method="lm")  # loess better
    )
    plt.save("test.jpg")

    df_auroc_fmt.to_csv("df_auroc.csv")
    df_pr_fmt.to_csv("df_pr.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_comparisons_dir", type=str, required=True)
    parser.add_argument("--input_granularity_dir", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    main(args)
