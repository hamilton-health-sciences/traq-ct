import multiprocessing as mp
import os
import pickle
from functools import partial, reduce
from typing import Dict, List

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_density,
    geom_jitter,
    geom_tile,
    ggplot,
    lims,
    scale_x_continuous,
    xlab,
    ylab,
    ylim,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from traq.data import PickledStudy

changes_thresholds = [0.0, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5]


def quantify_granularity_plate(plate, predictions) -> List[Dict]:
    # X = plate.to_frame()
    Y = plate.labels_frame()

    # variable_anomaly_proportion = Y["any_change"].mean()
    result = {"name": plate.name, "num_samples": len(Y)}
    plate_labels = []
    for threshold in changes_thresholds:
        plate_labels_threshold = Y["change_frac"] > threshold
        result[f"record_anomaly_proportion_{threshold}"] = plate_labels_threshold.mean()
        plate_labels.append(plate_labels_threshold.to_frame())
    plate_labels_df = pd.concat(plate_labels, axis=1)
    num_records_df = (
        plate_labels_df.groupby(["centre", "id"]).count().iloc[:, 0].to_frame()
    )
    num_records_df.columns = [plate.name]
    plate_labels_df.columns = pd.MultiIndex.from_product(
        [[plate.name], changes_thresholds], names=["plate", "record_threshold"]
    )
    plate_labels_df = (
        plate_labels_df.reset_index()
        .groupby(["centre", "id"])[plate_labels_df.columns]
        .sum()
    )
    performances = []
    if plate.name in predictions:
        Y_pred = predictions[plate.name]
        for algo in Y_pred:
            for record_threshold in plate_labels_df.columns.get_level_values(
                "record_threshold"
            ):
                y = np.array(plate_labels_df.loc[:, (slice(None), record_threshold)])
                yhat = Y_pred[algo]
                try:
                    auroc = roc_auc_score(y, yhat)
                    aupr = average_precision_score(y, yhat)
                except:  # noqa: E722
                    auroc = np.nan
                    aupr = np.nan
                performances.append(
                    {
                        "plate": plate.name,
                        "algorithm": str(algo),
                        "record_threshold": record_threshold,
                        "auroc": auroc,
                        "aupr": aupr,
                        "anomaly_rate": y.mean(),
                        "num_samples": len(y),
                    }
                )
        performance_df = pd.DataFrame(performances)
    else:
        performance_df = None

    return [result], plate_labels_df, num_records_df, performance_df


def quantify_granularity(dataset, predictions, num_workers=0):
    """
    Params:
        dataset: The dataset to evaluate PyOD models.
        predictions: The predictions of models.
    """
    f = partial(quantify_granularity_plate, predictions=predictions)
    plates = dataset.plates()
    results = []
    platewise_labels = []
    record_count_dfs = []
    performance_dfs = []
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            for plate_results, plate_labels, record_counts in tqdm(
                pool.imap_unordered(f, plates), total=len(plates)
            ):
                if plate_results is not None:
                    results += plate_results
                    platewise_labels.append(plate_labels)
                    record_count_dfs.append(record_counts)
    else:
        for plate_results, plate_labels, record_counts, performances in map(f, plates):
            if plate_results is not None:
                results += plate_results
                platewise_labels.append(plate_labels)
                record_count_dfs.append(record_counts)
                performance_dfs.append(performances)
    results_df = pd.DataFrame(results)
    performances_df = pd.concat(performance_dfs, axis=0)
    platewise_labels_df = reduce(
        partial(pd.merge, left_index=True, right_index=True, how="outer"),
        platewise_labels,
    )
    record_counts_df = reduce(
        partial(pd.merge, left_index=True, right_index=True, how="outer"),
        record_count_dfs,
    )
    participant_table_change_proportions = (
        platewise_labels_df.groupby(["record_threshold"], axis=1)
        .sum()
        .div(record_counts_df.sum(axis=1), axis=0)
    )
    participant_table_label_dfs = []
    for threshold in changes_thresholds:
        participant_table_label_dfs.append(
            participant_table_change_proportions > threshold
        )
    participant_table_labels = pd.concat(participant_table_label_dfs, axis=1)
    participant_table_labels.columns = pd.MultiIndex.from_product(
        [changes_thresholds, changes_thresholds],
        names=["participant_threshold", "record_threshold"],
    )
    performances_df["any_anomalies"] = ~performances_df["auroc"].isnull()
    performances_df["anomaly_rate_nonzero"] = performances_df["anomaly_rate"].replace(
        {0.0: np.nan}
    )
    performances_df["sample_weighted_auroc"] = (
        performances_df["auroc"] * performances_df["num_samples"]
    )
    performance_report = performances_df.groupby(["algorithm", "record_threshold"])[
        ["auroc", "aupr", "anomaly_rate_nonzero", "anomaly_rate", "any_anomalies"]
    ].mean()
    performance_report = performance_report.join(
        performances_df.groupby(["algorithm", "record_threshold"])[
            "sample_weighted_auroc"
        ].sum()
        / np.array(
            performances_df.groupby(["algorithm", "record_threshold"])[
                "num_samples"
            ].sum()
        )
    )

    return results_df, participant_table_labels, performances_df, performance_report


def main(args):
    os.makedirs(args.output_directory, exist_ok=True)

    predictions = pickle.load(open(args.input_predictions, "rb"))

    results = []
    participant_labels_snapshots = []
    performances = []
    performance_reports = []
    for snapshot_filename in tqdm(os.listdir(args.input_directory)):
        snapshot_filepath = os.path.join(args.input_directory, snapshot_filename)
        # trial_name = os.path.basename(args.input_directory)
        snapshot_name = snapshot_filename.split(".")[0]
        dataset = PickledStudy(snapshot_filepath)
        (
            results_snapshot,
            participant_labels_snapshot,
            performances_df,
            performance_report,
        ) = quantify_granularity(dataset, predictions[snapshot_name])
        results_snapshot["snapshot"] = snapshot_name
        results.append(results_snapshot)
        participant_labels_snapshot = pd.concat(
            [participant_labels_snapshot], keys=[snapshot_name], names=["snapshot"]
        )
        participant_labels_snapshots.append(participant_labels_snapshot)
        performances_df["snapshot"] = snapshot_name
        performance_report["snapshot"] = snapshot_name
        performances.append(performances_df)
        performance_reports.append(performance_report)
    results_df = pd.concat(results, axis=0)
    participant_labels_df = (
        pd.concat(participant_labels_snapshots, axis=0).groupby("snapshot").mean()
    )
    performances_df = pd.concat(performances, axis=0)
    # performance_reports_df = pd.concat(performance_reports, axis=0)

    results_df_filename = os.path.join(args.output_directory, "results.csv")
    results_df.to_csv(results_df_filename)

    performances_df_filename = os.path.join(args.output_directory, "performances.csv")
    performances_df.to_csv(performances_df_filename)

    plot_df = results_df.melt(id_vars=["snapshot", "name", "num_samples"])
    plot_df["threshold"] = pd.Categorical(
        plot_df["variable"].apply(lambda s: float(s.split("_")[-1]))
    )
    plot_df = plot_df.rename({"value": "anomalies (pct)"}, axis=1)
    plot_filename = os.path.join(args.output_directory, "anomaly_prop.jpg")
    anomaly_prop_plt = (
        ggplot(
            plot_df,
            aes(
                x="threshold", y="anomalies (pct)", size="num_samples", color="snapshot"
            ),
        )
        + geom_jitter(width=0.2, height=0.0)
        + xlab("threshold")
        + ylim([0.0, 1.0])
    )
    anomaly_prop_plt.save(plot_filename, height=8, width=8, dpi=300)

    plot_df = participant_labels_df.reset_index().melt(id_vars=["snapshot"])
    plot_filename = os.path.join(args.output_directory, "participant_results.jpg")
    plot_df["record_threshold"] = pd.Categorical(plot_df["record_threshold"])
    plot_df["participant_threshold"] = pd.Categorical(plot_df["participant_threshold"])
    plot_df = plot_df.rename({"value": "anomaly proportion"}, axis=1)
    participant_results_plt = (
        ggplot(
            plot_df,
            aes(
                x="record_threshold",
                y="participant_threshold",
                fill="anomaly proportion",
            ),
        )
        + geom_tile()
        + facet_wrap("~ snapshot")
        + lims(fill=[0.0, 1.0])
    )
    participant_results_plt.save(plot_filename, height=8, width=8, dpi=300)

    plot_df = results_df.melt(id_vars=["snapshot", "name"])
    plot_filename = os.path.join(args.output_directory, "results.jpg")
    results_plt = (
        ggplot(
            plot_df, aes(x="value", group="variable", color="variable", fill="variable")
        )
        + geom_density()
        + facet_wrap("~ snapshot")
        + xlab("proportion of anomalies")
        + ylab("number of tables")
        + scale_x_continuous(trans="sqrt")
    )
    results_plt.save(plot_filename, height=8, width=8, dpi=300)

    # Performance plots
    plot_df = performances_df[performances_df["algorithm"].str.contains("ECOD")]
    plot_filename = os.path.join(args.output_directory, "performances.jpg")
    plot_df["record_threshold"] = pd.Categorical(plot_df["record_threshold"])
    performances_plt = (
        ggplot(
            plot_df,
            aes(x="record_threshold", y="auroc", size="num_samples", color="snapshot"),
        )
        + geom_jitter(width=0.2, height=0.0)
        + xlab("threshold")
        + ylim([0.0, 1.0])
    )
    performances_plt.save(plot_filename, height=8, width=8, dpi=300)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--input_predictions", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    args = parser.parse_args()

    main(args)
