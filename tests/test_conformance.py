"""CI-only conformance test comparing experiment_tracker_lite against mlflow 3.18.0."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment_tracker_lite import Tracker


@pytest.mark.conformance
def test_mlflow_search_runs_conformance(tmp_path: Path) -> None:
    """Replay a 3-run scenario in both mlflow and experiment_tracker_lite and compare search_runs."""
    mlflow = pytest.importorskip("mlflow")

    etl_root = tmp_path / "etl_mlruns"
    mlf_root = tmp_path / "mlf_mlruns"

    clock_val = [1_700_000_000_000]

    def clock() -> int:
        clock_val[0] += 1000
        return clock_val[0]

    id_counter = [0]

    def new_id() -> str:
        id_counter[0] += 1
        return f"run{id_counter[0]:04d}"

    tracker = Tracker(etl_root, clock=clock, new_id=new_id)
    tracker.set_experiment("conformance_exp")

    mlflow.set_tracking_uri(mlf_root.as_uri())
    mlflow.set_experiment("conformance_exp")

    # Run 1: finished, two metric steps
    with tracker.start_run(run_name="run_alpha"):
        tracker.log_param("lr", "0.01")
        tracker.log_param("opt", "adam")
        tracker.log_metric("acc", 0.75, step=0)
        tracker.log_metric("acc", 0.92, step=1)
        tracker.log_metric("loss", 0.18, step=1)
        tracker.set_tag("stage", "prod_v1")

    with mlflow.start_run(run_name="run_alpha"):
        mlflow.log_param("lr", "0.01")
        mlflow.log_param("opt", "adam")
        mlflow.log_metric("acc", 0.75, step=0)
        mlflow.log_metric("acc", 0.92, step=1)
        mlflow.log_metric("loss", 0.18, step=1)
        mlflow.set_tag("stage", "prod_v1")

    # Run 2: finished
    with tracker.start_run(run_name="run_beta"):
        tracker.log_param("lr", "0.01")
        tracker.log_param("opt", "sgd")
        tracker.log_metric("acc", 0.85, step=0)
        tracker.log_metric("loss", 0.30, step=0)
        tracker.set_tag("stage", "staging_v2")

    with mlflow.start_run(run_name="run_beta"):
        mlflow.log_param("lr", "0.01")
        mlflow.log_param("opt", "sgd")
        mlflow.log_metric("acc", 0.85, step=0)
        mlflow.log_metric("loss", 0.30, step=0)
        mlflow.set_tag("stage", "staging_v2")

    # Run 3: failed run
    try:
        with tracker.start_run(run_name="run_gamma"):
            tracker.log_param("lr", "0.1")
            tracker.log_param("opt", "sgd")
            tracker.log_metric("acc", 0.60, step=0)
            tracker.set_tag("stage", "dev_v1")
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    try:
        with mlflow.start_run(run_name="run_gamma"):
            mlflow.log_param("lr", "0.1")
            mlflow.log_param("opt", "sgd")
            mlflow.log_metric("acc", 0.60, step=0)
            mlflow.set_tag("stage", "dev_v1")
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    filters = [
        "metrics.acc > 0.8 and params.lr = '0.01'",
        "params.opt = 'sgd'",
        "attributes.status = 'FAILED'",
        "tags.stage LIKE 'prod%'",
        "tags.stage ILIKE '%V1' and metrics.acc >= 0.6",
    ]
    for fstr in filters:
        etl_runs = tracker.search_runs(
            experiment_names=["conformance_exp"],
            filter_string=fstr,
            order_by=["metrics.acc DESC"],
        )
        mlf_runs = mlflow.search_runs(
            experiment_names=["conformance_exp"],
            filter_string=fstr,
            order_by=["metrics.acc DESC"],
            output_format="list",
        )
        assert [r.info.run_name for r in etl_runs] == [r.info.run_name for r in mlf_runs]
        for er, mr in zip(etl_runs, mlf_runs, strict=True):
            assert er.data.params == mr.data.params
            assert er.data.metrics == mr.data.metrics
            assert er.info.status == mr.info.status

    order_bys = [
        ["metrics.acc ASC"],
        ["params.opt ASC", "metrics.acc DESC"],
    ]
    for ob in order_bys:
        etl_runs = tracker.search_runs(experiment_names=["conformance_exp"], order_by=ob)
        mlf_runs = mlflow.search_runs(
            experiment_names=["conformance_exp"],
            order_by=ob,
            output_format="list",
        )
        assert [r.info.run_name for r in etl_runs] == [r.info.run_name for r in mlf_runs]
