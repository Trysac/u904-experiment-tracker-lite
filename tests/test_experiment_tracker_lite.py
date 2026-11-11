"""Comprehensive unit and acceptance tests for experiment_tracker_lite."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

import experiment_tracker_lite as etl
from experiment_tracker_lite import (
    Tracker,
    TrackerError,
    build_site,
    get_run,
    log_artifact,
    log_metrics,
    log_params,
    main,
    search_runs,
    set_experiment,
    set_tag,
    set_tracking_root,
    start_run,
)


class FakeClock:
    """Deterministic millisecond clock for tests."""

    def __init__(self, initial: int = 1_700_000_000_000, step: int = 100) -> None:
        self.current = initial
        self.step = step

    def __call__(self) -> int:
        val = self.current
        self.current += self.step
        return val


class FakeIds:
    """Deterministic ID generator for tests."""

    def __init__(self, prefix: str = "run") -> None:
        self.counter = 0
        self.prefix = prefix

    def __call__(self) -> str:
        self.counter += 1
        return f"{self.prefix}_{self.counter:04d}"


def test_acceptance_1_lifecycle_and_latest_metric(tmp_path: Path) -> None:
    """Acceptance 1: With a fixed clock and run ids, start_run(run_name='a') then log_param('lr', '0.01'), log_metric('acc', 0.8, step=0) and log_metric('acc', 0.92, step=1) and end_run(): get_run(run_id).data.metrics == {'acc': 0.92}, data.params == {'lr': '0.01'} and info.status == 'FINISHED'."""
    clock = FakeClock()
    ids = FakeIds()
    tracker = Tracker(tmp_path / "mlruns", clock=clock, new_id=ids)

    active = tracker.start_run(run_name="a")
    run_id = active.run_id
    assert active.info.status == "RUNNING"
    tracker.log_param("lr", "0.01")
    tracker.log_metric("acc", 0.8, step=0)
    tracker.log_metric("acc", 0.92, step=1)
    tracker.end_run()

    run = tracker.get_run(run_id)
    assert run.data.metrics == {"acc": 0.92}
    assert run.data.params == {"lr": "0.01"}
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "a"
    assert run.info.end_time is not None
    assert run.info.end_time > run.info.start_time
    assert run.run_id == run_id


def test_acceptance_2_search_filter_and_order_by(tmp_path: Path) -> None:
    """Acceptance 2: Runs with acc 0.92 (lr '0.01'), 0.85 (lr '0.01') and 0.7 (lr '0.1'): search_runs(filter_string=\"metrics.acc > 0.9 and params.lr = '0.01'\") returns only the first, and search_runs(order_by=['metrics.acc DESC']) returns the runs in the order 0.92, 0.85, 0.7."""
    clock = FakeClock()
    ids = FakeIds()
    tracker = Tracker(tmp_path / "mlruns", clock=clock, new_id=ids)

    with tracker.start_run(run_name="r1") as r1:
        tracker.log_param("lr", "0.01")
        tracker.log_metric("acc", 0.92)
    with tracker.start_run(run_name="r2"):
        tracker.log_param("lr", "0.01")
        tracker.log_metric("acc", 0.85)
    with tracker.start_run(run_name="r3"):
        tracker.log_param("lr", "0.1")
        tracker.log_metric("acc", 0.7)

    filtered = tracker.search_runs(filter_string="metrics.acc > 0.9 and params.lr = '0.01'")
    assert [r.run_id for r in filtered] == [r1.run_id]

    ordered = tracker.search_runs(order_by=["metrics.acc DESC"])
    assert [r.data.metrics["acc"] for r in ordered] == [0.92, 0.85, 0.7]


def test_acceptance_3_param_immutability_and_length_limits(tmp_path: Path) -> None:
    """Acceptance 3: log_param('lr', '0.02') after log_param('lr', '0.01') in the same run raises TrackerError with code 'invalid_parameter_value'; a value of 6_001 characters raises the same code while 6_000 is stored."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())
    with tracker.start_run(run_name="param_test") as active:
        tracker.log_param("lr", "0.01")
        # Idempotent re-logging of identical value is allowed
        tracker.log_param("lr", "0.01")
        with pytest.raises(TrackerError) as exc_info:
            tracker.log_param("lr", "0.02")
        assert exc_info.value.code == "invalid_parameter_value"

        val_6000 = "x" * 6_000
        tracker.log_param("p6000", val_6000)

        val_6001 = "x" * 6_001
        with pytest.raises(TrackerError) as exc_len:
            tracker.log_param("p6001", val_6001)
        assert exc_len.value.code == "invalid_parameter_value"

    stored = tracker.get_run(active.run_id)
    assert stored.data.params["p6000"] == val_6000
    assert "p6001" not in stored.data.params


def test_acceptance_4_error_codes(tmp_path: Path) -> None:
    """Acceptance 4: log_metric after end_run raises code 'invalid_state'; get_run('missing') raises code 'resource_does_not_exist'; filter 'metrics.acc >> 1' raises code 'bad_filter'."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())
    tracker.start_run(run_name="err_test")
    tracker.end_run()

    with pytest.raises(TrackerError) as exc_state:
        tracker.log_metric("acc", 0.5)
    assert exc_state.value.code == "invalid_state"

    with pytest.raises(TrackerError) as exc_missing:
        tracker.get_run("missing")
    assert exc_missing.value.code == "resource_does_not_exist"

    with pytest.raises(TrackerError) as exc_filter:
        tracker.search_runs(filter_string="metrics.acc >> 1")
    assert exc_filter.value.code == "bad_filter"


def test_acceptance_5_context_manager_exception_marks_failed(tmp_path: Path) -> None:
    """Acceptance 5: An exception raised inside with start_run(): ends the run with info.status == 'FAILED' and propagates unchanged."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())

    class CustomBoomError(RuntimeError):
        pass

    run_id_holder: list[str] = []
    with (
        pytest.raises(CustomBoomError, match="training exploded"),
        tracker.start_run(run_name="failing_run") as active,
    ):
        run_id_holder.append(active.run_id)
        tracker.log_param("batch_size", 32)
        raise CustomBoomError("training exploded")

    run = tracker.get_run(run_id_holder[0])
    assert run.info.status == "FAILED"
    assert run.info.end_time is not None
    assert run.data.params == {"batch_size": "32"}


def test_acceptance_6_build_site_reproducible_and_html_escaped(tmp_path: Path) -> None:
    """Acceptance 6: build_site(out_dir) twice from the same store produces files with identical sha256 digests; a run named '<b>x</b>' appears in index.html as '&lt;b&gt;x&lt;/b&gt;'."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())
    with tracker.start_run(run_name="<b>x</b>"):
        tracker.log_param("tag_like", "<script>alert(1)</script>")
        tracker.log_metric("loss", 0.1234)
        tracker.set_tag("owner", "alice & bob")

    out1 = tmp_path / "site1"
    out2 = tmp_path / "site2"
    files1 = tracker.build_site(out1)
    files2 = tracker.build_site(out2)

    assert [f.name for f in files1] == [f.name for f in files2]
    for f1, f2 in zip(files1, files2, strict=True):
        h1 = hashlib.sha256(f1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(f2.read_bytes()).hexdigest()
        assert h1 == h2

    index_html = (out1 / "index.html").read_text(encoding="utf-8")
    assert "&lt;b&gt;x&lt;/b&gt;" in index_html
    assert "<b>x</b>" not in index_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in index_html


def test_acceptance_7_cli_compare_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Acceptance 7: main(['compare', '--root', 'mlruns', id1, id2]) prints a table with one row per param and metric and returns 0; an unknown run id returns 1; missing arguments return 2."""
    root = tmp_path / "mlruns"
    tracker = Tracker(root, clock=FakeClock(), new_id=FakeIds())
    with tracker.start_run(run_name="first") as r1:
        tracker.log_param("lr", "0.01")
        tracker.log_metric("acc", 0.91)
    with tracker.start_run(run_name="second") as r2:
        tracker.log_param("lr", "0.05")
        tracker.log_metric("acc", 0.84)

    rc = main(["compare", "--root", str(root), r1.run_id, r2.run_id])
    assert rc == 0
    captured = capsys.readouterr()
    assert "param.lr" in captured.out
    assert "metric.acc" in captured.out
    assert r1.run_id in captured.out
    assert r2.run_id in captured.out

    rc_unknown = main(["compare", "--root", str(root), r1.run_id, "unknown_id"])
    assert rc_unknown == 1

    rc_missing = main(["compare", "--root", str(root), r1.run_id])
    assert rc_missing == 2

    rc_no_cmd = main([])
    assert rc_no_cmd == 2


def test_module_level_fluent_api_and_artifacts(tmp_path: Path) -> None:
    """Exercise module-level fluent functions mirroring mlflow 3.x."""
    root = tmp_path / "fluent_mlruns"
    set_tracking_root(root)
    exp_id = set_experiment("vision-exp")
    assert exp_id == "1"

    artifact_file = tmp_path / "weights.txt"
    artifact_file.write_text("model-weights-v1", encoding="utf-8")
    artifact_dir = tmp_path / "plots"
    artifact_dir.mkdir()
    (artifact_dir / "curve.csv").write_text("step,val\n0,0.5\n", encoding="utf-8")

    with start_run(run_name="fluent-run") as run_ctx:
        log_params({"epochs": 10, "optimizer": "adam"})
        log_metrics({"loss": 0.25, "f1": 0.88}, step=1)
        set_tag("env", "ci")
        log_artifact(str(artifact_file))
        log_artifact(str(artifact_dir), artifact_path="eval")
        rid = run_ctx.run_id

    fetched = get_run(rid)
    assert fetched.data.params == {"epochs": "10", "optimizer": "adam"}
    assert fetched.data.metrics == {"loss": 0.25, "f1": 0.88}
    assert fetched.data.tags == {"env": "ci"}

    runs = search_runs(experiment_names=["vision-exp"])
    assert len(runs) == 1
    assert runs[0].run_id == rid

    site_dir = tmp_path / "fluent_site"
    built = build_site(site_dir)
    assert len(built) == 2
    assert (site_dir / "index.html").exists()

    # Also test CLI site subcommand
    assert main(["site", "--root", str(root), "--out", str(tmp_path / "cli_site")]) == 0


def test_filter_grammar_operators_wildcards_backticks_and_nan(tmp_path: Path) -> None:
    """Test all filter operators, backtick keys, LIKE/ILIKE wildcards, NaN metrics, and missing keys."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())
    with tracker.start_run(run_name="alpha_model") as r1:
        tracker.log_param("model.type", "resnet_50")
        tracker.log_metric("val/loss", 0.15, step=0)
        tracker.log_metric("nan_metric", float("nan"), step=0)
        tracker.set_tag("team", "CoreML")

    with tracker.start_run(run_name="beta_model") as r2:
        tracker.log_param("model.type", "vit_base")
        tracker.log_metric("val/loss", 0.35, step=0)
        tracker.set_tag("team", "coreml")
    tracker.end_run()

    # Backtick-quoted key with slash and dot
    res_bt = tracker.search_runs(
        filter_string="metrics.`val/loss` <= 0.2 AND params.`model.type` != 'vit_base'"
    )
    assert [r.run_id for r in res_bt] == [r1.run_id]

    # LIKE (case-sensitive) and ILIKE (case-insensitive) with % and _
    res_like = tracker.search_runs(filter_string="tags.team LIKE 'Core%'")
    assert [r.run_id for r in res_like] == [r1.run_id]

    res_ilike = tracker.search_runs(
        filter_string="tags.team ILIKE 'core_l' and attributes.run_name LIKE '%_model'"
    )
    assert {r.run_id for r in res_ilike} == {r1.run_id, r2.run_id}

    # Attributes start_time and status
    res_attr = tracker.search_runs(
        filter_string=f"attributes.start_time >= {r1.info.start_time} and attributes.status = 'FINISHED'"
    )
    assert len(res_attr) == 2

    # NaN metric never matches numeric comparisons
    assert tracker.search_runs(filter_string="metrics.nan_metric = 0") == []
    assert tracker.search_runs(filter_string="metrics.nan_metric != 0") == []

    # Ordering with missing/NaN values places them last in both ASC and DESC
    with tracker.start_run(run_name="gamma_no_loss") as r3:
        tracker.log_param("model.type", "mlp")
    ordered_asc = tracker.search_runs(order_by=["metrics.`val/loss` ASC"])
    assert [r.run_id for r in ordered_asc] == [r1.run_id, r2.run_id, r3.run_id]

    ordered_desc = tracker.search_runs(order_by=["metrics.`val/loss` DESC"])
    assert [r.run_id for r in ordered_desc] == [r2.run_id, r1.run_id, r3.run_id]


@pytest.mark.parametrize(
    "bad_filter",
    [
        "unknown.key = 1",
        "metrics = 1",
        "metrics. = 1",
        "metrics.`unterminated = 1",
        "metrics.`` = 1",
        "attributes.unsupported = 'x'",
        "metrics.acc LIKE '1%'",
        "params.lr > 0.1",
        "params.lr = 0.1",
        "params.lr = 'unterminated",
        "params.lr = 'bad'extra",
        "metrics.acc = 1.2.3",
        "metrics.acc = 1 OR metrics.acc = 2",
        "metrics.acc = 1 AND ",
        "metrics.acc",
        "x" * 1001,
    ],
)
def test_bad_filters_rejected(tmp_path: Path, bad_filter: str) -> None:
    """Adversarial filter syntax probes must all raise TrackerError('bad_filter')."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())
    with pytest.raises(TrackerError) as exc_info:
        tracker.search_runs(filter_string=bad_filter)
    assert exc_info.value.code == "bad_filter"


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "a" * 251,
        "../secret",
        "a/../b",
        "./a",
        "key$invalid",
        "key;drop",
    ],
)
def test_invalid_keys_rejected(tmp_path: Path, bad_key: str) -> None:
    """Keys violating MLflow rules or containing traversal segments are rejected."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())
    with tracker.start_run():
        with pytest.raises(TrackerError) as exc_info:
            tracker.log_param(bad_key, "v")
        assert exc_info.value.code == "invalid_parameter_value"


def test_validation_and_state_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover remaining validation, state, and limit error branches."""
    tracker = Tracker(tmp_path / "mlruns", clock=FakeClock(), new_id=FakeIds())

    # Calling end_run when no run is active is a safe no-op (like mlflow.end_run)
    tracker.end_run()

    with tracker.start_run():
        # Cannot start nested run
        with pytest.raises(TrackerError) as exc_nested:
            tracker.start_run()
        assert exc_nested.value.code == "invalid_state"

        # Cannot change tracking root while active
        with pytest.raises(TrackerError) as exc_root:
            tracker.set_tracking_root(tmp_path / "other")
        assert exc_root.value.code == "invalid_state"

        # Cannot end run with status RUNNING or invalid status
        with pytest.raises(TrackerError) as exc_stat:
            tracker.end_run(status="RUNNING")
        assert exc_stat.value.code == "invalid_parameter_value"
        with pytest.raises(TrackerError) as exc_stat2:
            tracker.end_run(status="UNKNOWN")
        assert exc_stat2.value.code == "invalid_parameter_value"

        # Infinite or boolean metric rejected
        with pytest.raises(TrackerError):
            tracker.log_metric("m", math.inf)
        with pytest.raises(TrackerError):
            tracker.log_metric("m", True)
        with pytest.raises(TrackerError):
            tracker.log_metric("m", 1.0, step=True)

        # Artifact path traversal or missing file rejected
        f = tmp_path / "f.txt"
        f.write_text("ok", encoding="utf-8")
        with pytest.raises(TrackerError):
            tracker.log_artifact(f, artifact_path="../escape")
        with pytest.raises(TrackerError):
            tracker.log_artifact(tmp_path / "does_not_exist.txt")

        # Metric point limit exceeded
        monkeypatch.setattr("experiment_tracker_lite.storage.MAX_METRIC_POINTS", 2)
        tracker.log_metric("limited", 1.0, step=0)
        tracker.log_metric("limited", 2.0, step=1)
        with pytest.raises(TrackerError) as exc_lim:
            tracker.log_metric("limited", 3.0, step=2)
        assert exc_lim.value.code == "limit_exceeded"

        # Artifact size limit exceeded
        monkeypatch.setattr("experiment_tracker_lite.storage.MAX_ARTIFACT_BYTES", 5)
        big = tmp_path / "big.bin"
        big.write_bytes(b"123456")
        with pytest.raises(TrackerError) as exc_art_lim:
            tracker.log_artifact(big)
        assert exc_art_lim.value.code == "limit_exceeded"

    # Bad order_by syntax
    with pytest.raises(TrackerError) as exc_ob:
        tracker.search_runs(order_by=["metrics.acc SIDEWAYS"])
    assert exc_ob.value.code == "bad_filter"

    assert etl.__version__ == "0.1.0"
