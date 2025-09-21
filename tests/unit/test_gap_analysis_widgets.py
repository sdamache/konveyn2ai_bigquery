import pathlib

import pandas as pd

from examples.gap_analysis_widgets import (
    build_heatmap_frame,
    build_progress_frame,
    create_visualisation_data,
    findings_to_frame,
    load_sample_response,
)


DATA_PATH = pathlib.Path("examples/data/sample_gap_analysis.json")


def test_sample_response_loads():
    response = load_sample_response(DATA_PATH)
    assert response.total_results == 3
    assert response.topic == "FastAPI onboarding"


def test_findings_to_frame_columns():
    response = load_sample_response(DATA_PATH)
    frame = findings_to_frame(response)
    expected_columns = {
        "chunk_id",
        "artifact_type",
        "rule_name",
        "severity",
        "confidence",
        "suggested_fix",
        "source_path",
        "source_url",
    }
    assert set(frame.columns) == expected_columns
    assert frame.shape[0] == response.total_results


def test_heatmap_frame_has_expected_shape():
    response = load_sample_response(DATA_PATH)
    frame = findings_to_frame(response)
    heatmap = build_heatmap_frame(frame)
    assert not heatmap.empty
    assert list(heatmap.index) == sorted(heatmap.index.tolist())


def test_progress_frame_counts_down():
    response = load_sample_response(DATA_PATH)
    frame = findings_to_frame(response)
    progress = build_progress_frame(frame)
    assert progress["open_gaps"].tolist() == [3, 2, 1]


def test_visualisation_data_combines_frames():
    response = load_sample_response(DATA_PATH)
    data = create_visualisation_data(response)
    assert isinstance(data.findings_frame, pd.DataFrame)
    assert isinstance(data.heatmap_frame, pd.DataFrame)
    assert isinstance(data.progress_frame, pd.DataFrame)
    assert data.summary_text.startswith("Top gaps")
