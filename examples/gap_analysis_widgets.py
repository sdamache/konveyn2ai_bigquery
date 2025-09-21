"""Interactive notebook widgets for Svami gap analysis visualization."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.models import GapAnalysisRequest, GapAnalysisResponse

try:
    import ipywidgets as widgets
    from IPython.display import Markdown, display
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - handled by docs
    raise ImportError(
        "gap_analysis_widgets requires ipywidgets and plotly."
    ) from exc


DEFAULT_ENDPOINT = os.getenv("SVAMI_ORCHESTRATOR_URL", "http://localhost:8000")
DEFAULT_API_TOKEN = os.getenv("SVAMI_API_TOKEN")
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_gap_analysis.json"


@dataclass
class GapVisualizationData:
    """Container for derived visualisation frames."""

    summary_text: str
    findings_frame: pd.DataFrame
    heatmap_frame: pd.DataFrame
    progress_frame: pd.DataFrame


def _load_response_from_dict(payload: dict[str, Any]) -> GapAnalysisResponse:
    """Build a GapAnalysisResponse from raw dict."""

    return GapAnalysisResponse(**payload)


def load_sample_response(path: Path | None = None) -> GapAnalysisResponse:
    """Load bundled sample response for offline mode."""

    data_path = path or SAMPLE_DATA_PATH
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _load_response_from_dict(payload)


def fetch_gap_analysis(
    topic: str,
    *,
    artifact_type: Optional[str] = None,
    rule_name: Optional[str] = None,
    limit: int = 6,
    base_url: str | None = None,
    api_token: str | None = None,
    timeout: float = 15.0,
) -> GapAnalysisResponse:
    """Call Svami orchestrator and return parsed response."""

    request = GapAnalysisRequest(
        topic=topic,
        artifact_type=artifact_type,
        rule_name=rule_name,
        limit=limit,
    )

    url = f"{(base_url or DEFAULT_ENDPOINT).rstrip('/')}/gap-analysis"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = api_token or DEFAULT_API_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, content=request.model_dump_json(), headers=headers)
        response.raise_for_status()
        payload = response.json()

    return _load_response_from_dict(payload)


def findings_to_frame(response: GapAnalysisResponse) -> pd.DataFrame:
    """Convert response findings into a structured dataframe."""

    records: list[dict[str, Any]] = []
    for finding in response.findings:
        records.append(
            {
                "chunk_id": finding.chunk_id,
                "artifact_type": finding.artifact_type or "unknown",
                "rule_name": finding.rule_name,
                "severity": finding.severity,
                "confidence": finding.confidence,
                "suggested_fix": finding.suggested_fix or "",
                "source_path": finding.source_path or "",
                "source_url": finding.source_url or "",
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "artifact_type",
                "rule_name",
                "severity",
                "confidence",
                "suggested_fix",
                "source_path",
                "source_url",
            ]
        )

    return pd.DataFrame.from_records(records)


def build_heatmap_frame(findings_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate severity by artifact type and rule for heat map."""

    if findings_frame.empty:
        return pd.DataFrame()

    pivot = (
        findings_frame
        .pivot_table(
            index="artifact_type",
            columns="rule_name",
            values="severity",
            aggfunc="mean",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    return pivot


def build_progress_frame(findings_frame: pd.DataFrame) -> pd.DataFrame:
    """Derive a synthetic progress curve based on severity ranking."""

    if findings_frame.empty:
        return pd.DataFrame(columns=["rank", "open_gaps", "confidence"])

    ordered = findings_frame.sort_values(
        by=["severity", "confidence"], ascending=[False, False]
    ).reset_index(drop=True)
    ordered.index += 1
    ordered["rank"] = ordered.index
    ordered["open_gaps"] = ordered.shape[0] - ordered.index + 1
    ordered["confidence"] = ordered["confidence"].round(2)
    return ordered[["rank", "open_gaps", "confidence"]]


def create_visualisation_data(response: GapAnalysisResponse) -> GapVisualizationData:
    """Generate dataframes for downstream plotting."""

    findings_frame = findings_to_frame(response)
    heatmap_frame = build_heatmap_frame(findings_frame)
    progress_frame = build_progress_frame(findings_frame)
    return GapVisualizationData(
        summary_text=response.summary,
        findings_frame=findings_frame,
        heatmap_frame=heatmap_frame,
        progress_frame=progress_frame,
    )


def _render_summary_markdown(data: GapVisualizationData) -> None:
    """Display textual summary using Markdown."""

    display(Markdown(data.summary_text.replace("\n", "  \n")))


def _render_findings_table(data: GapVisualizationData) -> None:
    """Render the findings dataframe with useful columns."""

    if data.findings_frame.empty:
        display(Markdown("No documented gaps in the current selection."))
        return

    columns = [
        "artifact_type",
        "rule_name",
        "severity",
        "confidence",
        "suggested_fix",
        "source_path",
    ]
    display(data.findings_frame[columns])


def _render_heatmap(data: GapVisualizationData) -> None:
    """Plot severity heatmap with Plotly."""

    if data.heatmap_frame.empty:
        display(Markdown("Heat map unavailable – no gap metrics for current filters."))
        return

    fig = px.imshow(
        data.heatmap_frame,
        text_auto=".1f",
        color_continuous_scale="Reds",
        labels=dict(x="Rule", y="Artifact", color="Severity"),
        title="Gap Severity Heat Map",
    )
    fig.update_layout(height=420, margin=dict(l=60, r=20, t=60, b=40))
    fig.show()


def _render_progress(data: GapVisualizationData) -> None:
    """Visualise open gap countdown as interactive line chart."""

    if data.progress_frame.empty:
        display(Markdown("Progress chart unavailable – no gap metrics for current filters."))
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.progress_frame["rank"],
            y=data.progress_frame["open_gaps"],
            mode="lines+markers",
            name="Open gaps",
        )
    )
    fig.update_layout(
        title="Progress toward closing gaps",
        xaxis_title="Finding rank (highest severity first)",
        yaxis_title="Open gaps remaining",
        height=360,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.show()


def create_gap_analysis_dashboard(
    *,
    base_url: Optional[str] = None,
    api_token: Optional[str] = None,
    offline: bool | None = None,
    sample_path: Optional[Path] = None,
    default_topic: str = "FastAPI onboarding",
) -> None:
    """Display interactive widgets for requesting gap analysis data."""

    resolved_offline = (
        offline
        if offline is not None
        else os.getenv("GAP_ANALYSIS_OFFLINE", "0") not in {"0", "false", "False"}
    )

    topic_input = widgets.Text(
        value=default_topic,
        description="Topic:",
        layout=widgets.Layout(width="60%"),
    )
    artifact_input = widgets.Text(
        value="fastapi",
        description="Artifact:",
        layout=widgets.Layout(width="30%"),
    )
    rule_input = widgets.Text(
        value="",
        description="Rule:",
        layout=widgets.Layout(width="30%"),
    )
    limit_input = widgets.IntSlider(
        value=6,
        min=1,
        max=15,
        step=1,
        description="Limit:",
    )
    mode_label = widgets.HTML(
        value=(
            "<b>Offline preview:</b> using bundled sample data"
            if resolved_offline
            else "<b>Live mode:</b> calling Svami orchestrator"
        )
    )
    submit_button = widgets.Button(
        description="Run gap analysis",
        button_style="primary",
        icon="search",
    )
    output_area = widgets.Output(
        layout=widgets.Layout(border="1px solid var(--jp-border-color2)")
    )

    def _run_analysis(_: Any) -> None:
        with output_area:
            output_area.clear_output()
            try:
                if resolved_offline:
                    response = load_sample_response(sample_path)
                else:
                    response = fetch_gap_analysis(
                        topic_input.value,
                        artifact_type=artifact_input.value or None,
                        rule_name=rule_input.value or None,
                        limit=limit_input.value,
                        base_url=base_url,
                        api_token=api_token,
                    )
            except Exception as exc:  # Display fallback and message
                display(Markdown(f"**Error contacting orchestrator:** {exc}"))
                display(Markdown("Loading bundled sample data instead."))
                response = load_sample_response(sample_path)

            data = create_visualisation_data(response)
            _render_summary_markdown(data)
            _render_findings_table(data)
            _render_heatmap(data)
            _render_progress(data)

    submit_button.on_click(_run_analysis)

    controls = widgets.HBox([topic_input, artifact_input, rule_input])
    layout = widgets.VBox(
        [
            controls,
            limit_input,
            mode_label,
            widgets.HBox([submit_button]),
            output_area,
        ]
    )

    display(layout)

    _run_analysis(None)


__all__ = [
    "GapVisualizationData",
    "fetch_gap_analysis",
    "load_sample_response",
    "findings_to_frame",
    "build_heatmap_frame",
    "build_progress_frame",
    "create_visualisation_data",
    "create_gap_analysis_dashboard",
]
