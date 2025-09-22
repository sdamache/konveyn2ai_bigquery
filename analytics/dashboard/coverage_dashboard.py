"""Streamlit dashboard for documentation coverage progress monitoring."""

from __future__ import annotations

import datetime as dt
from typing import Iterable

import altair as alt
import pandas as pd
import streamlit as st
from google.api_core import exceptions as gcloud_exceptions
from google.cloud import bigquery

DEFAULT_PROJECT = "konveyn2ai"
DEFAULT_DATASET = "semantic_gap_detector"
DEFAULT_TABLE = "documentation_progress_snapshots"


def safe_int(value: float | int | None) -> int:
    """Convert a numeric value to int while tolerating missing data."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def configure_page() -> None:
    st.set_page_config(
        page_title="Documentation Coverage Progress",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Documentation Coverage Progress")
    st.caption(
        "Daily snapshots of documentation coverage, gap remediation, and rule-level progress."
    )


def build_client(project_id: str) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def _date_range_to_tuple(date_values: Iterable[dt.date]) -> tuple[dt.date, dt.date]:
    values = list(date_values)
    if not values:
        today = dt.date.today()
        return today - dt.timedelta(days=29), today
    if len(values) == 1:
        return values[0], values[0]
    return min(values), max(values)


@st.cache_data(show_spinner=False, ttl=900)
def load_progress_timeseries(
    project_id: str,
    dataset: str,
    table: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    client = build_client(project_id)
    query = f"""
        SELECT
            snapshot_date,
            coverage_percentage,
            total_chunks_processed,
            total_chunks_documented,
            total_gaps_detected,
            total_gaps_open,
            total_gaps_closed,
            mean_confidence_score
        FROM `{project_id}.{dataset}.{table}`
        WHERE snapshot_date BETWEEN @start_date AND @end_date
        ORDER BY snapshot_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date.isoformat()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date.isoformat()),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe(
        create_bqstorage_client=False
    )
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    numeric_columns = [
        "coverage_percentage",
        "total_chunks_processed",
        "total_chunks_documented",
        "total_gaps_detected",
        "total_gaps_open",
        "total_gaps_closed",
        "mean_confidence_score",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=900)
def load_severity_heatmap(
    project_id: str,
    dataset: str,
    table: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    client = build_client(project_id)
    query = f"""
        SELECT
            snapshot_date,
            severity_bucket.severity AS severity,
            severity_bucket.gaps_open AS gaps_open
        FROM `{project_id}.{dataset}.{table}`
        CROSS JOIN UNNEST(severity_distribution) AS severity_bucket
        WHERE snapshot_date BETWEEN @start_date AND @end_date
        ORDER BY snapshot_date, severity
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date.isoformat()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date.isoformat()),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe(
        create_bqstorage_client=False
    )
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["gaps_open"] = pd.to_numeric(df["gaps_open"], errors="coerce").fillna(0)
    return df


@st.cache_data(show_spinner=False, ttl=900)
def load_rule_closures(
    project_id: str,
    dataset: str,
    table: str,
) -> pd.DataFrame:
    client = build_client(project_id)
    query = f"""
        WITH latest_snapshot AS (
            SELECT MAX(snapshot_date) AS snapshot_date
            FROM `{project_id}.{dataset}.{table}`
        )
        SELECT
            rule_bucket.rule_id,
            rule_bucket.rule_name,
            rule_bucket.gaps_closed,
            latest.snapshot_date
        FROM `{project_id}.{dataset}.{table}` progress
        JOIN latest_snapshot latest
            ON progress.snapshot_date = latest.snapshot_date
        CROSS JOIN UNNEST(progress.gaps_closed_by_rule) AS rule_bucket
        ORDER BY rule_bucket.gaps_closed DESC, rule_bucket.rule_name
    """
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["gaps_closed"] = pd.to_numeric(df["gaps_closed"], errors="coerce").fillna(0)
    return df


def coverage_line_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("snapshot_date:T", title="Snapshot Date"),
            y=alt.Y("coverage_percentage:Q", title="Coverage (%)"),
            tooltip=[
                alt.Tooltip("snapshot_date:T", title="Date"),
                alt.Tooltip("coverage_percentage:Q", title="Coverage", format=".2f"),
                alt.Tooltip("total_gaps_open:Q", title="Gaps Open"),
                alt.Tooltip("total_gaps_closed:Q", title="Gaps Closed"),
            ],
        )
        .properties(height=320)
    )


def severity_heatmap(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("snapshot_date:T", title="Snapshot Date"),
            y=alt.Y("severity:N", title="Severity"),
            color=alt.Color(
                "gaps_open:Q", title="Open Gaps", scale=alt.Scale(scheme="reds")
            ),
            tooltip=[
                alt.Tooltip("snapshot_date:T", title="Date"),
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("gaps_open:Q", title="Open Gaps"),
            ],
        )
        .properties(height=320)
    )


def rule_closure_bar(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("gaps_closed:Q", title="Gaps Closed"),
            y=alt.Y("rule_name:N", title="Rule", sort="-x"),
            tooltip=[
                alt.Tooltip("rule_name:N", title="Rule"),
                alt.Tooltip("gaps_closed:Q", title="Gaps Closed"),
            ],
        )
        .properties(height=320)
    )


def render_summary_metrics(df: pd.DataFrame) -> None:
    latest = df.sort_values("snapshot_date").iloc[-1]
    previous = df.sort_values("snapshot_date").iloc[-2] if len(df) > 1 else None

    def format_delta(current: float, prev: float | None) -> str | None:
        if prev is None or pd.isna(prev):
            return None
        delta = current - prev
        return f"{delta:+.2f}"

    coverage_delta = format_delta(
        latest["coverage_percentage"],
        previous["coverage_percentage"] if previous is not None else None,
    )
    gaps_delta = format_delta(
        latest["total_gaps_open"],
        previous["total_gaps_open"] if previous is not None else None,
    )

    cols = st.columns(3)
    cols[0].metric(
        label="Coverage (%)",
        value=(
            f"{latest['coverage_percentage']:.2f}"
            if pd.notna(latest["coverage_percentage"])
            else "n/a"
        ),
        delta=coverage_delta,
    )
    cols[1].metric(
        label="Open Gaps",
        value=safe_int(latest.get("total_gaps_open")),
        delta=gaps_delta,
    )
    cols[2].metric(
        label="Confidence",
        value=(
            f"{latest['mean_confidence_score']:.2f}"
            if pd.notna(latest["mean_confidence_score"])
            else "n/a"
        ),
    )


def main() -> None:
    configure_page()

    sidebar = st.sidebar
    sidebar.header("Configuration")
    project_id = sidebar.text_input("Project ID", value=DEFAULT_PROJECT)
    dataset = sidebar.text_input("Dataset", value=DEFAULT_DATASET)
    table = sidebar.text_input("Progress table", value=DEFAULT_TABLE)

    default_start = dt.date.today() - dt.timedelta(days=29)
    default_end = dt.date.today()
    date_inputs = sidebar.date_input(
        "Snapshot range",
        value=(default_start, default_end),
        help="Select the date window for trend visualisations.",
    )
    start_date, end_date = _date_range_to_tuple(date_inputs)

    refresh = sidebar.button("Refresh data", use_container_width=True)
    if refresh:
        load_progress_timeseries.clear()
        load_severity_heatmap.clear()
        load_rule_closures.clear()

    try:
        progress_df = load_progress_timeseries(
            project_id, dataset, table, start_date, end_date
        )
    except gcloud_exceptions.GoogleAPIError as error:
        st.error(
            "Failed to load progress snapshots. Verify project credentials and dataset permissions."
        )
        st.exception(error)
        return

    if progress_df.empty:
        st.info("No progress snapshots were found for the selected window.")
        return

    render_summary_metrics(progress_df)

    left, right = st.columns((2, 1))
    with left:
        st.subheader("Coverage Trend")
        st.altair_chart(coverage_line_chart(progress_df), use_container_width=True)
    with right:
        st.subheader("Latest Snapshot")
        latest_row = progress_df.sort_values("snapshot_date").iloc[-1]
        st.write(
            pd.DataFrame(
                {
                    "Metric": [
                        "Snapshot Date",
                        "Chunks Processed",
                        "Chunks Documented",
                        "Gaps Detected",
                        "Gaps Closed",
                    ],
                    "Value": [
                        latest_row["snapshot_date"].date(),
                        safe_int(latest_row.get("total_chunks_processed")),
                        safe_int(latest_row.get("total_chunks_documented")),
                        safe_int(latest_row.get("total_gaps_detected")),
                        safe_int(latest_row.get("total_gaps_closed")),
                    ],
                }
            ).set_index("Metric")
        )

    severity_df = load_severity_heatmap(
        project_id, dataset, table, start_date, end_date
    )
    st.subheader("Gap Severity Heat Map")
    if severity_df.empty:
        st.info("No severity distribution data available for the selected window.")
    else:
        st.altair_chart(severity_heatmap(severity_df), use_container_width=True)

    rule_df = load_rule_closures(project_id, dataset, table)
    st.subheader("Gaps Closed per Rule (Latest Snapshot)")
    if rule_df.empty:
        st.info("No gap closure activity recorded in the latest snapshot.")
    else:
        st.altair_chart(rule_closure_bar(rule_df), use_container_width=True)
        st.caption(f"Latest snapshot: {rule_df['snapshot_date'].iloc[0].date()}")


if __name__ == "__main__":
    main()
