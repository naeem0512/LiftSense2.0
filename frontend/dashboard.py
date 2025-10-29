"""Dash application for visualising LiftSense training and session outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TRAINING_METRICS_PATH = RESULTS_DIR / "training_metrics.json"
SESSION_PATH = RESULTS_DIR / "latest_session.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open() as json_file:
        return json.load(json_file)


def _build_accuracy_figure(history: Dict[str, List[float]]) -> go.Figure:
    figure = go.Figure()
    if history:
        if history.get("accuracy"):
            figure.add_trace(go.Scatter(
                y=history["accuracy"],
                mode="lines+markers",
                name="Train Accuracy",
                line=dict(color="#2E86AB")
            ))
        if history.get("val_accuracy"):
            figure.add_trace(go.Scatter(
                y=history["val_accuracy"],
                mode="lines+markers",
                name="Validation Accuracy",
                line=dict(color="#F18F01")
            ))

    figure.update_layout(
        title="Accuracy by Epoch",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        template="plotly_white",
        height=320,
    )
    return figure


def _build_loss_figure(history: Dict[str, List[float]]) -> go.Figure:
    figure = go.Figure()
    if history:
        if history.get("loss"):
            figure.add_trace(go.Scatter(
                y=history["loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#6C5B7B")
            ))
        if history.get("val_loss"):
            figure.add_trace(go.Scatter(
                y=history["val_loss"],
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color="#C06C84")
            ))

    figure.update_layout(
        title="Loss by Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        height=320,
    )
    return figure


def _session_dataframe(session_data: List[Dict[str, Any]]) -> pd.DataFrame:
    if not session_data:
        return pd.DataFrame(columns=["set", "fatigue", "confidence", "avg_speed", "avg_hr", "velocity_loss"])

    frame = pd.DataFrame(session_data)
    display_columns = {
        "set": "Set",
        "fatigue": "Fatigue",
        "confidence": "Confidence",
        "avg_speed": "Avg Speed (m/s)",
        "avg_hr": "Avg HR (bpm)",
        "velocity_loss": "Velocity Loss (%)",
    }

    # Ensure numeric columns are rounded for readability
    for column in ("avg_speed", "avg_hr", "velocity_loss"):
        if column in frame:
            frame[column] = frame[column].astype(float).round(2)

    return frame.rename(columns=display_columns)


def _session_summary_components(payload: Dict[str, Any]) -> List[html.Div]:
    metrics = payload.get("performance_metrics", {})
    cards: List[html.Div] = []

    summary_items = [
        ("Fatigue Trend", " ".join(metrics.get("fatigue_trend", [])) or "—"),
        ("Avg Velocity Loss", f"{pd.Series(metrics.get('velocity_loss', [])).mean():.2f}%" if metrics.get("velocity_loss") else "—"),
        ("Peak HR", max(metrics.get("hr_response", [0])) if metrics.get("hr_response") else "—"),
    ]

    for title, value in summary_items:
        cards.append(
            html.Div(
                className="metric-card",
                children=[
                    html.H4(title),
                    html.P(value if value != 0 else "—")
                ],
            )
        )

    return cards


app = Dash(__name__, title="LiftSense Dashboard")
server = app.server


app.layout = html.Div(
    className="container",
    children=[
        html.H1("LiftSense Monitoring Dashboard"),
        html.P("Track training convergence and recent session analytics from the LiftSense pipeline."),
        html.Div(
            className="graphs",
            children=[
                dcc.Graph(id="accuracy-graph", figure=_build_accuracy_figure({})),
                dcc.Graph(id="loss-graph", figure=_build_loss_figure({})),
            ],
        ),
        html.Div(id="session-summary", className="summary"),
        dash_table.DataTable(
            id="session-table",
            columns=[
                {"name": "Set", "id": "Set"},
                {"name": "Fatigue", "id": "Fatigue"},
                {"name": "Confidence", "id": "Confidence"},
                {"name": "Avg Speed (m/s)", "id": "Avg Speed (m/s)"},
                {"name": "Avg HR (bpm)", "id": "Avg HR (bpm)"},
                {"name": "Velocity Loss (%)", "id": "Velocity Loss (%)"},
            ],
            data=[],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#2E86AB", "color": "white", "fontWeight": "bold"},
            style_cell={"padding": "0.5rem"},
            page_size=10,
        ),
        dcc.Interval(id="refresh-interval", interval=5_000, n_intervals=0),
    ],
)


@app.callback(
    Output("accuracy-graph", "figure"),
    Output("loss-graph", "figure"),
    Output("session-table", "data"),
    Output("session-summary", "children"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_dashboard(_: int):
    metrics_payload = _load_json(TRAINING_METRICS_PATH)
    history = metrics_payload.get("history", {})

    session_payload = _load_json(SESSION_PATH)
    session_frame = _session_dataframe(session_payload.get("session_data", []))

    accuracy_fig = _build_accuracy_figure(history)
    loss_fig = _build_loss_figure(history)

    summary_components = _session_summary_components(session_payload)

    return accuracy_fig, loss_fig, session_frame.to_dict("records"), summary_components


if __name__ == "__main__":
    app.run_server(debug=True)
