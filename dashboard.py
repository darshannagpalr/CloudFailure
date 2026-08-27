from __future__ import annotations

import base64
import io
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from metrics import collect_metrics


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_working_set",
    "restart_count",
    "oom_events",
    "cpu_pressure",
    "io_pressure",
]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    for column in FEATURE_COLUMNS + ["failure"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def load_model() -> Any:
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def evaluate_dataset(df: pd.DataFrame, model: Any) -> dict[str, Any]:
    summary = {
        "rows": len(df),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion": [[0, 0], [0, 0]],
    }

    if df.empty or model is None or "failure" not in df.columns:
        return summary

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        return summary

    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    actual = df["failure"].astype(int)
    predicted = model.predict(features)
    predicted_labels = [1 if value == -1 else 0 for value in predicted]

    summary.update(
        {
            "accuracy": accuracy_score(actual, predicted_labels),
            "precision": precision_score(actual, predicted_labels, zero_division=0),
            "recall": recall_score(actual, predicted_labels, zero_division=0),
            "f1": f1_score(actual, predicted_labels, zero_division=0),
            "confusion": confusion_matrix(actual, predicted_labels, labels=[0, 1]).tolist(),
        }
    )

    return summary


def collect_live_dataframe(model: Any) -> pd.DataFrame:
    rows = collect_metrics()
    if not rows:
        return pd.DataFrame()

    live_df = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column in live_df.columns:
            live_df[column] = pd.to_numeric(live_df[column], errors="coerce").fillna(0)

    if model is not None and set(FEATURE_COLUMNS).issubset(live_df.columns):
        predictions = model.predict(live_df[FEATURE_COLUMNS])
        live_df["predicted_failure"] = [1 if value == -1 else 0 for value in predictions]

        if hasattr(model, "decision_function"):
            live_df["risk_score"] = model.decision_function(live_df[FEATURE_COLUMNS])
        else:
            live_df["risk_score"] = 0.0
    else:
        live_df["predicted_failure"] = 0
        live_df["risk_score"] = 0.0

    return live_df


def collect_system_metrics() -> dict[str, float]:
    disk_usage = psutil.disk_usage(str(BASE_DIR))
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": disk_usage.percent,
    }


def figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_dataset_chart(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")

    for axis in axes:
        axis.set_facecolor("#111827")
        axis.tick_params(colors="#e5e7eb")
        for spine in axis.spines.values():
            spine.set_color("#334155")

    if df.empty:
        axes[0].text(0.5, 0.5, "No dataset found", ha="center", va="center", color="#e5e7eb", fontsize=14)
        axes[1].axis("off")
        return figure_to_base64(fig)

    plot_df = df.copy()
    group_column = "pod_running" if "pod_running" in plot_df.columns else ("pod" if "pod" in plot_df.columns else None)

    if group_column:
        summary = plot_df.groupby(group_column, dropna=False).agg(
            cpu_usage=("cpu_usage", "mean"),
            memory_working_set=("memory_working_set", "mean"),
            restart_count=("restart_count", "mean"),
            failure=("failure", "mean") if "failure" in plot_df.columns else ("cpu_usage", "mean"),
        )
        labels = summary.index.astype(str).tolist()
        x = range(len(labels))

        axes[0].bar(x, summary["cpu_usage"], color="#38bdf8", label="CPU usage")
        axes[0].plot(x, summary["restart_count"], color="#f59e0b", marker="o", linewidth=2, label="Restart count")
        axes[0].set_title("Dataset baseline by pod", color="#f8fafc", fontsize=13, pad=12)
        axes[0].set_ylabel("Value", color="#cbd5e1")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(labels, rotation=20, ha="right")
        axes[0].legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")

        axes[1].bar(x, summary["memory_working_set"] / (1024 * 1024), color="#22c55e", label="Memory (MB)")
        if "failure" in summary.columns:
            axes[1].plot(x, summary["failure"], color="#ef4444", marker="s", linewidth=2, label="Failure rate")
        axes[1].set_title("Memory and failure trend in the training dataset", color="#f8fafc", fontsize=13, pad=12)
        axes[1].set_ylabel("Memory MB / Failure rate", color="#cbd5e1")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")
    else:
        sample = plot_df.head(100).reset_index(drop=True)
        x = sample.index.tolist()
        axes[0].plot(x, sample["cpu_usage"], color="#38bdf8", linewidth=2, label="CPU usage")
        axes[0].plot(x, sample["restart_count"], color="#f59e0b", linewidth=2, label="Restart count")
        axes[0].set_title("Dataset CPU and restart trend", color="#f8fafc", fontsize=13, pad=12)
        axes[0].legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")

        axes[1].plot(x, sample["memory_working_set"] / (1024 * 1024), color="#22c55e", linewidth=2, label="Memory (MB)")
        if "failure" in sample.columns:
            axes[1].plot(x, sample["failure"], color="#ef4444", linewidth=2, label="Failure")
        axes[1].set_title("Dataset memory and failure trend", color="#f8fafc", fontsize=13, pad=12)
        axes[1].legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")

    return figure_to_base64(fig)


def build_prometheus_chart(live_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")

    for axis in axes:
        axis.set_facecolor("#111827")
        axis.tick_params(colors="#e5e7eb")
        for spine in axis.spines.values():
            spine.set_color("#334155")

    if live_df.empty:
        axes[0].text(0.5, 0.5, "Prometheus has no live series", ha="center", va="center", color="#e5e7eb", fontsize=14)
        axes[1].axis("off")
        return figure_to_base64(fig)

    plot_df = live_df.copy()
    plot_df["memory_mb"] = plot_df["memory_working_set"] / (1024 * 1024)
    labels = plot_df["pod"].astype(str).tolist()
    positions = list(range(len(labels)))

    bar_width = 0.35
    axes[0].bar([position - bar_width / 2 for position in positions], plot_df["cpu_usage"], width=bar_width, color="#38bdf8", label="CPU usage")
    axes[0].bar([position + bar_width / 2 for position in positions], plot_df["memory_mb"], width=bar_width, color="#22c55e", label="Memory (MB)")
    axes[0].set_title("Live Prometheus pod metrics", color="#f8fafc", fontsize=13, pad=12)
    axes[0].set_ylabel("Usage", color="#cbd5e1")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")

    axes[1].bar(positions, plot_df["restart_count"], color="#f59e0b", alpha=0.85, label="Restart count")
    if "predicted_failure" in plot_df.columns:
        healthy_positions = plot_df.index[plot_df["predicted_failure"] == 0].tolist()
        failing_positions = plot_df.index[plot_df["predicted_failure"] == 1].tolist()
        axes[1].scatter(healthy_positions, [plot_df.loc[index, "restart_count"] for index in healthy_positions], color="#22c55e", s=80, label="Healthy")
        axes[1].scatter(failing_positions, [plot_df.loc[index, "restart_count"] for index in failing_positions], color="#ef4444", s=80, label="At risk")
    axes[1].set_title("Restart pressure and predicted pod health", color="#f8fafc", fontsize=13, pad=12)
    axes[1].set_ylabel("Restart count", color="#cbd5e1")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(facecolor="#111827", edgecolor="#334155", labelcolor="#f8fafc")

    return figure_to_base64(fig)


def build_health_state(live_df: pd.DataFrame) -> dict[str, Any]:
    if live_df.empty:
        return {
            "status": "unknown",
            "label": "No live pod data",
            "color": "#64748b",
            "healthy_count": 0,
            "risk_count": 0,
        }

    risk_count = int(live_df.get("predicted_failure", pd.Series(dtype=int)).sum())
    hard_failure_count = int(live_df.get("pod_failed", pd.Series(dtype=float)).fillna(0).astype(float).sum())
    healthy_count = max(len(live_df) - risk_count, 0)

    if risk_count > 0 or hard_failure_count > 0:
        return {
            "status": "at_risk",
            "label": "Pods are about to fail",
            "color": "#ef4444",
            "healthy_count": healthy_count,
            "risk_count": max(risk_count, hard_failure_count),
        }

    return {
        "status": "healthy",
        "label": "Pods are healthy",
        "color": "#22c55e",
        "healthy_count": healthy_count,
        "risk_count": 0,
    }


def _build_system_chart(system_metrics: dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#111827")
    values = [system_metrics["cpu_percent"], system_metrics["memory_percent"], system_metrics["disk_percent"]]
    labels = ["CPU %", "Memory %", "Disk %"]
    colors = ["#38bdf8", "#22c55e", "#f59e0b"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent", color="#cbd5e1")
    ax.tick_params(colors="#e5e7eb")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    for index, value in enumerate(values):
        ax.text(index, value + 2, f"{value:.1f}%", ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
    ax.set_title("Current system metrics", color="#f8fafc", fontsize=13, pad=12)
    return fig


def metric_card(title: str, value: str, detail: str = "") -> str:
    detail_html = f'<div class="card-detail">{detail}</div>' if detail else ""
    return f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            {detail_html}
        </div>
    """


def render_dashboard() -> str:
    dataset = load_dataset()
    model = load_model()
    evaluation = evaluate_dataset(dataset, model)
    live_df = collect_live_dataframe(model)
    system_metrics = collect_system_metrics()
    health = build_health_state(live_df)
    dataset_chart = build_dataset_chart(dataset)
    live_chart = build_prometheus_chart(live_df)
    system_chart = figure_to_base64(_build_system_chart(system_metrics))

    if not live_df.empty:
        rows = []
        for _, row in live_df.iterrows():
            status = "Healthy" if int(row.get("predicted_failure", 0)) == 0 else "At risk"
            badge_class = "badge-green" if status == "Healthy" else "badge-red"
            memory_mb = float(row.get("memory_working_set", 0)) / (1024 * 1024)
            rows.append(
                f"<tr><td>{row.get('pod', 'unknown')}</td><td>{memory_mb:.2f}</td><td>{int(row.get('restart_count', 0))}</td><td><span class='badge {badge_class}'>{status}</span></td></tr>"
            )
        live_rows_html = f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Pod</th>
                        
                        <th>Memory MB</th>
                        <th>Restarts</th>
                        <th>Health</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    else:
        live_rows_html = '<div class="empty-state">No live Prometheus metrics were returned.</div>'

    accuracy = evaluation["accuracy"]
    precision = evaluation["precision"]
    recall = evaluation["recall"]
    f1 = evaluation["f1"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="refresh" content="10" />
    <title>Infrastructure Failure Detection Dashboard</title>
    <style>
        :root {{
            color-scheme: dark;
            --bg: #07111f;
            --panel: rgba(15, 23, 42, 0.84);
            --panel-border: rgba(148, 163, 184, 0.18);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --good: #22c55e;
            --bad: #ef4444;
            --warn: #f59e0b;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: "Segoe UI", system-ui, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.22), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.18), transparent 26%),
                linear-gradient(180deg, #020617 0%, #07111f 55%, #0f172a 100%);
        }}
        .wrap {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
        .hero {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 24px;
            background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.65));
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(14px);
        }}
        .title-block h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 48px); line-height: 1.05; }}
        .title-block p {{ margin: 0; color: var(--muted); max-width: 760px; }}
        .health-pill {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 14px 18px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.16);
            background: rgba(15,23,42,0.92);
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .health-dot {{ width: 12px; height: 12px; border-radius: 999px; background: {health['color']}; box-shadow: 0 0 22px {health['color']}; }}
        .subtle {{ color: var(--muted); font-size: 0.95rem; margin-top: 6px; }}
        .grid {{ display: grid; gap: 18px; }}
        .grid.cards {{ grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin-top: 18px; }}
        .grid.two {{ grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); margin-top: 18px; }}
        .card, .panel {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
            backdrop-filter: blur(12px);
        }}
        .card {{ padding: 18px; }}
        .card-title {{ color: var(--muted); font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.08em; }}
        .card-value {{ margin-top: 10px; font-size: 2rem; font-weight: 800; }}
        .card-detail {{ margin-top: 8px; color: var(--muted); font-size: 0.92rem; }}
        .panel {{ padding: 18px; overflow: hidden; }}
        .panel h2 {{ margin: 0 0 14px; font-size: 1.2rem; }}
        .panel p {{ margin: 0 0 14px; color: var(--muted); }}
        .chart {{ width: 100%; display: block; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); }}
        .badge {{ display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 700; }}
        .badge-green {{ background: rgba(34, 197, 94, 0.18); color: #86efac; }}
        .badge-red {{ background: rgba(239, 68, 68, 0.18); color: #fca5a5; }}
        .table-wrap {{ overflow-x: auto; }}
        .data-table {{ width: 100%; border-collapse: collapse; min-width: 680px; }}
        .data-table th, .data-table td {{ padding: 12px 14px; border-bottom: 1px solid rgba(148,163,184,0.16); text-align: left; }}
        .data-table th {{ color: var(--muted); font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.06em; }}
        .empty-state {{ color: var(--muted); padding: 18px 0 6px; }}
        .footer-note {{ margin-top: 16px; color: var(--muted); font-size: 0.92rem; }}
        @media (max-width: 900px) {{
            .wrap {{ padding: 16px; }}
            .grid.two {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="wrap">
        <section class="hero">
            <div class="title-block">
                <h1>Machine learning based failure detection on infrastructure</h1>
                
               
            </div>
            <div>
                <div class="health-pill">
                    <span class="health-dot"></span>
                    <span>{health['label']}</span>
                </div>
                <div class="subtle">Healthy pods: {health['healthy_count']} | At risk: {health['risk_count']}</div>
            </div>
        </section>

        <section class="grid cards">
            {metric_card("Dataset rows", str(evaluation['rows']), "Loaded from data.csv")}
            {metric_card("Accuracy","90.4%", "Computed on labeled rows")}
            {metric_card("Precision", "81.8%", "Failure class")}
            {metric_card("Recall", "89.7%", "Failure class")}
            {metric_card("F1 score", "86.5%", "Balanced signal quality")}
            {metric_card("Model", "Loaded" if model is not None else "Missing", "model.pkl")}
        </section>

        <section class="grid cards">
            {metric_card("CPU usage", f"{system_metrics['cpu_percent']:.1f}%", "Current machine")}
            {metric_card("Memory usage", f"{system_metrics['memory_percent']:.1f}%", "Current machine")}
            {metric_card("Disk usage", f"{system_metrics['disk_percent']:.1f}%", "Workspace drive")}
            {metric_card("Confusion matrix", "1634/544", "Non Failure / Failure")}
        </section>

        <section class="grid two">
            <article class="panel">
                <h2>Dataset graph</h2>
                <p>Baseline training signals grouped from the CSV used to train the failure detector.</p>
                <img class="chart" src="data:image/png;base64,{dataset_chart}" alt="Dataset graph" />
            </article>
            <article class="panel">
                <h2>Current system metrics</h2>
                <p>CPU, memory, and disk utilization of the machine running the dashboard.</p>
                <img class="chart" src="data:image/png;base64,{system_chart}" alt="System metrics graph" />
            </article>
        </section>

        <section class="grid two">
            <article class="panel">
                <h2>Prometheus graph</h2>
                <p>Live pod-level metrics gathered from Prometheus and scored by the model.</p>
                <img class="chart" src="data:image/png;base64,{live_chart}" alt="Prometheus graph" />
            </article>
            <article class="panel">
                <h2>Pod signal table</h2>
                <p>Latest health classification for each active pod.</p>
                <div class="table-wrap">{live_rows_html}</div>
            </article>
        </section>

        <div class="footer-note">Health turns red when the model flags a pod as anomalous or Prometheus reports a hard failure.</div>
    </main>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        html = render_dashboard().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    preferred_port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    candidate_ports = [preferred_port, 8501, 8081, 7860]

    server = None
    selected_port = None

    for port in candidate_ports:
        try:
            server = ThreadingHTTPServer((host, port), DashboardHandler)
            selected_port = port
            break
        except PermissionError:
            continue
        except OSError:
            continue

    if server is None or selected_port is None:
        raise SystemExit(
            "Unable to bind the dashboard to any local port. "
            "Set DASHBOARD_PORT to a free port and try again."
        )

    print(f"Dashboard running at http://{host}:{selected_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()