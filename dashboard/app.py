"""
Live training dashboard — open in browser while training runs.
Polls CSV logs every 2 seconds and updates charts.

Run:  python dashboard/app.py
Then open http://127.0.0.1:8050
"""
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

LOGS_DIR = Path("training_logs")
ALGOS    = ["dqn", "ppo", "a2c"]
COLORS   = {"dqn": "#FF6B35", "ppo": "#4ECDC4", "a2c": "#A855F7"}

app = dash.Dash(__name__, title="Element Quest — Training Dashboard")

app.layout = html.Div(
    style={"backgroundColor": "#0F0F1A", "minHeight": "100vh", "fontFamily": "Arial"},
    children=[
        # Header
        html.Div(
            style={"background": "linear-gradient(90deg,#FF6B35,#A855F7)",
                   "padding": "18px 30px"},
            children=[
                html.H1("Element Quest — Live Training Dashboard",
                        style={"color": "white", "margin": 0, "fontSize": "24px"}),
                html.P("Comparing DQN · PPO · A2C",
                       style={"color": "rgba(255,255,255,0.7)", "margin": "4px 0 0"}),
            ]
        ),

        # Toolbar: status cards + reset button
        html.Div(
            style={"display": "flex", "alignItems": "flex-start",
                   "gap": "16px", "padding": "20px 30px"},
            children=[
                html.Div(id="status-cards",
                         style={"display": "flex", "gap": "16px", "flex": "1",
                                "flexWrap": "wrap"}),
                html.Div(
                    style={"display": "flex", "flexDirection": "column",
                           "gap": "8px", "minWidth": "160px"},
                    children=[
                        html.Button(
                            "Clear All Logs & Models",
                            id="reset-btn",
                            style={
                                "backgroundColor": "#5A1A1A",
                                "color": "#FF8080",
                                "border": "1px solid #FF4444",
                                "borderRadius": "6px",
                                "padding": "10px 16px",
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "fontSize": "13px",
                            }
                        ),
                        html.Div(id="reset-msg",
                                 style={"color": "#90EE90", "fontSize": "12px",
                                        "textAlign": "center"}),
                    ]
                ),
            ]
        ),

        # Charts
        dcc.Graph(id="reward-chart",
                  style={"margin": "0 20px"},
                  config={"displayModeBar": False}),

        dcc.Graph(id="length-chart",
                  style={"margin": "0 20px 20px"},
                  config={"displayModeBar": False}),

        dcc.Graph(id="comparison-chart",
                  style={"margin": "0 20px 30px"},
                  config={"displayModeBar": False}),

        dcc.Interval(id="interval", interval=2000, n_intervals=0),
    ]
)


# ------------------------------------------------------------------ #
def load_logs() -> dict[str, pd.DataFrame]:
    data = {}
    for algo in ALGOS:
        path = LOGS_DIR / f"{algo}_log.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    df['reward_smooth'] = df['reward'].rolling(20, min_periods=1).mean()
                    data[algo] = df
            except Exception:
                pass
    return data


def make_chart(fig_data, key, title, ylabel, row, col, data):
    for algo, df in data.items():
        if key in df.columns:
            smooth_key = f"{key}_smooth" if f"{key}_smooth" in df.columns else key
            fig_data.add_trace(
                go.Scatter(
                    x=df['episode'], y=df[smooth_key],
                    name=algo.upper(), line=dict(color=COLORS[algo], width=2),
                    mode='lines', showlegend=(row == 1)
                ),
                row=row, col=col
            )


@app.callback(
    Output("reward-chart", "figure"),
    Output("length-chart", "figure"),
    Output("comparison-chart", "figure"),
    Output("status-cards", "children"),
    Input("interval", "n_intervals"),
)
def update(_):
    data = load_logs()

    plot_bg    = "#1A1A2E"
    paper_bg   = "#0F0F1A"
    grid_color = "#2A2A4A"
    text_color = "#CCCCDD"

    base_layout = dict(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#3A3A5A"),
        margin=dict(l=60, r=30, t=50, b=40),
    )

    # -- Reward chart --
    fig1 = go.Figure(layout=go.Layout(title="Episode Reward (smoothed 20ep)", **base_layout))
    fig1.update_xaxes(gridcolor=grid_color, title="Episode")
    fig1.update_yaxes(gridcolor=grid_color, title="Reward")
    for algo, df in data.items():
        if 'reward' in df.columns:
            fig1.add_trace(go.Scatter(
                x=df['episode'], y=df['reward_smooth'],
                name=algo.upper(),
                line=dict(color=COLORS[algo], width=2),
                mode='lines',
            ))

    # -- Episode length chart --
    fig2 = go.Figure(layout=go.Layout(title="Episode Length", **base_layout))
    fig2.update_xaxes(gridcolor=grid_color, title="Episode")
    fig2.update_yaxes(gridcolor=grid_color, title="Steps")
    for algo, df in data.items():
        if 'ep_length' in df.columns:
            smooth = df['ep_length'].rolling(20, min_periods=1).mean()
            fig2.add_trace(go.Scatter(
                x=df['episode'], y=smooth,
                name=algo.upper(),
                line=dict(color=COLORS[algo], width=2, dash='dot'),
                mode='lines',
            ))

    # -- Final comparison bar chart --
    fig3 = go.Figure(layout=go.Layout(
        title="Algorithm Comparison — Avg Reward (last 100 episodes)",
        **base_layout,
        barmode='group',
    ))
    fig3.update_xaxes(title="Algorithm")
    fig3.update_yaxes(gridcolor=grid_color, title="Avg Reward")
    names, avgs, colors_list = [], [], []
    for algo, df in data.items():
        if 'reward' in df.columns and len(df) >= 10:
            avg = df['reward'].tail(100).mean()
            names.append(algo.upper())
            avgs.append(round(avg, 2))
            colors_list.append(COLORS[algo])
    if names:
        fig3.add_trace(go.Bar(x=names, y=avgs, marker_color=colors_list,
                              showlegend=False))

    # -- Status cards --
    cards = []
    for algo in ALGOS:
        df = data.get(algo)
        if df is not None and not df.empty:
            ep   = int(df['episode'].iloc[-1])
            ts   = int(df['timestep'].iloc[-1])
            rew  = round(float(df['reward_smooth'].iloc[-1]), 1)
            best = round(float(df['reward'].max()), 1)
            status = "Training" if ts < _algo_target(algo) else "Complete"
            bg = "#1E2A1E" if status == "Complete" else "#1A1A2E"
        else:
            ep, ts, rew, best, status = 0, 0, 0, 0, "Waiting..."
            bg = "#1A1A2E"

        cards.append(html.Div(
            style={"background": bg, "border": f"1px solid {COLORS[algo]}",
                   "borderRadius": "8px", "padding": "14px 20px",
                   "flex": "1", "minWidth": "200px"},
            children=[
                html.H3(algo.upper(), style={"color": COLORS[algo], "margin": "0 0 8px"}),
                html.P(f"Episodes: {ep}", style={"margin": "2px", "color": text_color}),
                html.P(f"Timesteps: {ts:,}", style={"margin": "2px", "color": text_color}),
                html.P(f"Last reward: {rew}", style={"margin": "2px", "color": text_color}),
                html.P(f"Best reward: {best}", style={"margin": "2px", "color": "#90EE90"}),
                html.P(status, style={"margin": "6px 0 0", "fontWeight": "bold",
                                       "color": "#90EE90" if status == "Complete" else "#FFD700"}),
            ]
        ))

    return fig1, fig2, fig3, cards


def _algo_target(algo):
    return {"dqn": 300_000, "ppo": 500_000, "a2c": 400_000}.get(algo, 300_000)


# ------------------------------------------------------------------ #
# Reset callback
# ------------------------------------------------------------------ #
@app.callback(
    Output("reset-msg", "children"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_logs(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    deleted = 0
    models_dir = Path("rl/models")

    for f in LOGS_DIR.glob("*.csv"):
        f.unlink()
        deleted += 1
    for f in models_dir.glob("*.zip"):
        f.unlink()
        deleted += 1

    if deleted:
        return f"Cleared {deleted} file(s) ✓"
    return "Already empty"


if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
