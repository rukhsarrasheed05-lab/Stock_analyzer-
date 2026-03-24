"""
Stock Fund Analyzer — LSTM-based Investment Signal Generator
A complete Streamlit web application.

Run with:
    streamlit run stock_analyzer.py

Required packages:
    pip install streamlit yfinance tensorflow scikit-learn pandas numpy plotly
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report
)
from datetime import datetime, timedelta
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Fund Analyzer — LSTM",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TICKERS = {
    # Index ETFs
    "SPY — S&P 500 ETF":         "SPY",
    "QQQ — Nasdaq-100 ETF":      "QQQ",
    "DIA — Dow Jones ETF":       "DIA",
    # Large-cap stocks
    "AAPL — Apple":              "AAPL",
    "MSFT — Microsoft":          "MSFT",
    "TSLA — Tesla":              "TSLA",
    "GOOGL — Alphabet":          "GOOGL",
    "AMZN — Amazon":             "AMZN",
    "NVDA — NVIDIA":             "NVDA",
    # Commodity ETF
    "GLD — Gold ETF":            "GLD",
    # Bond ETF (bonus)
    "TLT — 20yr Treasury ETF":   "TLT",
}

PERIOD_MAP = {
    "1 Year":  365,
    "2 Years": 730,
    "5 Years": 1825,
}

# ─────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

ticker_label = st.sidebar.selectbox("Ticker", list(TICKERS.keys()), index=0)
ticker = TICKERS[ticker_label]

period_label = st.sidebar.selectbox("Historical Period", list(PERIOD_MAP.keys()), index=1)
period_days  = PERIOD_MAP[period_label]

lookback_window = st.sidebar.slider(
    "Lookback Window (days)", min_value=10, max_value=90, value=30, step=5,
    help="Number of past trading days fed into the LSTM as one input sequence."
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)", min_value=1, max_value=30, value=5, step=1,
    help="How many trading days ahead the model tries to predict."
)

buy_threshold_pct = st.sidebar.slider(
    "BUY Threshold (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.5,
    help="A day is labelled BUY (1) if the close N days later is ≥ this % higher."
)

st.sidebar.markdown("---")
st.sidebar.subheader("LSTM Architecture")

lstm_units_1 = st.sidebar.select_slider(
    "LSTM Layer 1 Units", options=[32, 64, 128, 256], value=128
)
lstm_units_2 = st.sidebar.select_slider(
    "LSTM Layer 2 Units", options=[16, 32, 64, 128], value=64
)
dense_units = st.sidebar.select_slider(
    "Dense Layer Units", options=[16, 32, 64], value=32
)
dropout_rate = st.sidebar.slider(
    "Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.05
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶  Run Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────
st.title("📈 Stock Fund Analyzer — LSTM Investment Signal")
st.caption(
    "Uses a stacked LSTM Recurrent Neural Network to generate BUY / HOLD / SELL signals "
    "from historical OHLCV data. Configure parameters in the sidebar, then press **Run Analysis**."
)

if not run_btn:
    st.info("👈  Select your parameters in the sidebar and press **Run Analysis** to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
st.header("1 · Data Collection")

end_date   = datetime.today()
start_date = end_date - timedelta(days=period_days + 100)  # +100 for feature warm-up

with st.spinner(f"Downloading {ticker} data from Yahoo Finance…"):
    raw = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

if raw.empty:
    st.error("❌ No data returned. Check your internet connection or try a different ticker.")
    st.stop()

# Flatten MultiIndex columns if present (yfinance ≥ 0.2.x may return them)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

# Strip timezone from DatetimeIndex
raw.index = raw.index.tz_localize(None) if raw.index.tzinfo else raw.index

df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.dropna(inplace=True)

st.success(f"✅  Downloaded **{len(df):,} trading days** for **{ticker}** "
           f"({df.index[0].date()} → {df.index[-1].date()})")

# ── Candlestick chart with volume and MAs ──────────────────────────────────
fig_candle = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.75, 0.25], vertical_spacing=0.03,
    subplot_titles=(f"{ticker} Price", "Volume"),
)

fig_candle.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ),
    row=1, col=1,
)

# Moving averages
for period, color in [(20, "#f6c90e"), (50, "#fc6955")]:
    ma = df["Close"].rolling(period).mean()
    fig_candle.add_trace(
        go.Scatter(x=df.index, y=ma, name=f"SMA{period}", line=dict(color=color, width=1.2)),
        row=1, col=1,
    )

fig_candle.add_trace(
    go.Bar(x=df.index, y=df["Volume"], name="Volume",
           marker_color="rgba(100,181,246,0.5)"),
    row=2, col=1,
)

fig_candle.update_layout(
    xaxis_rangeslider_visible=False,
    height=500, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_candle, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
st.header("2 · Feature Engineering")

feat = pd.DataFrame(index=df.index)

# Candlestick body features
feat["body_size"]   = (df["Close"] - df["Open"]).abs()
feat["body_ratio"]  = feat["body_size"] / (df["High"] - df["Low"] + 1e-9)
feat["upper_shadow"]= (df["High"]  - df[["Open","Close"]].max(axis=1)) / (df["High"] - df["Low"] + 1e-9)
feat["lower_shadow"]= (df[["Open","Close"]].min(axis=1) - df["Low"])   / (df["High"] - df["Low"] + 1e-9)
feat["direction"]   = np.sign(df["Close"] - df["Open"])

# Return features
feat["ret_1d"]  = df["Close"].pct_change(1)
feat["ret_5d"]  = df["Close"].pct_change(5)
feat["ret_10d"] = df["Close"].pct_change(10)

# Trend features
sma20 = df["Close"].rolling(20).mean()
sma50 = df["Close"].rolling(50).mean()
feat["price_vs_sma20"] = (df["Close"] - sma20) / (sma20 + 1e-9)
feat["price_vs_sma50"] = (df["Close"] - sma50) / (sma50 + 1e-9)

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
feat["macd"]        = ema12 - ema26
feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
feat["macd_hist"]   = feat["macd"] - feat["macd_signal"]

# Volatility — 20-day rolling std of returns
feat["volatility_20d"] = feat["ret_1d"].rolling(20).std()

# Volume relative to 20-day average
vol_ma20 = df["Volume"].rolling(20).mean()
feat["volume_ratio"] = df["Volume"] / (vol_ma20 + 1e-9)

# RSI (normalised to [0,1])
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / (loss + 1e-9)
feat["rsi"] = (100 - 100 / (1 + rs)) / 100   # normalised

# Bollinger Band position
bb_std = df["Close"].rolling(20).std()
bb_upper = sma20 + 2 * bb_std
bb_lower = sma20 - 2 * bb_std
feat["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

# ATR (Average True Range, normalised)
tr = pd.concat([
    (df["High"] - df["Low"]),
    (df["High"] - df["Close"].shift()).abs(),
    (df["Low"]  - df["Close"].shift()).abs(),
], axis=1).max(axis=1)
feat["atr_norm"] = tr.rolling(14).mean() / (df["Close"] + 1e-9)

FEATURE_COLS = feat.columns.tolist()

with st.expander(f"📊 Feature Table ({len(FEATURE_COLS)} features) — click to expand"):
    st.dataframe(
        feat.tail(50).style.format("{:.4f}"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
st.header("3 · Data Preparation")

# Binary labels: 1 = BUY if close N days later is ≥ threshold% higher
future_close = df["Close"].shift(-forecast_horizon)
threshold    = buy_threshold_pct / 100.0
labels = ((future_close - df["Close"]) / df["Close"] >= threshold).astype(int)

# Merge features and labels; drop NaNs introduced by rolling windows
data = feat.join(labels.rename("label")).dropna()

X_raw = data[FEATURE_COLS].values.astype(np.float32)
y_raw = data["label"].values.astype(np.int32)
dates = data.index

# ── Sliding window construction ──────────────────────────────────────────
W = lookback_window

def make_windows(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ── Chronological split indices (80 / 10 / 10) ─────────────────────────
n       = len(X_raw)
n_train = int(n * 0.80)
n_val   = int(n * 0.10)

X_train_raw, y_train_raw = X_raw[:n_train],          y_raw[:n_train]
X_val_raw,   y_val_raw   = X_raw[n_train:n_train+n_val], y_raw[n_train:n_train+n_val]
X_test_raw,  y_test_raw  = X_raw[n_train+n_val:],    y_raw[n_train+n_val:]

# ── Fit scaler on training data ONLY ────────────────────────────────────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# ── Build windowed sequences ─────────────────────────────────────────────
X_train, y_train = make_windows(X_train_scaled, y_train_raw, W)
X_val,   y_val   = make_windows(X_val_scaled,   y_val_raw,   W)
X_test,  y_test  = make_windows(X_test_scaled,  y_test_raw,  W)

# Dates for test set (for plotting)
test_dates = dates[n_train + n_val + W: n_train + n_val + W + len(y_test)]

col1, col2, col3 = st.columns(3)
col1.metric("Train samples",      f"{len(X_train):,}")
col2.metric("Validation samples", f"{len(X_val):,}")
col3.metric("Test samples",       f"{len(X_test):,}")

st.markdown(f"""
| Attribute | Value |
|---|---|
| Total rows after NaN drop | {len(data):,} |
| Lookback window **W** | {W} days |
| Forecast horizon | {forecast_horizon} days |
| BUY threshold | {buy_threshold_pct}% |
| Class balance (train BUY%) | {y_train.mean()*100:.1f}% |
| Features | {len(FEATURE_COLS)} |
| Input shape | ({W}, {len(FEATURE_COLS)}) |
""")

if len(X_train) < 50:
    st.warning("⚠️ Very few training samples. Try increasing the historical period or reducing the lookback window.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
st.header("4 · LSTM Model Architecture")

n_features = len(FEATURE_COLS)

def build_model(units1, units2, dense, dropout, window, n_feat):
    inp = keras.Input(shape=(window, n_feat), name="input")

    x = layers.LSTM(units1, return_sequences=True, name="lstm_1")(inp)
    x = layers.Dropout(dropout, name="dropout_1")(x)

    x = layers.LSTM(units2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)

    x = layers.Dense(dense, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout / 2, name="dropout_3")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LSTM_StockAnalyzer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model(lstm_units_1, lstm_units_2, dense_units, dropout_rate, W, n_features)

# Capture model summary
buf = io.StringIO()
model.summary(print_fn=lambda s: buf.write(s + "\n"))
summary_str = buf.getvalue()

with st.expander("🧠 Model Architecture Summary — click to expand"):
    st.code(summary_str, language="text")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────
st.header("5 · Training")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
)

progress_placeholder = st.empty()
progress_placeholder.info("⏳ Training the LSTM… this may take 1–3 minutes.")

with st.spinner("Training LSTM model…"):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

progress_placeholder.success(
    f"✅ Training complete — {len(history.history['loss'])} epochs ran "
    f"(early stopping patience = 10)."
)

# ── Training loss curve ──────────────────────────────────────────────────
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(
    y=history.history["loss"], name="Train Loss",
    line=dict(color="#4fc3f7", width=2)
))
fig_loss.add_trace(go.Scatter(
    y=history.history["val_loss"], name="Val Loss",
    line=dict(color="#ef5350", width=2, dash="dot")
))
fig_loss.update_layout(
    title="Training vs Validation Loss",
    xaxis_title="Epoch", yaxis_title="Binary Cross-Entropy Loss",
    template="plotly_dark", height=350,
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_loss, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EVALUATION & SIGNAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
st.header("6 · Evaluation & Investment Signal")

y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    auc = float("nan")

report_str = classification_report(y_test, y_pred, target_names=["SELL/HOLD", "BUY"])

col_a, col_b = st.columns(2)
col_a.metric("Test Accuracy",  f"{acc:.2%}")
col_b.metric("ROC-AUC Score",  f"{auc:.4f}" if not np.isnan(auc) else "N/A")

with st.expander("📋 Full Classification Report"):
    st.code(report_str, language="text")

# ── BUY probability chart over test period ───────────────────────────────
if len(test_dates) == len(y_prob):
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=test_dates, y=y_prob,
        name="BUY Probability",
        fill="tozeroy",
        line=dict(color="#26a69a", width=1.5),
        fillcolor="rgba(38,166,154,0.15)",
    ))
    fig_prob.add_hline(y=0.65, line_dash="dash", line_color="#26a69a",  annotation_text="STRONG BUY 0.65")
    fig_prob.add_hline(y=0.52, line_dash="dot",  line_color="#a5d6a7",  annotation_text="BUY 0.52")
    fig_prob.add_hline(y=0.35, line_dash="dot",  line_color="#ffcc80",  annotation_text="SELL 0.35")
    fig_prob.add_hline(y=0.20, line_dash="dash", line_color="#ef5350",  annotation_text="STRONG SELL 0.20")
    fig_prob.update_layout(
        title="Model BUY Probability — Test Set",
        xaxis_title="Date", yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark", height=350,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Signals overlaid on price chart ────────────────────────────────
    test_close = df["Close"].reindex(test_dates)
    signals    = pd.Series(y_pred, index=test_dates)
    buy_idx    = test_dates[y_pred == 1]
    sell_idx   = test_dates[y_pred == 0]

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=test_dates, y=test_close.values,
        name="Close Price", line=dict(color="#90caf9", width=1.5),
    ))
    fig_sig.add_trace(go.Scatter(
        x=buy_idx,
        y=test_close.reindex(buy_idx).values,
        mode="markers",
        name="BUY Signal",
        marker=dict(symbol="triangle-up", size=8, color="#26a69a"),
    ))
    fig_sig.add_trace(go.Scatter(
        x=sell_idx,
        y=test_close.reindex(sell_idx).values,
        mode="markers",
        name="SELL/HOLD Signal",
        marker=dict(symbol="triangle-down", size=8, color="#ef5350"),
    ))
    fig_sig.update_layout(
        title="Buy / Sell Signals on Price — Test Period",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_sig, use_container_width=True)
else:
    st.info("Date alignment skipped (date count mismatch — check data size).")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL INVESTMENT DECISION
# ─────────────────────────────────────────────────────────────────────────────
st.header("🎯 Investment Decision")

# Prepare the most recent window of data
latest_raw   = X_raw[-W:]
latest_scaled = scaler.transform(latest_raw)
latest_seq   = latest_scaled.reshape(1, W, n_features)
final_prob   = float(model.predict(latest_seq, verbose=0)[0, 0])

# Map probability → signal
if final_prob >= 0.65:
    signal, color_hex, emoji = "STRONG BUY",  "#26a69a", "🟢"
elif final_prob >= 0.52:
    signal, color_hex, emoji = "BUY",          "#a5d6a7", "🟢"
elif final_prob >= 0.35:
    signal, color_hex, emoji = "HOLD / WAIT",  "#ffcc80", "🟡"
elif final_prob >= 0.20:
    signal, color_hex, emoji = "SELL",          "#ef9a9a", "🔴"
else:
    signal, color_hex, emoji = "STRONG SELL",  "#ef5350", "🔴"

st.markdown(
    f"""
    <div style="
        background: {color_hex}22;
        border: 2px solid {color_hex};
        border-radius: 12px;
        padding: 24px 32px;
        text-align: center;
        margin-bottom: 16px;
    ">
        <h1 style="color:{color_hex}; margin:0; font-size:2.8rem;">{emoji} {signal}</h1>
        <p style="color:#ccc; font-size:1.1rem; margin:8px 0 0;">
            Model BUY probability for <strong>{ticker}</strong>:&nbsp;
            <strong style="color:{color_hex};">{final_prob:.2%}</strong>
        </p>
        <p style="color:#aaa; font-size:0.85rem; margin:6px 0 0;">
            Forecast horizon: {forecast_horizon} trading days &nbsp;|&nbsp;
            Lookback: {W} days &nbsp;|&nbsp;
            BUY threshold: {buy_threshold_pct}%
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Written explanation ───────────────────────────────────────────────────
last_row  = data.iloc[-1]
rsi_val   = last_row["rsi"] * 100
macd_val  = last_row["macd"]
vol_ratio = last_row["volume_ratio"]
vs_sma20  = last_row["price_vs_sma20"] * 100
ret_5d    = last_row["ret_5d"] * 100

st.subheader("📝 Signal Explanation")
st.markdown(f"""
The LSTM model analysed the most recent **{W} trading days** of {ticker} and output a BUY
probability of **{final_prob:.2%}**, leading to the **{signal}** decision.

Key factors visible in the feature data:

| Indicator | Latest Value | Interpretation |
|---|---|---|
| RSI | {rsi_val:.1f} | {'Overbought (>70)' if rsi_val > 70 else 'Oversold (<30)' if rsi_val < 30 else 'Neutral'} |
| MACD | {macd_val:.4f} | {'Bullish (positive)' if macd_val > 0 else 'Bearish (negative)'} |
| Price vs SMA20 | {vs_sma20:+.2f}% | {'Above — upward momentum' if vs_sma20 > 0 else 'Below — downward pressure'} |
| 5-day return | {ret_5d:+.2f}% | {'Positive recent momentum' if ret_5d > 0 else 'Negative recent momentum'} |
| Volume ratio | {vol_ratio:.2f}× | {'Above average — stronger conviction' if vol_ratio > 1 else 'Below average — weaker conviction'} |

The model has learned temporal patterns across {W} sequential days; its output reflects
non-linear relationships across all {len(FEATURE_COLS)} features simultaneously.
""")

# ── Disclaimer ───────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Educational Disclaimer:** This tool is for learning purposes only. "
    "The investment signals generated by this model are NOT financial advice "
    "and should NOT be used to make real investment decisions. "
    "Past patterns in historical data do not guarantee future performance. "
    "Always consult a qualified financial advisor before investing."
)

# ─────────────────────────────────────────────────────────────────────────────
# TUNING GUIDE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.header("🔧 Tuning Guide")
st.markdown("""
| Parameter | Lower value | Higher value | Recommended starting point |
|---|---|---|---|
| Lookback Window | Faster, less context | Slower, more historical context | 20–40 days |
| Forecast Horizon | Predict near-term moves | Predict longer-term trends | 5 days |
| BUY Threshold | More BUY labels (class imbalance) | Fewer BUY labels (stricter) | 1.5% |
| LSTM Units 1 | Faster, may underfit | Slower, may overfit | 64–128 |
| LSTM Units 2 | Lighter decoder | Heavier decoder | 32–64 |
| Dropout | Risk of overfitting | Risk of underfitting | 0.2–0.3 |
| Historical Period | Less training data | More training data | 2–5 Years |

**If ROC-AUC < 0.55:** Increase historical period, reduce dropout, or try a different ticker with clearer trends.

**If val_loss increases early:** Increase dropout, reduce LSTM units, or increase the BUY threshold to rebalance classes.

**If model predicts only one class:** The dataset may be highly imbalanced — try lowering the BUY threshold or increasing the forecast horizon.
""")

st.caption("Stock Fund Analyzer · Built with Streamlit + TensorFlow/Keras + yfinance · For educational use only.")