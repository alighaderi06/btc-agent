import base64
import io
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import pandas as pd
import requests
import anthropic

SYMBOL = "BTCUSDT"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]

# (interval, candles to fetch, candles on chart, candles as text)
TIMEFRAMES = [("4h", 120, 70, 20), ("1h", 150, 80, 30), ("15m", 200, 90, 40)]

BASES = ["https://data-api.binance.vision", "https://api.binance.com"]

SYSTEM_PROMPT = """You are a price action analyst who follows Al Brooks' methodology.
Write in Persian (Farsi). Keep it concise and structured. Plain text only, no markdown symbols.

Analyze in this order:
1) 4h: market type (strong trend / broad channel / trading range / breakout / reversal), always-in direction,
   20 EMA behavior, key swing highs/lows, magnets, measured moves.
2) 1h: current leg or pullback (H1/H2/L1/L2), channel or range structure, trend bar vs doji bars, tails, overlap,
   failed breakouts, two-legged moves, wedges.
3) 15m (entry timeframe): is there a valid setup now? Signal bar quality, entry trigger, stop, first target,
   and what would invalidate the setup. If no clean setup exists, say clearly "no trade for now" and where to wait.
4) Final: most likely scenario with rough probability, alternative scenario, and key levels.

Rules: base your read on the charts and candle data only. Never invent numbers. Mention that the price is the
close of the last completed candle. This is an educational read, not financial advice; state it in one short line at the end.
Keep the whole message under 3000 characters."""


def fetch(interval, limit):
    last_err = None
    for base in BASES:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": SYMBOL, "interval": interval, "limit": limit},
                timeout=20,
            )
            r.raise_for_status()
            rows = r.json()
            df = pd.DataFrame(
                rows,
                columns=["t", "Open", "High", "Low", "Close", "Volume", "ct", "qv", "n", "tb", "tq", "ig"],
            )
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = df[c].astype(float)
            df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
            df.index.name = "Date"
            if int(df["ct"].iloc[-1]) > int(time.time() * 1000):
                df = df.iloc[:-1]
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Binance fetch failed for {interval}: {last_err}")


def make_chart(df, interval, show):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    view = df.iloc[-show:]
    buf = io.BytesIO()
    mpf.plot(
        view,
        type="candle",
        style="yahoo",
        addplot=[mpf.make_addplot(view["EMA20"], color="orange", width=1.2)],
        volume=False,
        title=f"{SYMBOL} {interval} (EMA20)",
        figsize=(12, 6.5),
        savefig=dict(fname=buf, format="png", dpi=110),
    )
    return buf.getvalue()


def candles_text(df, interval, n):
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d = d.iloc[-n:]
    lines = [f"[{interval}] time(UTC) O H L C EMA20"]
    for ts, r in d.iterrows():
        lines.append(
            f"{ts:%m-%d %H:%M} {r.Open:.0f} {r.High:.0f} {r.Low:.0f} {r.Close:.0f} {r.EMA20:.0f}"
        )
    return "\n".join(lines)


def analyze(charts, texts):
    client = anthropic.Anthropic()
    content = []
    for interval, png in charts:
        content.append({"type": "text", "text": f"Chart {interval}:"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(png).decode(),
                },
            }
        )
    content.append({"type": "text", "text": "Recent candle data:\n\n" + "\n\n".join(texts)})
    content.append({"type": "text", "text": "Write today's daily analysis."})
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def tg_message(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for i in range(0, len(text), 4000):
        r = requests.post(url, data={"chat_id": TG_CHAT, "text": text[i : i + 4000]}, timeout=30)
        r.raise_for_status()


def tg_charts(charts):
    media, files = [], {}
    for i, (interval, png) in enumerate(charts):
        media.append({"type": "photo", "media": f"attach://c{i}", "caption": f"{SYMBOL} {interval}"})
        files[f"c{i}"] = (f"{interval}.png", png, "image/png")
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMediaGroup",
        data={"chat_id": TG_CHAT, "media": json.dumps(media)},
        files=files,
        timeout=60,
    )
    r.raise_for_status()


def main():
    charts, texts = [], []
    for interval, limit, show, n in TIMEFRAMES:
        df = fetch(interval, limit)
        charts.append((interval, make_chart(df, interval, show)))
        texts.append(candles_text(df, interval, n))
    try:
        analysis = analyze(charts, texts)
    except Exception as e:
        analysis = f"تحلیل امروز انجام نشد: {e}"
    tg_message(f"📊 تحلیل روزانه بیت‌کوین (پرایس اکشن)\n\n{analysis}")
    tg_charts(charts)


if __name__ == "__main__":
    main()
