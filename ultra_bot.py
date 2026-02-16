#!/usr/bin/env python3
"""Ultra multi-timeframe signal bot (Python-only, no dashboard/JS)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from telegram import Bot, Update
from telegram.error import InvalidToken, TelegramError


@dataclass
class BotConfig:
    bybit_testnet: bool
    bybit_api_key: str
    bybit_api_secret: str
    symbol: str
    telegram_token: str
    telegram_chat_id: str
    poll_seconds: int
    min_signal_confidence: float
    cooldown_minutes: int
    default_risk_percent: float
    fear_greed_api_url: str

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            bybit_testnet=os.getenv("BYBIT_TESTNET", "false").lower() == "true",
            bybit_api_key=os.getenv("BYBIT_API_KEY", ""),
            bybit_api_secret=os.getenv("BYBIT_API_SECRET", ""),
            symbol=os.getenv("SYMBOL", "BTCUSDT"),
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
            min_signal_confidence=float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.90")),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "45")),
            default_risk_percent=float(os.getenv("DEFAULT_RISK_PERCENT", "1.0")),
            fear_greed_api_url=os.getenv("FEAR_GREED_API_URL", "https://api.alternative.me/fng/?limit=1&format=json"),
        )


@dataclass
class RuntimeStats:
    started_at: datetime
    cycles: int = 0
    alerts_sent: int = 0
    high_conf_count: int = 0
    errors: int = 0


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


class UltraSignalBot:
    TF_SETTINGS = {
        "5": {"weight": 0.20, "lookback": 220},
        "60": {"weight": 0.30, "lookback": 220},
        "240": {"weight": 0.25, "lookback": 220},
        "D": {"weight": 0.25, "lookback": 220},
    }

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.session = HTTP(testnet=config.bybit_testnet, api_key=config.bybit_api_key, api_secret=config.bybit_api_secret)
        self.tg = Bot(token=config.telegram_token)

        self.last_alert_key: str | None = None
        self.last_alert_at: datetime | None = None
        self.last_signal: Dict[str, float | str] | None = None
        self.last_tf_data: Dict[str, Dict[str, float]] = {}

        self.history: List[Dict[str, float | str]] = []
        self.max_history: int = 60000
        self.stats = RuntimeStats(started_at=datetime.now(timezone.utc))
        self.update_offset: int = 0
        self.telegram_ready: bool = True

    def _fetch_ohlcv(self, interval: str, limit: int) -> pd.DataFrame:
        res = self.session.get_kline(category="linear", symbol=self.config.symbol, interval=interval, limit=limit)
        rows = res.get("result", {}).get("list", [])
        if not rows:
            raise RuntimeError(f"No klines for interval={interval}")

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
        return df.dropna().copy()

    def _fetch_orderbook_liquidity(self) -> Dict[str, float]:
        try:
            res = self.session.get_orderbook(category="linear", symbol=self.config.symbol, limit=50)
            rows = res.get("result", {})
            bids = rows.get("b", [])
            asks = rows.get("a", [])
            if not bids or not asks:
                return {"orderbook_imbalance": 0.0, "spread_bps": 0.0, "liquidity_score": 0.5}

            bid_vol = sum(float(x[1]) for x in bids)
            ask_vol = sum(float(x[1]) for x in asks)
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread_bps = ((best_ask - best_bid) / ((best_ask + best_bid) / 2.0)) * 10000
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            liquidity_score = float(np.clip((bid_vol + ask_vol) / 1500.0, 0.0, 1.0))
            return {
                "orderbook_imbalance": float(np.clip(imbalance, -1.0, 1.0)),
                "spread_bps": float(max(spread_bps, 0.0)),
                "liquidity_score": liquidity_score,
            }
        except Exception as exc:
            logging.warning("Orderbook fetch failed: %s", exc)
            return {"orderbook_imbalance": 0.0, "spread_bps": 0.0, "liquidity_score": 0.5}

    def _fetch_fear_greed(self) -> Dict[str, float]:
        try:
            req = Request(self.config.fear_greed_api_url, headers={"User-Agent": "ultra-bot/1.0"})
            with urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            value = float(payload["data"][0]["value"])
            normalized = float(np.clip((value - 50.0) / 50.0, -1.0, 1.0))
            return {"fear_greed_index": value, "fear_greed_norm": normalized}
        except Exception as exc:
            logging.warning("Fear/Greed fetch failed: %s", exc)
            return {"fear_greed_index": 50.0, "fear_greed_norm": 0.0}

    @staticmethod
    def _fib_analysis(price: float, low: float, high: float) -> Dict[str, float | str]:
        span = max(high - low, 1e-9)
        levels = {
            "0.236": high - span * 0.236,
            "0.382": high - span * 0.382,
            "0.5": high - span * 0.5,
            "0.618": high - span * 0.618,
            "0.786": high - span * 0.786,
        }
        nearest_name = min(levels, key=lambda k: abs(price - levels[k]))
        pos = (price - low) / span
        zone = "премиум (дорого)" if pos > 0.7 else "дисконт (дешево)" if pos < 0.3 else "середина диапазона"
        return {
            "fib_nearest": nearest_name,
            "fib_level_price": float(levels[nearest_name]),
            "fib_position_pct": float(np.clip(pos * 100, 0, 100)),
            "fib_zone": zone,
        }

    def _fetch_market_context(self, tf_data: Dict[str, Dict[str, float]]) -> Dict[str, float | str]:
        ob = self._fetch_orderbook_liquidity()
        fg = self._fetch_fear_greed()
        fib = self._fib_analysis(tf_data["5"]["close"], tf_data["D"]["support"], tf_data["D"]["resistance"])
        return {**ob, **fg, **fib}

    @staticmethod
    def _analyze_timeframe(df: pd.DataFrame, weight: float) -> Dict[str, float]:
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

        ema20 = EMAIndicator(close, window=20).ema_indicator()
        ema50 = EMAIndicator(close, window=50).ema_indicator()
        ema200 = EMAIndicator(close, window=200).ema_indicator()
        rsi = RSIIndicator(close, window=14).rsi()
        macd_hist = MACD(close).macd_diff()
        adx = ADXIndicator(high, low, close, window=14).adx()
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        bb = BollingerBands(close, window=20, window_dev=2)
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
        cci = CCIIndicator(high, low, close, window=20).cci()

        c = close.iloc[-1]
        e20, e50, e200 = ema20.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]
        rsi_v, macd_v, adx_v, atr_v = rsi.iloc[-1], macd_hist.iloc[-1], adx.iloc[-1], atr.iloc[-1]
        bb_h, bb_l = bb.bollinger_hband().iloc[-1], bb.bollinger_lband().iloc[-1]
        stoch_v, cci_v = stoch.iloc[-1], cci.iloc[-1]

        trend_long = float(c > e20 > e50 > e200)
        trend_short = float(c < e20 < e50 < e200)
        rsi_long = np.clip((55 - rsi_v) / 20, -1, 1)
        rsi_short = np.clip((rsi_v - 45) / 20, -1, 1)
        macd_long = np.clip(macd_v / (close.pct_change().rolling(100).std().iloc[-1] + 1e-6), -1, 1)
        macd_short = -macd_long
        momentum = close.pct_change(5).iloc[-1]

        bb_pos = np.clip((c - bb_l) / (max(bb_h - bb_l, 1e-9)), 0, 1)
        bb_long = np.clip((0.55 - bb_pos) * 2, -1, 1)
        bb_short = np.clip((bb_pos - 0.45) * 2, -1, 1)
        stoch_long = np.clip((55 - stoch_v) / 25, -1, 1)
        stoch_short = np.clip((stoch_v - 45) / 25, -1, 1)
        cci_long = np.clip((-cci_v) / 180, -1, 1)
        cci_short = -cci_long

        volume_z = ((volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 1e-9)).iloc[-1]
        volume_boost = np.clip(volume_z / 2, -0.5, 0.5)
        adx_boost = np.clip((adx_v - 18) / 22, 0, 1)
        support = low.tail(50).min()
        resistance = high.tail(50).max()

        long_score = (
            0.30 * trend_long
            + 0.18 * rsi_long
            + 0.18 * macd_long
            + 0.13 * np.clip(momentum * 100, -1, 1)
            + 0.08 * volume_boost
            + 0.06 * bb_long
            + 0.04 * stoch_long
            + 0.03 * cci_long
        ) * (0.7 + 0.3 * adx_boost)

        short_score = (
            0.30 * trend_short
            + 0.18 * rsi_short
            + 0.18 * macd_short
            + 0.13 * np.clip(-momentum * 100, -1, 1)
            + 0.08 * volume_boost
            + 0.06 * bb_short
            + 0.04 * stoch_short
            + 0.03 * cci_short
        ) * (0.7 + 0.3 * adx_boost)

        return {
            "weight": weight,
            "close": float(c),
            "atr": float(atr_v),
            "support": float(support),
            "resistance": float(resistance),
            "adx": float(adx_v),
            "rsi": float(rsi_v),
            "stoch": float(stoch_v),
            "cci": float(cci_v),
            "long_score": float(np.clip(long_score, -1.5, 1.5)),
            "short_score": float(np.clip(short_score, -1.5, 1.5)),
        }

    def _compose_signal(self, tf_data: Dict[str, Dict[str, float]], market_ctx: Dict[str, float | str]) -> Dict[str, float | str]:
        long_total = sum(v["long_score"] * v["weight"] for v in tf_data.values())
        short_total = sum(v["short_score"] * v["weight"] for v in tf_data.values())
        net = long_total - short_total
        direction = "LONG" if net >= 0 else "SHORT"

        agreement = sum(v["weight"] for v in tf_data.values() if (v["long_score"] > v["short_score"]) == (direction == "LONG"))
        conviction = sum(abs(v["long_score"] - v["short_score"]) * v["weight"] for v in tf_data.values())
        adx_quality = np.average([min(v["adx"], 40.0) / 40.0 for v in tf_data.values()], weights=[v["weight"] for v in tf_data.values()])

        confidence_raw = (0.55 * abs(net) + 0.25 * conviction + 0.20 * agreement) * (0.75 + 0.25 * adx_quality)

        liquidity_score = float(market_ctx["liquidity_score"])
        orderbook_imbalance = float(market_ctx["orderbook_imbalance"])
        fear_greed_norm = float(market_ctx["fear_greed_norm"])
        directional_liquidity_boost = 1 + (0.12 * orderbook_imbalance if direction == "LONG" else -0.12 * orderbook_imbalance)
        sentiment_boost = 1 + (0.08 * fear_greed_norm if direction == "LONG" else -0.08 * fear_greed_norm)
        confidence_raw *= float(np.clip(directional_liquidity_boost, 0.85, 1.15))
        confidence_raw *= float(np.clip(sentiment_boost, 0.90, 1.10))
        confidence_raw *= (0.90 + 0.10 * liquidity_score)

        confidence = float(1 / (1 + math.exp(-12 * (confidence_raw - 0.50))))

        last_price = tf_data["5"]["close"]
        atr = np.average([v["atr"] for v in tf_data.values()], weights=[v["weight"] for v in tf_data.values()])
        support = min(v["support"] for v in tf_data.values())
        resistance = max(v["resistance"] for v in tf_data.values())

        sl_mult = max(1.1, 2.15 - confidence)
        rr = 2.0 + confidence
        if direction == "LONG":
            stop_loss = max(last_price - sl_mult * atr, support * 0.998)
            take_profit = last_price + rr * (last_price - stop_loss)
        else:
            stop_loss = min(last_price + sl_mult * atr, resistance * 1.002)
            take_profit = last_price - rr * (stop_loss - last_price)

        risk = abs(last_price - stop_loss)
        reward = abs(take_profit - last_price)
        return {
            "direction": direction,
            "confidence": float(confidence),
            "price": float(last_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "risk": float(risk),
            "reward": float(reward),
            "rr": float(reward / (risk + 1e-9)),
            "long_total": float(long_total),
            "short_total": float(short_total),
            "agreement": float(agreement),
            "conviction": float(conviction),
            "adx_quality": float(adx_quality),
            "entry_hint": self._entry_hint(direction, float(confidence), float(agreement)),
            "orderbook_imbalance": orderbook_imbalance,
            "spread_bps": float(market_ctx["spread_bps"]),
            "liquidity_score": liquidity_score,
            "fear_greed_index": float(market_ctx["fear_greed_index"]),
            "fear_greed_norm": fear_greed_norm,
            "fib_nearest": str(market_ctx["fib_nearest"]),
            "fib_level_price": float(market_ctx["fib_level_price"]),
            "fib_position_pct": float(market_ctx["fib_position_pct"]),
            "fib_zone": str(market_ctx["fib_zone"]),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _entry_hint(direction: str, success_prob: float, agreement: float) -> str:
        if success_prob >= 0.9 and agreement >= 0.75:
            return f"Сильный {direction}: вход по рынку или от первого отката на 5м."
        if success_prob >= 0.75:
            return f"Умеренный {direction}: вход после подтверждения свечой на 5м/1ч."
        return f"Слабый {direction}: вход рискованный, лучше ждать усиления сигнала."

    @staticmethod
    def _recommendations(signal: Dict[str, float | str]) -> str:
        prob = float(signal["confidence"]) * 100
        rr = float(signal["rr"])
        fgi = float(signal.get("fear_greed_index", 50.0))
        direction = str(signal["direction"])
        lines = []
        if prob >= 90:
            lines.append("✅ Вероятность высокая: рабочий вход по риск-менеджменту допустим.")
        elif prob >= 75:
            lines.append("⚠️ Вероятность средняя: уменьшите объем и ждите подтверждения.")
        else:
            lines.append("⛔ Вероятность низкая: лучше пропустить вход.")

        if rr >= 2.5:
            lines.append("✅ R/R отличное (выше 1:2.5).")
        elif rr >= 1.8:
            lines.append("⚠️ R/R нормальное, но фиксируйте частями.")
        else:
            lines.append("⛔ R/R слабое — вход не рекомендуется.")

        if fgi >= 70:
            lines.append("😬 Индекс жадности высокий: возможны резкие откаты.")
        elif fgi <= 30:
            lines.append("😨 Индекс страха высокий: возможны выносы и волатильность.")

        lines.append(f"📌 Направление: {direction}. Не входите против сигнала.")
        return "\n".join(lines)

    def _signal_text(self, signal: Dict[str, float | str], tf_data: Dict[str, Dict[str, float]], extended: bool = False) -> str:
        success_prob = float(signal["confidence"]) * 100
        tf_lines = []
        for tf, data in tf_data.items():
            bias = "LONG" if data["long_score"] > data["short_score"] else "SHORT"
            tf_lines.append(
                f"• {tf:>3}: {bias} | L={data['long_score']:.3f} S={data['short_score']:.3f} "
                f"RSI={data['rsi']:.1f} STOCH={data['stoch']:.1f} CCI={data['cci']:.1f} ADX={data['adx']:.1f}"
            )

        summary = (
            f"🔥 *ULTRA SIGNAL {signal['direction']}*\n"
            f"Символ: *{self.config.symbol}*\n"
            f"Вероятность успеха: *{success_prob:.2f}%*\n"
            f"Вход: `{signal['price']:.4f}`\n"
            f"Stop Loss: `{signal['stop_loss']:.4f}`\n"
            f"Take Profit: `{signal['take_profit']:.4f}`\n"
            f"R/R: `{signal['rr']:.2f}`\n"
            f"Risk: `{signal['risk']:.4f}` | Reward: `{signal['reward']:.4f}`\n"
            f"Сила LONG/SHORT: `{signal['long_total']:.3f} / {signal['short_total']:.3f}`\n"
            f"Совпадение ТФ: `{float(signal['agreement'])*100:.0f}%` | Убедительность: `{signal['conviction']:.3f}`\n"
            f"Качество тренда (ADX): `{float(signal['adx_quality'])*100:.0f}%`\n"
            f"Когда входить: _{signal['entry_hint']}_\n"
            f"Ликвидность: `{float(signal['liquidity_score'])*100:.0f}%` | Спред: `{signal['spread_bps']:.2f} bps` | Дисбаланс стакана: `{float(signal['orderbook_imbalance'])*100:.1f}%`\n"
            f"Индекс страха/жадности: `{signal['fear_greed_index']:.0f}` | Fibo: `{signal['fib_nearest']}` @ `{signal['fib_level_price']:.4f}` ({signal['fib_zone']})\n\n"
            "*Разбор по ТФ:*\n"
            + "\n".join(tf_lines)
        )

        if not extended:
            return summary + "\n\n*Рекомендации:*\n" + self._recommendations(signal)

        uptime_min = int((datetime.now(timezone.utc) - self.stats.started_at).total_seconds() // 60)
        return (
            summary
            + "\n\n*Рекомендации:*\n"
            + self._recommendations(signal)
            + "\n\n*Runtime:*\n"
            + f"Циклов: `{self.stats.cycles}` | Сигналов 90%+: `{self.stats.high_conf_count}` | Алертов: `{self.stats.alerts_sent}`\n"
            + f"Ошибок: `{self.stats.errors}` | Аптайм: `{uptime_min} мин`"
        )

    async def _send_text(self, text: str, chat_id: int | str | None = None) -> None:
        try:
            await self.tg.send_message(chat_id=chat_id or self.config.telegram_chat_id, text=text, parse_mode="Markdown")
        except InvalidToken:
            self.telegram_ready = False
            self.stats.errors += 1
            logging.error("Invalid TELEGRAM_TOKEN: Telegram rejected sendMessage (404 Not Found).")
        except TelegramError as exc:
            self.stats.errors += 1
            logging.exception("Telegram send failed: %s", exc)

    async def _send_signal(self, signal: Dict[str, float | str], tf_data: Dict[str, Dict[str, float]]) -> None:
        logging.info("Sending HIGH-CONFIDENCE signal alert to Telegram.")
        await self._send_text(self._signal_text(signal, tf_data, extended=False))

    def _can_alert(self, signal: Dict[str, float | str]) -> bool:
        if float(signal["confidence"]) < self.config.min_signal_confidence:
            return False

        key = f"{signal['direction']}:{round(signal['price'], 2)}:{round(signal['stop_loss'], 2)}:{round(signal['take_profit'], 2)}"
        now = datetime.now(timezone.utc)
        if self.last_alert_key == key and self.last_alert_at:
            if (now - self.last_alert_at).total_seconds() / 60 < self.config.cooldown_minutes:
                return False

        self.last_alert_key = key
        self.last_alert_at = now
        return True

    def _build_sdel_text(self, amount: float, risk_percent: float) -> str:
        if not self.last_signal:
            return "Нет свежего сигнала. Подожди 1-2 цикла сканера и повтори /sdel"

        signal = self.last_signal
        entry = float(signal["price"])
        sl = float(signal["stop_loss"])
        tp = float(signal["take_profit"])
        stop_distance = abs(entry - sl)
        risk_capital = amount * (risk_percent / 100)
        qty = risk_capital / (stop_distance + 1e-9)
        notional = qty * entry
        pnl_tp = abs(tp - entry) * qty

        return (
            f"💼 *План сделки* ({self.config.symbol})\n"
            f"Направление: *{signal['direction']}*\n"
            f"Сумма депо: `{amount:.2f} USDT`\n"
            f"Риск на сделку: `{risk_percent:.2f}%` (`{risk_capital:.2f} USDT`)\n\n"
            f"Вход: `{entry:.4f}`\n"
            f"Stop Loss: `{sl:.4f}`\n"
            f"Take Profit: `{tp:.4f}`\n"
            f"Дистанция до SL: `{stop_distance:.4f}`\n\n"
            f"Реком. размер позиции: `{qty:.6f}` {self.config.symbol.replace('USDT', '')}\n"
            f"Номинал позиции: `{notional:.2f} USDT`\n"
            f"Потенц. прибыль до TP: `{pnl_tp:.2f} USDT`\n"
            f"R/R: `{signal['rr']:.2f}`"
        )

    def _window_stats(self, days: int) -> Dict[str, float]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        window = [x for x in self.history if datetime.fromisoformat(str(x["ts"])) >= cutoff]
        if not window:
            return {"count": 0, "long_pct": 0.0, "short_pct": 0.0, "prob": 0.0, "ind_long": 0.0, "ind_short": 0.0}

        long_pct = sum(1 for x in window if x["direction"] == "LONG") / len(window) * 100
        avg_prob = sum(float(x["confidence"]) for x in window) / len(window) * 100
        ind_long = sum(float(x["agreement"]) for x in window) / len(window) * 100
        return {
            "count": len(window),
            "long_pct": long_pct,
            "short_pct": 100 - long_pct,
            "prob": avg_prob,
            "ind_long": ind_long,
            "ind_short": 100 - ind_long,
        }

    async def _handle_command(self, update: Update) -> None:
        message = update.effective_message
        if not message or not message.text:
            return

        text = message.text.strip()
        cmd, *args = text.split()
        chat_id = str(update.effective_chat.id)
        if chat_id != str(self.config.telegram_chat_id):
            return

        logging.info("Command received: %s", cmd)

        if cmd in {"/start", "/help"}:
            await self._send_text(
                "🚀 *Команды бота:*\n"
                "`/status` — полный анализ: вероятность, стакан, фибо, индикаторы и рекомендации\n"
                "`/stats` — статистика день/неделя/месяц + % индикаторов LONG/SHORT\n"
                "`/sdel <summa_usdt> [risk_%]` — план сделки со SL/TP\n"
                "Пример: `/sdel 1000 1.5`",
                chat_id=chat_id,
            )
            return

        if cmd in {"/status", "/signal"}:
            if not self.last_signal:
                await self._send_text("Сигналов пока нет. Ждём первый проход сканера.", chat_id=chat_id)
                return
            await self._send_text(self._signal_text(self.last_signal, self.last_tf_data, extended=True), chat_id=chat_id)
            return

        if cmd == "/stats":
            uptime_min = int((datetime.now(timezone.utc) - self.stats.started_at).total_seconds() // 60)
            recent = self.history[-5:]
            history_text = (
                "\n".join([f"• {x['ts'][11:16]} {x['direction']} prob={float(x['confidence'])*100:.1f}% rr={x['rr']:.2f}" for x in recent])
                if recent
                else "нет данных"
            )

            high_rate = (self.stats.high_conf_count / self.stats.cycles * 100) if self.stats.cycles else 0.0
            alert_rate = (self.stats.alerts_sent / self.stats.cycles * 100) if self.stats.cycles else 0.0
            long_pct = (sum(1 for x in self.history if x["direction"] == "LONG") / len(self.history) * 100) if self.history else 0.0
            short_pct = 100 - long_pct if self.history else 0.0

            day = self._window_stats(1)
            week = self._window_stats(7)
            month = self._window_stats(30)

            await self._send_text(
                "📊 *Статистика бота*\n"
                f"Символ: `{self.config.symbol}`\n"
                f"Циклов: `{self.stats.cycles}`\n"
                f"Сигналов 90%+: `{self.stats.high_conf_count}` (`{high_rate:.1f}%`)\n"
                f"Отправлено алертов: `{self.stats.alerts_sent}` (`{alert_rate:.1f}%`)\n"
                f"Распределение: LONG `{long_pct:.1f}%` | SHORT `{short_pct:.1f}%`\n"
                f"Ошибок: `{self.stats.errors}`\n"
                f"Аптайм: `{uptime_min} мин`\n\n"
                "*Индикаторы и сигналы (день/неделя/месяц):*\n"
                f"1д: сигналов `{day['count']}`, LONG `{day['long_pct']:.1f}%`, SHORT `{day['short_pct']:.1f}%`, сред.вероятность `{day['prob']:.1f}%`, индикаторы L/S `{day['ind_long']:.1f}%/{day['ind_short']:.1f}%`\n"
                f"7д: сигналов `{week['count']}`, LONG `{week['long_pct']:.1f}%`, SHORT `{week['short_pct']:.1f}%`, сред.вероятность `{week['prob']:.1f}%`, индикаторы L/S `{week['ind_long']:.1f}%/{week['ind_short']:.1f}%`\n"
                f"30д: сигналов `{month['count']}`, LONG `{month['long_pct']:.1f}%`, SHORT `{month['short_pct']:.1f}%`, сред.вероятность `{month['prob']:.1f}%`, индикаторы L/S `{month['ind_long']:.1f}%/{month['ind_short']:.1f}%`\n\n"
                "*Последние сигналы:*\n"
                f"{history_text}",
                chat_id=chat_id,
            )
            return

        if cmd == "/sdel":
            if not args:
                await self._send_text("Использование: `/sdel 1000` или `/sdel 1000 1.5`", chat_id=chat_id)
                return
            try:
                amount = float(args[0])
                risk_percent = float(args[1]) if len(args) > 1 else self.config.default_risk_percent
                if amount <= 0 or risk_percent <= 0:
                    raise ValueError
            except ValueError:
                await self._send_text("Неверные значения. Пример: `/sdel 1000 1.5`", chat_id=chat_id)
                return
            await self._send_text(self._build_sdel_text(amount, risk_percent), chat_id=chat_id)
            return

    async def _process_updates(self) -> None:
        if not self.telegram_ready:
            return

        try:
            updates = await self.tg.get_updates(offset=self.update_offset, timeout=0, allowed_updates=["message"])
        except InvalidToken:
            self.telegram_ready = False
            self.stats.errors += 1
            logging.error("Invalid TELEGRAM_TOKEN: Telegram rejected getUpdates (404 Not Found).")
            logging.error("Set correct TELEGRAM_TOKEN in .env and restart bot.")
            return
        except TelegramError as exc:
            self.stats.errors += 1
            logging.exception("Telegram updates failed: %s", exc)
            return

        for update in updates:
            self.update_offset = update.update_id + 1
            try:
                await self._handle_command(update)
            except Exception as exc:
                self.stats.errors += 1
                logging.exception("Command handling failed: %s", exc)

    async def run(self) -> None:
        logging.info("UltraSignalBot started for %s", self.config.symbol)
        while True:
            try:
                await self._process_updates()

                tf_data: Dict[str, Dict[str, float]] = {}
                for tf, cfg in self.TF_SETTINGS.items():
                    df = self._fetch_ohlcv(tf, cfg["lookback"])
                    tf_data[tf] = self._analyze_timeframe(df, cfg["weight"])

                market_ctx = self._fetch_market_context(tf_data)
                signal = self._compose_signal(tf_data, market_ctx)

                self.stats.cycles += 1
                self.last_signal = signal
                self.last_tf_data = tf_data
                self.history.append(signal)
                self.history = self.history[-self.max_history :]

                if float(signal["confidence"]) >= self.config.min_signal_confidence:
                    self.stats.high_conf_count += 1

                logging.info(
                    "Signal check: dir=%s prob=%.3f long=%.3f short=%.3f agreement=%.2f conviction=%.3f liq=%.2f fng=%.0f",
                    signal["direction"],
                    signal["confidence"],
                    signal["long_total"],
                    signal["short_total"],
                    signal["agreement"],
                    signal["conviction"],
                    signal["liquidity_score"],
                    signal["fear_greed_index"],
                )

                if self._can_alert(signal):
                    await self._send_signal(signal, tf_data)
                    self.stats.alerts_sent += 1
                    logging.info("Signal sent to Telegram.")
            except Exception as exc:
                self.stats.errors += 1
                logging.exception("Bot cycle failed: %s", exc)

            await asyncio.sleep(self.config.poll_seconds)


async def _verify_telegram(config: BotConfig) -> bool:
    try:
        tg = Bot(token=config.telegram_token)
        me = await tg.get_me()
        logging.info("Telegram bot auth OK: @%s", me.username or me.first_name)
        return True
    except InvalidToken:
        logging.error("Invalid TELEGRAM_TOKEN: Telegram API returned 404 Not Found.")
        logging.error("Open @BotFather, regenerate token, update .env, restart bot.")
        return False
    except TelegramError as exc:
        logging.error("Telegram auth check failed: %s", exc)
        return False


async def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    _load_env_file(".env")
    cfg = BotConfig.from_env()

    missing = [name for name, value in {"TELEGRAM_TOKEN": cfg.telegram_token, "TELEGRAM_CHAT_ID": cfg.telegram_chat_id}.items() if not value]
    if missing:
        logging.error("Missing required env vars: %s", ", ".join(missing))
        logging.error("Create .env from template and fill Telegram settings:")
        logging.error("  cp .env.ultra.example .env")
        logging.error("  nano .env")
        return

    ok = await _verify_telegram(cfg)
    if not ok:
        return

    bot = UltraSignalBot(cfg)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
