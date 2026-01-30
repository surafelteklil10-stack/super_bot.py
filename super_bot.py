import os
import time
import threading
import requests
from datetime import datetime
from pybit.unified_trading import HTTP

# ======================================================
# 1️⃣ ENV + MODE CONFIG
# ======================================================

BOT_NAME = os.getenv("BOT_NAME", "SUPER_BOT")

MODE = os.getenv("MODE", "DEMO")  # DEMO | REAL
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ======================================================
# 2️⃣ API KEYS
# ======================================================

DEMO_KEY = os.getenv("BYBIT_DEMO_KEY")
DEMO_SECRET = os.getenv("BYBIT_DEMO_SECRET")

REAL_KEY = os.getenv("BYBIT_REAL_KEY")
REAL_SECRET = os.getenv("BYBIT_REAL_SECRET")

TG_TOKEN = os.getenv("TG_TOKEN")
TG_ADMIN = int(os.getenv("TG_ADMIN", "0"))

if MODE == "REAL":
    API_KEY = REAL_KEY
    API_SECRET = REAL_SECRET
    TESTNET = False
else:
    API_KEY = DEMO_KEY
    API_SECRET = DEMO_SECRET
    TESTNET = True

# ======================================================
# 3️⃣ GLOBAL BOT STATE
# ======================================================

BOT_ACTIVE = True
KILL_SWITCH = False

START_DAY_BALANCE = None
TRADES_TODAY = 0
OPEN_TRADES = {}

LAST_HEARTBEAT = time.time()

# ======================================================
# 4️⃣ RISK & TRADE SETTINGS (BASE – WILL EXTEND LATER)
# ======================================================

LEVERAGE = int(os.getenv("LEVERAGE", 20))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))   # 2%
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 0.10))  # 10%
MAX_DAILY_PROFIT = float(os.getenv("MAX_DAILY_PROFIT", 0.25))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES", 5))

# ======================================================
# 5️⃣ CONNECT TO BYBIT
# ======================================================

print("🔌 Connecting to Bybit...")
print("MODE:", MODE, "| TESTNET:", TESTNET)

session = HTTP(
    testnet=TESTNET,
    api_key=API_KEY,
    api_secret=API_SECRET
)

# ======================================================
# 6️⃣ TELEGRAM CORE (SAFE)
# ======================================================

def tg(msg: str):
    if not TG_TOKEN or TG_ADMIN == 0:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={
                "chat_id": TG_ADMIN,
                "text": msg
            },
            timeout=5
        )
    except Exception as e:
        if DEBUG:
            print("Telegram error:", e)

# ======================================================
# 7️⃣ WALLET / BALANCE
# ======================================================

def get_balance() -> float:
    try:
        r = session.get_wallet_balance(accountType="UNIFIED")
        return float(r["result"]["list"][0]["totalWalletBalance"])
    except Exception as e:
        if DEBUG:
            print("Balance error:", e)
        return 0.0

# ======================================================
# 8️⃣ DAILY INIT
# ======================================================

def init_day():
    global START_DAY_BALANCE, TRADES_TODAY, KILL_SWITCH

    START_DAY_BALANCE = get_balance()
    TRADES_TODAY = 0
    KILL_SWITCH = False

    tg(
        f"🚀 {BOT_NAME} STARTED\n"
        f"Mode: {MODE}\n"
        f"Balance: {START_DAY_BALANCE}"
    )

# ======================================================
# 9️⃣ DAILY RISK CHECK (CORE)
# ======================================================

def daily_risk_check():
    global KILL_SWITCH

    if START_DAY_BALANCE is None:
        return

    bal = get_balance()
    pnl = (bal - START_DAY_BALANCE) / START_DAY_BALANCE

    if pnl <= -MAX_DAILY_LOSS:
        KILL_SWITCH = True
        tg("🛑 DAILY LOSS LIMIT HIT")

    if pnl >= MAX_DAILY_PROFIT:
        KILL_SWITCH = True
        tg("🎯 DAILY PROFIT TARGET HIT")

# ======================================================
# 10️⃣ HEARTBEAT (BOT ALIVE)
# ======================================================

def heartbeat():
    global LAST_HEARTBEAT
    while True:
        LAST_HEARTBEAT = time.time()
        if DEBUG:
            print("❤️ Heartbeat | Balance:", get_balance())
        time.sleep(60)

# ======================================================
# 11️⃣ MAIN (TEMP – WILL EXPAND)
# ======================================================

if __name__ == "__main__":
    init_day()
    threading.Thread(target=heartbeat, daemon=True).start()

    while True:
        daily_risk_check()
        time.sleep(15)

  # ======================================================
# PART 2 : MARKET DATA + SYMBOL SCANNER
# ======================================================

# ===============================
# 12️⃣ SYMBOL UNIVERSE (100+ PAIRS)
# ===============================

QUOTE_ASSET = "USDT"

def load_symbols():
    """
    Load all USDT perpetual contracts from Bybit
    """
    symbols = []
    try:
        r = session.get_instruments_info(
            category="linear",
            status="Trading"
        )
        for s in r["result"]["list"]:
            if s["quoteCoin"] == QUOTE_ASSET:
                symbols.append(s["symbol"])
    except Exception as e:
        if DEBUG:
            print("Symbol load error:", e)
    return symbols


SYMBOLS = load_symbols()

print(f"📈 Loaded {len(SYMBOLS)} trading pairs")

# ===============================
# 13️⃣ MARKET DATA FETCHER
# ===============================

def get_klines(symbol, interval="15", limit=100):
    """
    Fetch candlestick data
    """
    try:
        r = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        return r["result"]["list"]
    except Exception as e:
        if DEBUG:
            print(f"Kline error {symbol}:", e)
        return []


# ===============================
# 14️⃣ BASIC MARKET FILTERS
# ===============================

def is_market_active(symbol):
    """
    Basic liquidity & activity check
    """
    try:
        ticker = session.get_tickers(
            category="linear",
            symbol=symbol
        )["result"]["list"][0]

        volume = float(ticker["turnover24h"])
        price = float(ticker["lastPrice"])

        if volume < 5_000_000:   # low liquidity filter
            return False
        if price <= 0:
            return False

        return True

    except Exception:
        return False


# ===============================
# 15️⃣ SIMPLE TECH CALCULATIONS
# ===============================

def calculate_trend(klines):
    """
    Very basic trend logic (will be upgraded later)
    """
    closes = [float(k[4]) for k in klines]

    if len(closes) < 20:
        return None

    sma_fast = sum(closes[-10:]) / 10
    sma_slow = sum(closes[-20:]) / 20

    if sma_fast > sma_slow:
        return "LONG"
    elif sma_fast < sma_slow:
        return "SHORT"
    else:
        return None


# ===============================
# 16️⃣ SIGNAL SCANNER (NO ORDER!)
# ===============================

def scan_signals():
    """
    Scan all symbols and return trade candidates
    """
    signals = []

    if KILL_SWITCH or not BOT_ACTIVE:
        return signals

    for symbol in SYMBOLS:
        if symbol in OPEN_TRADES:
            continue

        if not is_market_active(symbol):
            continue

        klines = get_klines(symbol)
        if not klines:
            continue

        direction = calculate_trend(klines)
        if not direction:
            continue

        signal = {
            "symbol": symbol,
            "side": direction,
            "time": datetime.utcnow().isoformat()
        }

        signals.append(signal)

    if DEBUG:
        print(f"🔍 Signals found: {len(signals)}")

    return signals

# ======================================================
# PART 3 : ADVANCED STRATEGY & AI TRADE FILTER
# ======================================================

# ===============================
# 17️⃣ MULTI-TIMEFRAME DATA
# ===============================

def get_multi_tf(symbol):
    """
    Get multiple timeframe klines
    """
    return {
        "5m": get_klines(symbol, "5", 100),
        "15m": get_klines(symbol, "15", 100),
        "1h": get_klines(symbol, "60", 100)
    }


# ===============================
# 18️⃣ TREND CONFIRMATION
# ===============================

def confirm_trend(mtf_data):
    """
    Confirm direction across timeframes
    """
    directions = []

    for tf in ["5m", "15m", "1h"]:
        trend = calculate_trend(mtf_data[tf])
        if trend:
            directions.append(trend)

    if directions.count("LONG") >= 2:
        return "LONG"
    if directions.count("SHORT") >= 2:
        return "SHORT"

    return None


# ===============================
# 19️⃣ VOLATILITY FILTER
# ===============================

def volatility_ok(klines):
    """
    Filter out dead or crazy markets
    """
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]

    if len(closes) < 20:
        return False

    avg_range = sum([(h - l) for h, l in zip(highs[-20:], lows[-20:])]) / 20
    price = closes[-1]

    # Too flat or too wild
    if avg_range / price < 0.002:
        return False
    if avg_range / price > 0.05:
        return False

    return True


# ===============================
# 20️⃣ MOMENTUM SCORE (AI-LIKE)
# ===============================

def momentum_score(klines):
    """
    Simple momentum scoring system
    """
    closes = [float(k[4]) for k in klines]

    score = 0

    if closes[-1] > closes[-5]:
        score += 1
    if closes[-1] > closes[-10]:
        score += 1
    if closes[-5] > closes[-10]:
        score += 1

    return score  # max 3


# ===============================
# 21️⃣ AI TRADE FILTER
# ===============================

def ai_trade_filter(signal):
    """
    Final decision maker (NO ORDER)
    """
    symbol = signal["symbol"]

    mtf = get_multi_tf(symbol)

    # Trend alignment
    direction = confirm_trend(mtf)
    if not direction or direction != signal["side"]:
        return None

    # Volatility check
    if not volatility_ok(mtf["15m"]):
        return None

    # Momentum score
    score = momentum_score(mtf["15m"])
    if score < 2:
        return None

    signal["confidence"] = score
    signal["tf_bias"] = direction

    return signal


# ===============================
# 22️⃣ FILTERED SIGNAL PIPELINE
# ===============================

def filter_signals(raw_signals):
    """
    Apply AI filter to all signals
    """
    final_signals = []

    for sig in raw_signals:
        filtered = ai_trade_filter(sig)
        if filtered:
            final_signals.append(filtered)

    if DEBUG:
        print(f"🤖 AI Approved signals: {len(final_signals)}")

    return final_signals

# ======================================================
# PART 4 : TRADE ENGINE STRUCTURE & RISK LOGIC
# ======================================================

# ===============================
# 23️⃣ POSITION SIZE CALCULATOR
# ===============================

def calculate_position_size(balance, entry, stop):
    """
    Risk-based position sizing
    """
    risk_amount = balance * RISK_PER_TRADE
    stop_distance = abs(entry - stop)

    if stop_distance == 0:
        return 0

    position_size = risk_amount / stop_distance
    return round(position_size, 3)


# ===============================
# 24️⃣ DYNAMIC STOP LOSS
# ===============================

def calculate_stop_loss(signal, klines):
    """
    ATR-like stop logic (simplified)
    """
    highs = [float(k[2]) for k in klines[-20:]]
    lows = [float(k[3]) for k in klines[-20:]]

    avg_range = sum([(h - l) for h, l in zip(highs, lows)]) / len(highs)
    entry = signal["entry"]

    if signal["side"] == "LONG":
        return entry - avg_range * 1.5
    else:
        return entry + avg_range * 1.5


# ===============================
# 25️⃣ TAKE PROFIT LOGIC
# ===============================

def calculate_take_profit(entry, stop, rr=2):
    """
    Risk : Reward calculation
    """
    risk = abs(entry - stop)
    return entry + risk * rr if entry > stop else entry - risk * rr


# ===============================
# 26️⃣ TRADE BLUEPRINT
# ===============================

def build_trade(signal):
    """
    Build full trade plan (NO ORDER)
    """
    balance = get_balance()
    klines = get_klines(signal["symbol"], "15", 100)

    stop = calculate_stop_loss(signal, klines)
    tp = calculate_take_profit(signal["entry"], stop)

    size = calculate_position_size(balance, signal["entry"], stop)

    if size <= 0:
        return None

    trade = {
        "symbol": signal["symbol"],
        "side": signal["side"],
        "entry": signal["entry"],
        "stop_loss": round(stop, 4),
        "take_profit": round(tp, 4),
        "size": size,
        "confidence": signal.get("confidence", 0),
        "timestamp": datetime.utcnow().isoformat()
    }

    return trade


# ===============================
# 27️⃣ DAILY LIMIT CHECK
# ===============================

def can_trade():
    """
    Check global limits
    """
    if not BOT_ACTIVE:
        return False

    if KILL_SWITCH:
        return False

    if TRADES_TODAY >= MAX_TRADES:
        return False

    return True


# ===============================
# 28️⃣ TRADE ENGINE (DRY RUN)
# ===============================

def trade_engine(filtered_signals):
    """
    Build trades without execution
    """
    trades = []

    if not can_trade():
        return trades

    for sig in filtered_signals:
        trade = build_trade(sig)
        if trade:
            trades.append(trade)

    if DEBUG:
        print(f"⚙️ Trades prepared: {len(trades)}")

    return trades

# ======================================================
# PART 4 : TRADE ENGINE STRUCTURE & RISK LOGIC
# ======================================================

# ===============================
# 23️⃣ POSITION SIZE CALCULATOR
# ===============================

def calculate_position_size(balance, entry, stop):
    """
    Risk-based position sizing
    """
    risk_amount = balance * RISK_PER_TRADE
    stop_distance = abs(entry - stop)

    if stop_distance == 0:
        return 0

    position_size = risk_amount / stop_distance
    return round(position_size, 3)


# ===============================
# 24️⃣ DYNAMIC STOP LOSS
# ===============================

def calculate_stop_loss(signal, klines):
    """
    ATR-like stop logic (simplified)
    """
    highs = [float(k[2]) for k in klines[-20:]]
    lows = [float(k[3]) for k in klines[-20:]]

    avg_range = sum([(h - l) for h, l in zip(highs, lows)]) / len(highs)
    entry = signal["entry"]

    if signal["side"] == "LONG":
        return entry - avg_range * 1.5
    else:
        return entry + avg_range * 1.5


# ===============================
# 25️⃣ TAKE PROFIT LOGIC
# ===============================

def calculate_take_profit(entry, stop, rr=2):
    """
    Risk : Reward calculation
    """
    risk = abs(entry - stop)
    return entry + risk * rr if entry > stop else entry - risk * rr


# ===============================
# 26️⃣ TRADE BLUEPRINT
# ===============================

def build_trade(signal):
    """
    Build full trade plan (NO ORDER)
    """
    balance = get_balance()
    klines = get_klines(signal["symbol"], "15", 100)

    stop = calculate_stop_loss(signal, klines)
    tp = calculate_take_profit(signal["entry"], stop)

    size = calculate_position_size(balance, signal["entry"], stop)

    if size <= 0:
        return None

    trade = {
        "symbol": signal["symbol"],
        "side": signal["side"],
        "entry": signal["entry"],
        "stop_loss": round(stop, 4),
        "take_profit": round(tp, 4),
        "size": size,
        "confidence": signal.get("confidence", 0),
        "timestamp": datetime.utcnow().isoformat()
    }

    return trade


# ===============================
# 27️⃣ DAILY LIMIT CHECK
# ===============================

def can_trade():
    """
    Check global limits
    """
    if not BOT_ACTIVE:
        return False

    if KILL_SWITCH:
        return False

    if TRADES_TODAY >= MAX_TRADES:
        return False

    return True


# ===============================
# 28️⃣ TRADE ENGINE (DRY RUN)
# ===============================

def trade_engine(filtered_signals):
    """
    Build trades without execution
    """
    trades = []

    if not can_trade():
        return trades

    for sig in filtered_signals:
        trade = build_trade(sig)
        if trade:
            trades.append(trade)

    if DEBUG:
        print(f"⚙️ Trades prepared: {len(trades)}")

    return trades

# ======================================================
# PART 5 : TRADE EXECUTION (DEMO / REAL)
# ======================================================

# ===============================
# 29️⃣ SET LEVERAGE
# ===============================

def set_leverage(symbol, leverage):
    try:
        session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=leverage,
            sellLeverage=leverage
        )
    except Exception as e:
        print("Leverage error:", e)


# ===============================
# 30️⃣ PLACE MARKET ORDER
# ===============================

def place_order(trade):
    """
    Execute trade on Bybit
    """
    try:
        set_leverage(trade["symbol"], LEVERAGE)

        side = "Buy" if trade["side"] == "LONG" else "Sell"

        order = session.place_order(
            category="linear",
            symbol=trade["symbol"],
            side=side,
            orderType="Market",
            qty=trade["size"],
            timeInForce="GoodTillCancel",
            reduceOnly=False,
            closeOnTrigger=False
        )

        return order

    except Exception as e:
        print("Order error:", e)
        return None


# ===============================
# 31️⃣ SET SL & TP
# ===============================

def set_sl_tp(trade):
    try:
        session.set_trading_stop(
            category="linear",
            symbol=trade["symbol"],
            stopLoss=trade["stop_loss"],
            takeProfit=trade["take_profit"],
            tpTriggerBy="LastPrice",
            slTriggerBy="LastPrice"
        )
    except Exception as e:
        print("SL/TP error:", e)


# ===============================
# 32️⃣ EXECUTE FULL TRADE
# ===============================

def execute_trade(trade):
    global TRADES_TODAY, OPEN_TRADES

    if not can_trade():
        return False

    order = place_order(trade)
    if not order:
        return False

    set_sl_tp(trade)

    TRADES_TODAY += 1
    OPEN_TRADES[trade["symbol"]] = trade

    tg(
        f"📥 TRADE OPENED ({MODE})\n"
        f"{trade['symbol']} | {trade['side']}\n"
        f"Entry: {trade['entry']}\n"
        f"SL: {trade['stop_loss']} | TP: {trade['take_profit']}\n"
        f"Size: {trade['size']}"
    )

    return True


# ===============================
# 33️⃣ BULK EXECUTION (100+ PAIRS)
# ===============================

def execute_trades(trades):
    for trade in trades:
        if trade["symbol"] in OPEN_TRADES:
            continue
        execute_trade(trade)


# ===============================
# 34️⃣ OPEN POSITION MONITOR
# ===============================

def monitor_positions():
    """
    Basic position sync
    """
    try:
        pos = session.get_positions(category="linear")
        active_symbols = [p["symbol"] for p in pos["result"]["list"] if float(p["size"]) > 0]

        for symbol in list(OPEN_TRADES.keys()):
            if symbol not in active_symbols:
                tg(f"✅ TRADE CLOSED: {symbol}")
                OPEN_TRADES.pop(symbol, None)

    except Exception as e:
        print("Monitor error:", e)

  # ======================================================
# PART 6 : AI TRADE FILTER & CONFIDENCE SCORE
# ======================================================

# ===============================
# 35️⃣ AI FILTER SETTINGS
# ===============================

MIN_CONFIDENCE = 0.65   # 65% በታች trade አይከፈት
WEIGHT_RSI = 0.30
WEIGHT_TREND = 0.30
WEIGHT_VOLUME = 0.20
WEIGHT_VOLATILITY = 0.20


# ===============================
# 36️⃣ RSI SCORE
# ===============================

def rsi_score(rsi, side):
    """
    Score RSI based on trade direction
    """
    if side == "LONG":
        if rsi < 30:
            return 1.0
        elif rsi < 40:
            return 0.7
        else:
            return 0.0
    else:
        if rsi > 70:
            return 1.0
        elif rsi > 60:
            return 0.7
        else:
            return 0.0


# ===============================
# 37️⃣ TREND SCORE
# ===============================

def trend_score(trend, side):
    """
    trend = BULL / BEAR / RANGE
    """
    if side == "LONG" and trend == "BULL":
        return 1.0
    if side == "SHORT" and trend == "BEAR":
        return 1.0
    if trend == "RANGE":
        return 0.3
    return 0.0


# ===============================
# 38️⃣ VOLUME SCORE
# ===============================

def volume_score(volume, avg_volume):
    if avg_volume == 0:
        return 0.0
    ratio = volume / avg_volume
    if ratio >= 1.5:
        return 1.0
    elif ratio >= 1.1:
        return 0.6
    else:
        return 0.2


# ===============================
# 39️⃣ VOLATILITY SCORE
# ===============================

def volatility_score(atr, price):
    if price == 0:
        return 0.0
    v = atr / price
    if 0.002 <= v <= 0.01:
        return 1.0
    elif v < 0.002:
        return 0.3
    else:
        return 0.5


# ===============================
# 40️⃣ AI CONFIDENCE ENGINE
# ===============================

def ai_confidence(trade):
    """
    trade must include:
    rsi, trend, volume, avg_volume, atr, price
    """
    s1 = rsi_score(trade["rsi"], trade["side"]) * WEIGHT_RSI
    s2 = trend_score(trade["trend"], trade["side"]) * WEIGHT_TREND
    s3 = volume_score(trade["volume"], trade["avg_volume"]) * WEIGHT_VOLUME
    s4 = volatility_score(trade["atr"], trade["price"]) * WEIGHT_VOLATILITY

    confidence = round(s1 + s2 + s3 + s4, 2)
    return confidence


# ===============================
# 41️⃣ AI TRADE FILTER
# ===============================

def ai_trade_filter(trades):
    """
    Filters trades using AI confidence score
    """
    approved = []

    for trade in trades:
        score = ai_confidence(trade)
        trade["confidence"] = score

        if score >= MIN_CONFIDENCE:
            approved.append(trade)
        else:
            print(
                f"❌ REJECTED {trade['symbol']} | "
                f"{trade['side']} | score={score}"
            )

    return approved


# ===============================
# 42️⃣ SMART EXECUTION PIPELINE
# ===============================

def smart_execute(trades):
    """
    AI → Risk → Execution
    """
    filtered = ai_trade_filter(trades)

    if not filtered:
        return

    execute_trades(filtered)

# ======================================================
# PART 7 : STRATEGY ENGINE (RSI + EMA + STRUCTURE)
# ======================================================

# ===============================
# 43️⃣ STRATEGY SETTINGS
# ===============================

RSI_PERIOD = 14
EMA_FAST = 21
EMA_SLOW = 50


# ===============================
# 44️⃣ RSI CALCULATION
# ===============================

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50

    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


# ===============================
# 45️⃣ EMA CALCULATION
# ===============================

def calculate_ema(closes, period):
    if len(closes) < period:
        return closes[-1]

    k = 2 / (period + 1)
    ema = closes[0]

    for price in closes[1:]:
        ema = price * k + ema * (1 - k)

    return round(ema, 4)


# ===============================
# 46️⃣ MARKET TREND DETECTION
# ===============================

def detect_trend(closes):
    ema_fast = calculate_ema(closes[-EMA_FAST:], EMA_FAST)
    ema_slow = calculate_ema(closes[-EMA_SLOW:], EMA_SLOW)

    if ema_fast > ema_slow:
        return "BULL"
    elif ema_fast < ema_slow:
        return "BEAR"
    else:
        return "RANGE"


# ===============================
# 47️⃣ MARKET STRUCTURE
# ===============================

def market_structure(highs, lows):
    if len(highs) < 3 or len(lows) < 3:
        return "RANGE"

    if highs[-1] > highs[-2] > highs[-3] and lows[-1] > lows[-2]:
        return "BULL"
    if highs[-1] < highs[-2] < highs[-3] and lows[-1] < lows[-2]:
        return "BEAR"

    return "RANGE"


# ===============================
# 48️⃣ STRATEGY SIGNAL
# ===============================

def strategy_signal(symbol, candles):
    """
    candles: list of dicts {open, high, low, close, volume}
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    rsi = calculate_rsi(closes)
    trend = detect_trend(closes)
    structure = market_structure(highs, lows)

    # ===========================
    # LONG CONDITION
    # ===========================
    if rsi < 35 and trend == "BULL" and structure != "BEAR":
        return {
            "symbol": symbol,
            "side": "LONG",
            "rsi": rsi,
            "trend": trend,
            "volume": volumes[-1],
            "avg_volume": sum(volumes[-20:]) / 20,
            "price": closes[-1]
        }

    # ===========================
    # SHORT CONDITION
    # ===========================
    if rsi > 65 and trend == "BEAR" and structure != "BULL":
        return {
            "symbol": symbol,
            "side": "SHORT",
            "rsi": rsi,
            "trend": trend,
            "volume": volumes[-1],
            "avg_volume": sum(volumes[-20:]) / 20,
            "price": closes[-1]
        }

    return None


# ===============================
# 49️⃣ MULTI-PAIR SCANNER
# ===============================

def scan_markets(symbols):
    signals = []

    for symbol in symbols:
        candles = fetch_candles(symbol)
        if not candles:
            continue

        signal = strategy_signal(symbol, candles)
        if signal:
            signals.append(signal)

    return signals

# ======================================================
# PART 7 : STRATEGY ENGINE (RSI + EMA + STRUCTURE)
# ======================================================

# ===============================
# 43️⃣ STRATEGY SETTINGS
# ===============================

RSI_PERIOD = 14
EMA_FAST = 21
EMA_SLOW = 50


# ===============================
# 44️⃣ RSI CALCULATION
# ===============================

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50

    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


# ===============================
# 45️⃣ EMA CALCULATION
# ===============================

def calculate_ema(closes, period):
    if len(closes) < period:
        return closes[-1]

    k = 2 / (period + 1)
    ema = closes[0]

    for price in closes[1:]:
        ema = price * k + ema * (1 - k)

    return round(ema, 4)


# ===============================
# 46️⃣ MARKET TREND DETECTION
# ===============================

def detect_trend(closes):
    ema_fast = calculate_ema(closes[-EMA_FAST:], EMA_FAST)
    ema_slow = calculate_ema(closes[-EMA_SLOW:], EMA_SLOW)

    if ema_fast > ema_slow:
        return "BULL"
    elif ema_fast < ema_slow:
        return "BEAR"
    else:
        return "RANGE"


# ===============================
# 47️⃣ MARKET STRUCTURE
# ===============================

def market_structure(highs, lows):
    if len(highs) < 3 or len(lows) < 3:
        return "RANGE"

    if highs[-1] > highs[-2] > highs[-3] and lows[-1] > lows[-2]:
        return "BULL"
    if highs[-1] < highs[-2] < highs[-3] and lows[-1] < lows[-2]:
        return "BEAR"

    return "RANGE"


# ===============================
# 48️⃣ STRATEGY SIGNAL
# ===============================

def strategy_signal(symbol, candles):
    """
    candles: list of dicts {open, high, low, close, volume}
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    rsi = calculate_rsi(closes)
    trend = detect_trend(closes)
    structure = market_structure(highs, lows)

    # ===========================
    # LONG CONDITION
    # ===========================
    if rsi < 35 and trend == "BULL" and structure != "BEAR":
        return {
            "symbol": symbol,
            "side": "LONG",
            "rsi": rsi,
            "trend": trend,
            "volume": volumes[-1],
            "avg_volume": sum(volumes[-20:]) / 20,
            "price": closes[-1]
        }

    # ===========================
    # SHORT CONDITION
    # ===========================
    if rsi > 65 and trend == "BEAR" and structure != "BULL":
        return {
            "symbol": symbol,
            "side": "SHORT",
            "rsi": rsi,
            "trend": trend,
            "volume": volumes[-1],
            "avg_volume": sum(volumes[-20:]) / 20,
            "price": closes[-1]
        }

    return None


# ===============================
# 49️⃣ MULTI-PAIR SCANNER
# ===============================

def scan_markets(symbols):
    signals = []

    for symbol in symbols:
        candles = fetch_candles(symbol)
        if not candles:
            continue

        signal = strategy_signal(symbol, candles)
        if signal:
            signals.append(signal)

    return signals

# ======================================================
# PART 9 : MINI WEB DASHBOARD (FLASK)
# Mobile-style UI + Telegram WebApp Ready
# ======================================================

from flask import Flask, jsonify, render_template_string
import threading

# ===============================
# 70️⃣ FLASK APP
# ===============================

app = Flask(__name__)

@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Super Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                background:#0f172a;
                color:white;
                font-family:Arial;
                padding:20px;
            }
            button {
                width:100%;
                padding:15px;
                margin:10px 0;
                font-size:18px;
                border:none;
                border-radius:10px;
            }
            .start { background:#22c55e; }
            .stop { background:#ef4444; }
        </style>
    </head>
    <body>
        <h2>🤖 Super Bot</h2>
        <button class="start" onclick="fetch('/api/start')">START BOT</button>
        <button class="stop" onclick="fetch('/api/stop')">STOP BOT</button>
        <pre id="status"></pre>

        <script>
        async function loadStatus(){
            let r = await fetch('/api/status');
            let d = await r.json();
            document.getElementById('status').innerText =
                JSON.stringify(d, null, 2);
        }
        loadStatus();
        setInterval(loadStatus, 5000);
        </script>
    </body>
    </html>
    """)

# ===============================
# 71️⃣ DASHBOARD DATA API
# ===============================

@app.route("/api/status")
def api_status():
    @app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Super Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                background:#0f172a;
                color:white;
                font-family:Arial;
                padding:20px;
            }
            button {
                width:100%;
                padding:15px;
                margin:10px 0;
                font-size:18px;
                border:none;
                border-radius:10px;
            }
            .start { background:#22c55e; }
            .stop { background:#ef4444; }
        </style>
    </head>
    <body>
        <h2>🤖 Super Bot</h2>
        <button class="start" onclick="fetch('/api/start')">START BOT</button>
        <button class="stop" onclick="fetch('/api/stop')">STOP BOT</button>
        <pre id="status"></pre>

        <script>
        async function loadStatus(){
            let r = await fetch('/api/status');
            let d = await r.json();
            document.getElementById('status').innerText =
                JSON.stringify(d, null, 2);
        }
        loadStatus();
        setInterval(loadStatus, 5000);
        </script>
    </body>
    </html>
    """)
    return jsonify({
        "mode": MODE,
        "balance": get_balance(),
        "bot_active": BOT_ACTIVE,
        "kill_switch": KILL_SWITCH,
        "trades_today": TRADES_TODAY,
        "open_trades": len(OPEN_TRADES),
        "leverage": LEVERAGE,
        "risk_per_trade": RISK_PER_TRADE
    })

# ===============================
# 72️⃣ MOBILE STYLE UI (HTML)
# ===============================

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SUPER BOT</title>
<style>
body {
    font-family: Arial;
    background: #0f172a;
    color: white;
    margin: 0;
    padding: 15px;
}
.card {
    background: #1e293b;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 12px;
}
.title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}
.value {
    font-size: 18px;
}
.green { color: #22c55e; }
.red { color: #ef4444; }
</style>
</head>
<body>

<div class="title">🤖 SUPER BOT DASHBOARD</div>

<div class="card">
    <div>Mode</div>
    <div class="value" id="mode">-</div>
</div>

<div class="card">
    <div>Balance</div>
    <div class="value" id="balance">-</div>
</div>

<div class="card">
    <div>Bot Status</div>
    <div class="value" id="bot">-</div>
</div>

<div class="card">
    <div>Kill Switch</div>
    <div class="value" id="kill">-</div>
</div>

<div class="card">
    <div>Trades Today</div>
    <div class="value" id="trades">-</div>
</div>

<script>
async function loadData(){
    const r = await fetch("/api/status");
    const d = await r.json();

    document.getElementById("mode").innerText = d.mode;
    document.getElementById("balance").innerText = d.balance;
    document.getElementById("bot").innerText = d.bot_active ? "ON ✅" : "OFF ❌";
    document.getElementById("kill").innerText = d.kill_switch ? "YES 🛑" : "NO 🟢";
    document.getElementById("trades").innerText = d.trades_today;
}

setInterval(loadData, 3000);
loadData();
</script>

</body>
</html>
"""

# ===============================
# 73️⃣ DASHBOARD ROUTE
# ===============================

@app.route("/")
def dashboard():
    return render_template_string(HTML_UI)

# ===============================
# 74️⃣ START WEB SERVER
# ===============================

def start_web():
    print("🌐 Web dashboard running...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

# ===============================
# 75️⃣ RUN EVERYTHING
# ===============================

def start_web():
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    init_day()

    # Telegram thread
    threading.Thread(target=start_telegram, daemon=True).start()

    # Trading loop thread
    threading.Thread(target=scan_markets, daemon=True).start()

    # Web dashboard (Flask)
    threading.Thread(target=start_web, daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(60)
