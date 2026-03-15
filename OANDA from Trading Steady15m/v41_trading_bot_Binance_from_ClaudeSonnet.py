# Fichier: config.py
# Créer ce fichier séparément avec vos identifiants
"""
BINANCE_API_KEY = "GwvK4xNzrBr1HpEozMWx2AnWevMRlUZxPgroPvDcy2l4tqdzBH1eBa6PSpriyQTb"
BINANCE_API_SECRET = "DGij2y3b9teZHM0vrqafkGzuKbtKCM4wtI3RpA96DqPFNo433HCUJhs4cnSI3MQq"
"""

# Fichier principal: trading_bot_binance.py

# Imports
import config
from binance import Client
from binance.enums import *
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone
import time
import os

# ==================== CONFIGURATION ====================

# Connexion Binance
client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

# Paramètres du bot (MODIFIABLES)
SYMBOL = "BTCUSDC"              # Paire de trading
BASE_ASSET = "BTC"              # Asset à trader
QUOTE_ASSET = "USDC"            # Asset de cotation
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE  # M1, M15, H1, etc.
POSITION_SIZE_PERCENT = 15      # % du solde USDC à utiliser par trade
BINANCE_FEE = 0.001             # 0.1% de frais par ordre
TP_RATIO = 1.5                  # Ratio Take Profit / Stop Loss

# Stratégies (activables individuellement)
ENABLE_EMA_CROSSOVER = False     # Stratégie EMA 5/8 crossover
ENABLE_CHOCH = True            # Stratégie CHoCH Smart Money

# Filtre EMA Trend Cloud (Pine Script EMA Trend 50 200 280)
ENABLE_EMA_TREND_CLOUD = False   # Filtre optionnel EMA Trend Cloud
EMA_TREND_FAST = 50              # EMA rapide (Fast)
EMA_TREND_MEDIUM = 200           # EMA moyenne (Medium)
EMA_TREND_SLOW = 280             # EMA lente (Slow)

# Paramètres CHoCH (optimisés pour M1)
CHOCH_SWING_SIZE = 12           # Structure principale (pivots majeurs de la trendline)
CHOCH_MICRO_SWING = 3           # Micro-structure (petit HL/LH avant cassure)
CHOCH_SEARCH_WINDOW = 25        # Fenêtre de recherche du micro-swing (en bougies)
CHOCH_CONF_TYPE = 'close'       # 'close' ou 'wick' pour confirmation BOS

# Auto-adaptation des intervalles de vérification selon timeframe
if TIMEFRAME == Client.KLINE_INTERVAL_1MINUTE:
    CHECK_INTERVAL_IN_POSITION = 10   # 10 secondes en position
    CHECK_INTERVAL_NO_POSITION = 60   # 1 minute sans position
elif TIMEFRAME == Client.KLINE_INTERVAL_15MINUTE:
    CHECK_INTERVAL_IN_POSITION = 30   # 30 secondes en position
    CHECK_INTERVAL_NO_POSITION = 60   # 1 minute sans position
elif TIMEFRAME == Client.KLINE_INTERVAL_1HOUR:
    CHECK_INTERVAL_IN_POSITION = 60   # 1 minute en position
    CHECK_INTERVAL_NO_POSITION = 300  # 5 minutes sans position
else:
    CHECK_INTERVAL_IN_POSITION = 10   # Défaut 10 secondes
    CHECK_INTERVAL_NO_POSITION = 60   # Défaut 1 minute

# Fichier de log
log_file = "trade_log_binance.csv"

# DataFrame de log en mémoire
trade_log = pd.DataFrame(columns=[
    'timestamp_open', 'day_of_week_name', 'day_of_week_number', 'hour', 'minute', 'second',
    'week_number', 'month', 'is_weekend',
    'signal', 'entry_price', 'sl', 'tp', 'quantity_btc',
    'buy_win', 'buy_loss', 'sell_win', 'sell_loss',
    'choch_bull', 'choch_bear', 'bos_a', 'choch_a',
    'cloud_up', 'cloud_dw',
    'timestamp_close', 'exit_price', 'pnl_usdc', 'pnl_percent',
    'duration_minutes', 'status', 'ema_5', 'ema_8', 'ema_30', 'atr_14', 'fees_usdc', 'notes'
])

# ==================== FONCTIONS UTILITAIRES ====================

def get_account_balance():
    """
    Récupère le solde USDC disponible
    """
    balance = float(client.get_asset_balance(asset=QUOTE_ASSET)['free'])
    return balance


def get_btc_balance():
    """
    Récupère le solde BTC disponible
    """
    balance = float(client.get_asset_balance(asset=BASE_ASSET)['free'])
    return balance


def get_current_price():
    """
    Récupère le prix actuel en temps réel
    API call léger pour monitoring de positions
    """
    ticker = client.get_symbol_ticker(symbol=SYMBOL)
    return float(ticker['price'])


# ==================== FONCTIONS DE LOGGING ====================

def init_log_file():
    """
    Initialise le fichier de log CSV
    Charge l'historique existant avec toutes les colonnes
    """
    global trade_log

    if os.path.exists(log_file):
        print(f"Loading existing log file: {log_file}")
        loaded_log = pd.read_csv(log_file)

        # S'assurer que toutes les colonnes nécessaires existent
        required_columns = [
            'timestamp_open', 'day_of_week_name', 'day_of_week_number', 'hour', 'minute', 'second',
            'week_number', 'month', 'is_weekend',
            'signal', 'entry_price', 'sl', 'tp', 'quantity_btc',
            'buy_win', 'buy_loss', 'sell_win', 'sell_loss',
            'choch_bull', 'choch_bear', 'bos_a', 'choch_a',
            'cloud_up', 'cloud_dw',
            'timestamp_close', 'exit_price', 'pnl_usdc', 'pnl_percent',
            'duration_minutes', 'status', 'ema_5', 'ema_8', 'ema_30', 'atr_14', 'fees_usdc', 'notes'
        ]

        # Ajouter les colonnes manquantes avec valeurs par défaut
        for col in required_columns:
            if col not in loaded_log.columns:
                if col in ['bos_a', 'choch_a', 'choch_bull', 'choch_bear', 'cloud_up', 'cloud_dw', 'day_of_week_number', 'hour', 'minute', 'second', 'week_number', 'month', 'is_weekend']:
                    loaded_log[col] = 0
                else:
                    loaded_log[col] = None

        trade_log = loaded_log[required_columns]
    else:
        print(f"Creating new log file: {log_file}")
        trade_log.to_csv(log_file, index=False)


def add_trade_entry(timestamp, signal, entry, sl, tp, qty, ema5, ema8, ema30, atr, choch_bull=0, choch_bear=0, bos_a=0, choch_a=0, cloud_up=0, cloud_dw=0, notes=""):
    """
    Ajoute un nouveau trade au log avec status OPEN
    Inclut les distances CHoCH/BOS et indicateurs BOS_A/CHoCH_A
    Inclut cloud_up et cloud_dw pour le filtre EMA Trend Cloud
    """
    global trade_log

    new_entry = {
        'timestamp_open': timestamp,
        'signal': signal,
        'entry_price': entry,
        'sl': sl,
        'tp': tp,
        'quantity_btc': qty,
        'buy_win': 0,
        'buy_loss': 0,
        'sell_win': 0,
        'sell_loss': 0,
        'choch_bull': choch_bull,
        'choch_bear': choch_bear,
        'bos_a': bos_a,
        'choch_a': choch_a,
        'cloud_up': cloud_up,
        'cloud_dw': cloud_dw,
        'timestamp_close': None,
        'exit_price': None,
        'pnl_usdc': None,
        'pnl_percent': None,
        'duration_minutes': None,
        'status': 'OPEN',
        'ema_5': ema5,
        'ema_8': ema8,
        'ema_30': ema30,
        'atr_14': atr,
        'fees_usdc': None,
        'notes': notes
    }

    trade_log = pd.concat([trade_log, pd.DataFrame([new_entry])], ignore_index=True)
    trade_log.to_csv(log_file, index=False)

    quality_info = ""
    if bos_a == 1:
        quality_info = " | ⭐ BOS_A (High Quality)"
    elif choch_a == 1:
        quality_info = " | ⭐⭐ CHoCH_A (Premium Quality)"

    choch_info = ""
    if choch_bull > 0:
        choch_info = f" | Distance: {choch_bull} candles"
    elif choch_bear > 0:
        choch_info = f" | Distance: {choch_bear} candles"

    print(f"✓ Trade logged: {signal} {qty:.6f} BTC at {entry:.2f} USDC (OPEN){choch_info}{quality_info}")


def close_trade(index, exit_price, result, current_time):
    """
    Ferme un trade et met à jour toutes les statistiques
    """
    global trade_log

    trade = trade_log.iloc[index]
    signal = trade['signal']
    entry = trade['entry_price']
    qty = trade['quantity_btc']

    # Réinitialiser les colonnes win/loss
    trade_log.at[index, 'buy_win'] = 0
    trade_log.at[index, 'buy_loss'] = 0
    trade_log.at[index, 'sell_win'] = 0
    trade_log.at[index, 'sell_loss'] = 0

    # Mettre à jour la colonne appropriée
    if signal == "BUY" and result == "WIN":
        trade_log.at[index, 'buy_win'] = 1
    elif signal == "BUY" and result == "LOSS":
        trade_log.at[index, 'buy_loss'] = 1
    elif signal == "SELL" and result == "WIN":
        trade_log.at[index, 'sell_win'] = 1
    elif signal == "SELL" and result == "LOSS":
        trade_log.at[index, 'sell_loss'] = 1

    # Calculer P&L en USDC
    if signal == "BUY":
        entry_value = entry * qty
        exit_value = exit_price * qty
        pnl_usdc = exit_value - entry_value
    else:  # SELL
        entry_value = entry * qty
        exit_value = exit_price * qty
        pnl_usdc = entry_value - exit_value

    # Calculer les frais (entrée + sortie)
    fees_total = (entry * qty * BINANCE_FEE) + (exit_price * qty * BINANCE_FEE)
    pnl_usdc -= fees_total

    # Calculer P&L en %
    pnl_percent = (pnl_usdc / (entry * qty)) * 100

    # Calculer la durée
    time_open = pd.to_datetime(trade['timestamp_open'])
    time_close = pd.to_datetime(current_time)
    duration = (time_close - time_open).total_seconds() / 60

    # Mettre à jour le trade
    trade_log.at[index, 'timestamp_close'] = current_time
    trade_log.at[index, 'exit_price'] = exit_price
    trade_log.at[index, 'pnl_usdc'] = pnl_usdc
    trade_log.at[index, 'pnl_percent'] = pnl_percent
    trade_log.at[index, 'fees_usdc'] = fees_total
    trade_log.at[index, 'duration_minutes'] = duration
    trade_log.at[index, 'status'] = f'CLOSED_{result}'

    trade_log.to_csv(log_file, index=False)

    symbol = "✅" if result == "WIN" else "❌"
    print(f"\n{symbol} {signal} CLOSED: {result}")
    print(f"   Entry: {entry:.2f} → Exit: {exit_price:.2f}")
    print(f"   P&L: {pnl_usdc:+.2f} USDC ({pnl_percent:+.2f}%)")
    print(f"   Duration: {duration:.1f} minutes")
    print(f"   Fees: {fees_total:.4f} USDC\n")


def check_open_trades_realtime(current_price, current_time):
    """
    Vérifie les trades ouverts avec le prix temps réel
    Plus précis et réactif que l'ancien système par chandelier
    """
    global trade_log

    open_trades = trade_log[trade_log['status'] == 'OPEN']

    if len(open_trades) == 0:
        return

    for idx, trade in open_trades.iterrows():
        signal = trade['signal']
        sl = trade['sl']
        tp = trade['tp']

        # Vérifier SL/TP avec prix actuel
        if signal == "BUY":
            if current_price >= tp:
                print(f"🎯 TP touched! {current_price:.2f} >= {tp:.2f}")
                close_trade(idx, tp, "WIN", current_time)
            elif current_price <= sl:
                print(f"🛑 SL touched! {current_price:.2f} <= {sl:.2f}")
                close_trade(idx, sl, "LOSS", current_time)

        elif signal == "SELL":
            if current_price <= tp:
                print(f"🎯 TP touched! {current_price:.2f} <= {tp:.2f}")
                close_trade(idx, tp, "WIN", current_time)
            elif current_price >= sl:
                print(f"🛑 SL touched! {current_price:.2f} >= {sl:.2f}")
                close_trade(idx, sl, "LOSS", current_time)


def analyze_performance():
    """
    Analyse de performance complète
    """
    closed_trades = trade_log[trade_log['status'].str.contains('CLOSED', na=False)]

    if len(closed_trades) == 0:
        return "No closed trades yet"

    total_trades = len(closed_trades)
    total_pnl = closed_trades['pnl_usdc'].sum()
    total_fees = closed_trades['fees_usdc'].sum()

    buy_wins = int(closed_trades['buy_win'].sum())
    buy_losses = int(closed_trades['buy_loss'].sum())
    total_buys = buy_wins + buy_losses
    buy_win_rate = (buy_wins / total_buys * 100) if total_buys > 0 else 0

    sell_wins = int(closed_trades['sell_win'].sum())
    sell_losses = int(closed_trades['sell_loss'].sum())
    total_sells = sell_wins + sell_losses
    sell_win_rate = (sell_wins / total_sells * 100) if total_sells > 0 else 0

    total_wins = buy_wins + sell_wins
    global_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    current_balance = get_account_balance()

    report = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           BINANCE BOT - PERFORMANCE REPORT               ║
    ╚══════════════════════════════════════════════════════════╝

    💰 ACCOUNT BALANCE
    ─────────────────────────────────────────────────────────
    Current USDC Balance: {current_balance:.2f} USDC

    📊 GLOBAL STATISTICS
    ─────────────────────────────────────────────────────────
    Total Trades: {total_trades}
    Wins: {total_wins} | Losses: {total_trades - total_wins}
    Win Rate: {global_win_rate:.1f}%

    Net P&L: {total_pnl:+.2f} USDC
    Total Fees Paid: {total_fees:.2f} USDC

    🔵 BUY TRADES
    ─────────────────────────────────────────────────────────
    Total: {total_buys} | Wins: {buy_wins} | Losses: {buy_losses}
    Win Rate: {buy_win_rate:.1f}%

    🔴 SELL TRADES
    ─────────────────────────────────────────────────────────
    Total: {total_sells} | Wins: {sell_wins} | Losses: {sell_losses}
    Win Rate: {sell_win_rate:.1f}%

    📈 OPEN POSITIONS
    ─────────────────────────────────────────────────────────
    Currently Open: {len(trade_log[trade_log['status'] == 'OPEN'])}
    """

    return report


# ==================== FONCTIONS DE TRADING ====================

def get_candles():
    """
    Récupère les chandeliers depuis Binance
    """
    klines = client.get_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=500)

    data = []
    for kline in klines:
        data.append({
            'time': pd.to_datetime(kline[0], unit='ms'),
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5])
        })

    df = pd.DataFrame(data)
    return df


def calculate_indicators(df):
    """
    Calcule les indicateurs techniques
    Inclut EMA Trend Cloud (EMA 50, 200, 280)
    """
    df['EMA_5'] = ta.ema(df['close'], length=5)
    df['EMA_8'] = ta.ema(df['close'], length=8)
    df['EMA_30'] = ta.ema(df['close'], length=30)  # EMA 30 pour stratégie Trader DNA
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # EMA Trend Cloud (Pine Script: EMA Trend 50 200 280)
    df['EMA_50'] = ta.ema(df['close'], length=EMA_TREND_FAST)
    df['EMA_200'] = ta.ema(df['close'], length=EMA_TREND_MEDIUM)
    df['EMA_280'] = ta.ema(df['close'], length=EMA_TREND_SLOW)

    return df


def detect_choch(df, swing_size=12, micro_swing=3, search_window=25, conf_type='close'):
    """
    Détecte les BOS (Break of Structure) et CHoCH (Change of Character)
    avec validation EMA 30 selon méthode Trader DNA (double swing)

    Architecture double swing :
    - swing_size : Structure principale (pivots majeurs de trendline)
    - micro_swing : Micro-structure (petit HL/LH juste avant cassure)
    - search_window : Fenêtre de recherche du micro-pivot

    BOS = Cassure de structure dans le sens de la tendance
    CHoCH = Cassure de structure CONTRE la tendance (plus rare, plus fort)
    BOS_A / CHoCH_A = Le petit HL/LH (micro) avant cassure est du bon côté de l'EMA 30

    Paramètres:
    - df: DataFrame avec colonnes 'high', 'low', 'close', 'EMA_30'
    - swing_size: Taille pour pivots majeurs (défaut: 12 pour M1)
    - micro_swing: Taille pour micro-pivots (défaut: 3 pour M1)
    - search_window: Fenêtre de recherche (défaut: 25 bougies)
    - conf_type: 'close' ou 'wick' pour confirmation

    Retourne:
    - (bull_distance, bear_distance, is_choch_bull, is_choch_bear, is_quality_bull, is_quality_bear)
    """

    if len(df) < swing_size * 2 + 1:
        return 0, 0, False, False, False, False

    # ========== PHASE 1: Détecter les PIVOTS MAJEURS (structure principale) ==========
    prev_high = None
    prev_low = None
    prev_high_index = None
    prev_low_index = None
    prev_breakout_dir = 0

    bull_distance = 0
    bear_distance = 0
    is_choch_bull = False
    is_choch_bear = False
    is_quality_bull = False
    is_quality_bear = False

    # Trouver les pivots majeurs avec swing_size
    for i in range(swing_size, len(df) - swing_size):
        # Pivot High majeur
        window_high = df['high'].iloc[i - swing_size:i + swing_size + 1]
        current_high = df['high'].iloc[i]

        if current_high == window_high.max():
            prev_high = current_high
            prev_high_index = i

        # Pivot Low majeur
        window_low = df['low'].iloc[i - swing_size:i + swing_size + 1]
        current_low = df['low'].iloc[i]

        if current_low == window_low.min():
            prev_low = current_low
            prev_low_index = i

    last_index = len(df) - 1

    # ========== PHASE 2: Vérifier CASSURE de structure principale ==========

    # Cassure BULLISH (high cassé)
    if prev_high is not None and prev_high_index is not None:
        if conf_type == 'close':
            price_to_check = df['close'].iloc[last_index]
        else:
            price_to_check = df['high'].iloc[last_index]

        if price_to_check > prev_high:
            bull_distance = last_index - prev_high_index

            # CHoCH si direction précédente était bearish
            if prev_breakout_dir == -1:
                is_choch_bull = True
            else:
                is_choch_bull = False

            # ========== PHASE 3: Chercher MICRO-PIVOT (petit HL) pour validation qualité ==========
            # Chercher dans les dernières 'search_window' bougies
            start_search = max(micro_swing, last_index - search_window)
            end_search = max(start_search + micro_swing, last_index - micro_swing)

            micro_pivots_low = []

            for i in range(start_search, end_search + 1):
                if i >= micro_swing and i < len(df) - micro_swing:
                    # Fenêtre pour micro-pivot
                    micro_window = df['low'].iloc[i - micro_swing:i + micro_swing + 1]
                    current_low = df['low'].iloc[i]

                    if current_low == micro_window.min():
                        micro_pivots_low.append((i, current_low))

            # Prendre le DERNIER micro-pivot LOW trouvé (le plus récent)
            if len(micro_pivots_low) > 0 and 'EMA_30' in df.columns:
                last_micro_hl_index, last_micro_hl_value = micro_pivots_low[-1]
                ema30_at_hl = df['EMA_30'].iloc[last_micro_hl_index]

                # Validation : HL > EMA_30
                if not pd.isna(ema30_at_hl) and last_micro_hl_value > ema30_at_hl:
                    is_quality_bull = True

            prev_breakout_dir = 1

    # Cassure BEARISH (low cassé)
    if prev_low is not None and prev_low_index is not None:
        if conf_type == 'close':
            price_to_check = df['close'].iloc[last_index]
        else:
            price_to_check = df['low'].iloc[last_index]

        if price_to_check < prev_low:
            bear_distance = last_index - prev_low_index

            # CHoCH si direction précédente était bullish
            if prev_breakout_dir == 1:
                is_choch_bear = True
            else:
                is_choch_bear = False

            # ========== PHASE 3: Chercher MICRO-PIVOT (petit LH) pour validation qualité ==========
            start_search = max(micro_swing, last_index - search_window)
            end_search = max(start_search + micro_swing, last_index - micro_swing)

            micro_pivots_high = []

            for i in range(start_search, end_search + 1):
                if i >= micro_swing and i < len(df) - micro_swing:
                    # Fenêtre pour micro-pivot
                    micro_window = df['high'].iloc[i - micro_swing:i + micro_swing + 1]
                    current_high = df['high'].iloc[i]

                    if current_high == micro_window.max():
                        micro_pivots_high.append((i, current_high))

            # Prendre le DERNIER micro-pivot HIGH trouvé (le plus récent)
            if len(micro_pivots_high) > 0 and 'EMA_30' in df.columns:
                last_micro_lh_index, last_micro_lh_value = micro_pivots_high[-1]
                ema30_at_lh = df['EMA_30'].iloc[last_micro_lh_index]

                # Validation : LH < EMA_30
                if not pd.isna(ema30_at_lh) and last_micro_lh_value < ema30_at_lh:
                    is_quality_bear = True

            prev_breakout_dir = -1

    return bull_distance, bear_distance, is_choch_bull, is_choch_bear, is_quality_bull, is_quality_bear


def place_buy_order(entry_price, sl_price, tp_price, ema5, ema8, ema30, atr, choch_bull=0, choch_bear=0, bos_a=0, choch_a=0, cloud_up=0, cloud_dw=0, strategy_note="BUY - Strategy"):
    """
    Place un ordre BUY simple
    Logger AVANT l'ordre pour capturer même les échecs
    Retourne (success, quantity) ou (False, 0)
    """
    try:
        balance_usdc = get_account_balance()
        position_usdc = balance_usdc * (POSITION_SIZE_PERCENT / 100)

        # Calculer quantité et forcer format décimal (pas scientifique)
        quantity_btc = position_usdc / entry_price
        quantity_str = f"{quantity_btc:.6f}"

        print(f"💰 Using {position_usdc:.2f} USDC to buy ~{quantity_str} BTC")

        # LOGGER AVANT l'ordre (pour capturer même si échec)
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        add_trade_entry(current_time, "BUY", entry_price, sl_price, tp_price,
                       float(quantity_str), ema5, ema8, ema30, atr, choch_bull, choch_bear, bos_a, choch_a, cloud_up, cloud_dw, strategy_note)

        # Ordre MARKET simple
        order = client.create_order(
            symbol=SYMBOL,
            side='BUY',
            type='MARKET',
            quantity=quantity_str
        )

        print(f"✓ BUY MARKET executed: {quantity_str} BTC")
        print(f"   Order ID: {order['orderId']}")

        return True, float(quantity_str)

    except Exception as e:
        print(f"✗ BUY order failed: {e}")
        print(f"   (Trade logged in CSV but order not executed)")
        return False, 0


def place_sell_order(entry_price, sl_price, tp_price, ema5, ema8, ema30, atr, choch_bull=0, choch_bear=0, bos_a=0, choch_a=0, cloud_up=0, cloud_dw=0, strategy_note="SELL - Strategy"):
    """
    Place un ordre SELL simple
    Logger AVANT l'ordre pour capturer même les échecs
    Retourne (success, quantity) ou (False, 0)
    """
    try:
        btc_balance = get_btc_balance()

        if btc_balance < 0.00001:
            print(f"✗ Insufficient BTC balance: {btc_balance:.8f} BTC")
            return False, 0

        # Utiliser 90% du BTC disponible
        quantity_btc = btc_balance * 0.9
        quantity_str = f"{quantity_btc:.6f}"

        print(f"💰 Selling {quantity_str} BTC")

        # LOGGER AVANT l'ordre (pour capturer même si échec)
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        add_trade_entry(current_time, "SELL", entry_price, sl_price, tp_price,
                       float(quantity_str), ema5, ema8, ema30, atr, choch_bull, choch_bear, bos_a, choch_a, cloud_up, cloud_dw, strategy_note)

        # Ordre MARKET simple
        order = client.create_order(
            symbol=SYMBOL,
            side='SELL',
            type='MARKET',
            quantity=quantity_str
        )

        print(f"✓ SELL MARKET executed: {quantity_str} BTC")
        print(f"   Order ID: {order['orderId']}")

        return True, float(quantity_str)

    except Exception as e:
        print(f"✗ SELL order failed: {e}")
        print(f"   (Trade logged in CSV but order not executed)")
        return False, 0


def ema_crossover(df):
    """
    Stratégie EMA crossover (indépendante)
    Génère signaux BUY/SELL sur croisement EMA 5/8
    Avec filtre optionnel EMA Trend Cloud
    """
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    ema5 = last_candle['EMA_5']
    ema8 = last_candle['EMA_8']
    ema30 = last_candle['EMA_30']
    atr = last_candle['ATR_14']
    entry_price = last_candle['close']

    # EMA Trend Cloud status
    ema_50 = last_candle.get('EMA_50', None)
    ema_200 = last_candle.get('EMA_200', None)
    cloud_up = 1 if (ema_50 is not None and ema_200 is not None and ema_50 > ema_200) else 0
    cloud_dw = 1 if (ema_50 is not None and ema_200 is not None and ema_50 < ema_200) else 0

    if ema_50 is not None and ema_200 is not None:
        cloud_status = "☁️ CLOUD UP (Bullish)" if cloud_up else ("☁️ CLOUD DOWN (Bearish)" if cloud_dw else "☁️ CLOUD NEUTRAL")
        print(f"   {cloud_status} | EMA50: {ema_50:.2f} | EMA200: {ema_200:.2f}")

    # Signal BUY
    if (last_candle['EMA_5'] > last_candle['EMA_8'] and
        prev_candle['EMA_5'] <= prev_candle['EMA_8']):

        # Filtre EMA Trend Cloud
        if ENABLE_EMA_TREND_CLOUD and cloud_up != 1:
            print(f"\n🔵 EMA CROSSOVER BUY SIGNAL detected but BLOCKED by Cloud filter (cloud_up={cloud_up})")
            return

        buy_sl = entry_price - atr
        stop_distance = entry_price - buy_sl
        buy_tp = entry_price + (stop_distance * TP_RATIO)

        print(f"\n🔵 EMA CROSSOVER BUY SIGNAL!")
        print(f"   Entry: {entry_price:.2f} | SL: {buy_sl:.2f} | TP: {buy_tp:.2f}")

        success, quantity = place_buy_order(entry_price, buy_sl, buy_tp, ema5, ema8, ema30, atr, 0, 0, 0, 0, cloud_up, cloud_dw, "BUY - EMA Crossover 5/8")

        if success:
            print("✅ EMA Crossover trade executed")
        else:
            print("⚠️  EMA Crossover trade logged but execution failed")

    # Signal SELL
    elif (last_candle['EMA_5'] < last_candle['EMA_8'] and
          prev_candle['EMA_5'] >= prev_candle['EMA_8']):

        # Filtre EMA Trend Cloud
        if ENABLE_EMA_TREND_CLOUD and cloud_dw != 1:
            print(f"\n🔴 EMA CROSSOVER SELL SIGNAL detected but BLOCKED by Cloud filter (cloud_dw={cloud_dw})")
            return

        sell_sl = entry_price + atr
        stop_distance = sell_sl - entry_price
        sell_tp = entry_price - (stop_distance * TP_RATIO)

        print(f"\n🔴 EMA CROSSOVER SELL SIGNAL!")
        print(f"   Entry: {entry_price:.2f} | SL: {sell_sl:.2f} | TP: {sell_tp:.2f}")

        success, quantity = place_sell_order(entry_price, sell_sl, sell_tp, ema5, ema8, ema30, atr, 0, 0, 0, 0, cloud_up, cloud_dw, "SELL - EMA Crossover 5/8")

        if success:
            print("✅ EMA Crossover trade executed")
        else:
            print("⚠️  EMA Crossover trade logged but execution failed")

    else:
        print("⚪ No EMA crossover detected")


def strategy_choch(df):
    """
    Stratégie BOS/CHoCH Smart Money (indépendante)
    Génère signaux sur BOS (Break of Structure) ET CHoCH (Change of Character)
    Avec détection de qualité supérieure (_A) selon méthode Trader DNA
    Utilise architecture double swing (structure principale + micro-structure)
    Avec filtre optionnel EMA Trend Cloud

    BOS = Cassure dans le sens de la tendance (plus fréquent)
    CHoCH = Cassure contre la tendance (plus rare, signal plus fort)
    BOS_A / CHoCH_A = Le petit HL/LH (micro) avant cassure est du bon côté de l'EMA 30 (qualité premium)
    """
    # Calculer les indicateurs basiques pour SL/TP
    last_candle = df.iloc[-1]
    ema5 = last_candle.get('EMA_5', 0)
    ema8 = last_candle.get('EMA_8', 0)
    ema30 = last_candle.get('EMA_30', 0)
    atr = last_candle.get('ATR_14', last_candle['high'] - last_candle['low'])
    entry_price = last_candle['close']

    # EMA Trend Cloud status
    ema_50 = last_candle.get('EMA_50', None)
    ema_200 = last_candle.get('EMA_200', None)
    cloud_up = 1 if (ema_50 is not None and ema_200 is not None and ema_50 > ema_200) else 0
    cloud_dw = 1 if (ema_50 is not None and ema_200 is not None and ema_50 < ema_200) else 0

    if ema_50 is not None and ema_200 is not None:
        cloud_status = "☁️ CLOUD UP (Bullish)" if cloud_up else ("☁️ CLOUD DOWN (Bearish)" if cloud_dw else "☁️ CLOUD NEUTRAL")
        print(f"   {cloud_status} | EMA50: {ema_50:.2f} | EMA200: {ema_200:.2f}")

    # Détecter BOS/CHoCH avec validation EMA 30 (double swing)
    bull_distance, bear_distance, is_choch_bull, is_choch_bear, is_quality_bull, is_quality_bear = detect_choch(
        df, CHOCH_SWING_SIZE, CHOCH_MICRO_SWING, CHOCH_SEARCH_WINDOW, CHOCH_CONF_TYPE
    )

    # Signal BUY si cassure bullish détectée (BOS ou CHoCH)
    if bull_distance > 0:
        # Filtre EMA Trend Cloud
        if ENABLE_EMA_TREND_CLOUD and cloud_up != 1:
            signal_type = "CHoCH" if is_choch_bull else "BOS"
            print(f"\n🔵 {signal_type} BULLISH SIGNAL detected but BLOCKED by Cloud filter (cloud_up={cloud_up})")
            return

        buy_sl = entry_price - atr
        stop_distance = entry_price - buy_sl
        buy_tp = entry_price + (stop_distance * TP_RATIO)

        # Déterminer le type de signal
        signal_type = "CHoCH" if is_choch_bull else "BOS"
        emoji = "🔄" if is_choch_bull else "📈"

        # Déterminer la qualité (avec _A si validé par EMA 30)
        quality_suffix = "_A" if is_quality_bull else ""
        quality_label = " (Premium Quality ⭐⭐)" if is_quality_bull else ""

        # Valeurs binaires pour les colonnes
        bos_a = 1 if (not is_choch_bull and is_quality_bull) else 0
        choch_a = 1 if (is_choch_bull and is_quality_bull) else 0

        # Créer la note pour le CSV
        cloud_note = " [CLOUD UP]" if cloud_up else (" [CLOUD DW]" if cloud_dw else "")
        strategy_note = f"BUY - {signal_type}{quality_suffix} Bullish (swing={CHOCH_SWING_SIZE}, micro={CHOCH_MICRO_SWING}){cloud_note}"

        print(f"\n🔵 {emoji} {signal_type}{quality_suffix} BULLISH SIGNAL!")
        print(f"   Type: {'Change of Character' if is_choch_bull else 'Break of Structure'}{quality_label}")
        print(f"   Structure: Main swing={CHOCH_SWING_SIZE}, Micro swing={CHOCH_MICRO_SWING}")
        print(f"   Distance: {bull_distance} candles from major swing")
        if is_quality_bull:
            print(f"   ✅ Quality Check: Micro HL > EMA 30 (Validated by Trader DNA method)")
        print(f"   Entry: {entry_price:.2f} | SL: {buy_sl:.2f} | TP: {buy_tp:.2f}")

        success, quantity = place_buy_order(entry_price, buy_sl, buy_tp, ema5, ema8, ema30, atr,
                                            bull_distance, 0, bos_a, choch_a, cloud_up, cloud_dw, strategy_note)

        if success:
            print(f"✅ {signal_type}{quality_suffix} Bullish trade executed")
        else:
            print(f"⚠️  {signal_type} trade logged but execution failed")

    # Signal SELL si cassure bearish détectée (BOS ou CHoCH)
    elif bear_distance > 0:
        # Filtre EMA Trend Cloud
        if ENABLE_EMA_TREND_CLOUD and cloud_dw != 1:
            signal_type = "CHoCH" if is_choch_bear else "BOS"
            print(f"\n🔴 {signal_type} BEARISH SIGNAL detected but BLOCKED by Cloud filter (cloud_dw={cloud_dw})")
            return

        sell_sl = entry_price + atr
        stop_distance = sell_sl - entry_price
        sell_tp = entry_price - (stop_distance * TP_RATIO)

        # Déterminer le type de signal
        signal_type = "CHoCH" if is_choch_bear else "BOS"
        emoji = "🔄" if is_choch_bear else "📉"

        # Déterminer la qualité (avec _A si validé par EMA 30)
        quality_suffix = "_A" if is_quality_bear else ""
        quality_label = " (Premium Quality ⭐⭐)" if is_quality_bear else ""

        # Valeurs binaires pour les colonnes
        bos_a = 1 if (not is_choch_bear and is_quality_bear) else 0
        choch_a = 1 if (is_choch_bear and is_quality_bear) else 0

        # Créer la note pour le CSV
        cloud_note = " [CLOUD UP]" if cloud_up else (" [CLOUD DW]" if cloud_dw else "")
        strategy_note = f"SELL - {signal_type}{quality_suffix} Bearish (swing={CHOCH_SWING_SIZE}, micro={CHOCH_MICRO_SWING}){cloud_note}"

        print(f"\n🔴 {emoji} {signal_type}{quality_suffix} BEARISH SIGNAL!")
        print(f"   Type: {'Change of Character' if is_choch_bear else 'Break of Structure'}{quality_label}")
        print(f"   Structure: Main swing={CHOCH_SWING_SIZE}, Micro swing={CHOCH_MICRO_SWING}")
        print(f"   Distance: {bear_distance} candles from major swing")
        if is_quality_bear:
            print(f"   ✅ Quality Check: Micro LH < EMA 30 (Validated by Trader DNA method)")
        print(f"   Entry: {entry_price:.2f} | SL: {sell_sl:.2f} | TP: {sell_tp:.2f}")

        success, quantity = place_sell_order(entry_price, sell_sl, sell_tp, ema5, ema8, ema30, atr,
                                             0, bear_distance, bos_a, choch_a, cloud_up, cloud_dw, strategy_note)

        if success:
            print(f"✅ {signal_type}{quality_suffix} Bearish trade executed")
        else:
            print(f"⚠️  {signal_type} trade logged but execution failed")

    else:
        print("⚪ No BOS/CHoCH detected")


def run_bot():
    """
    Boucle principale du bot avec deux modes :
    - MODE ACTIF : Position ouverte → monitoring temps réel
    - MODE PASSIF : Pas de position → recherche de signaux

    Pour arrêter le bot :
    - Ctrl+C (KeyboardInterrupt)
    - Ou créer un fichier 'STOP_BOT.txt' dans le dossier
    """
    print("=" * 60)
    print("🤖 BINANCE ALGORITHMIC TRADING BOT")
    print("=" * 60)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Position Size: {POSITION_SIZE_PERCENT}% of balance")
    print(f"Strategies: ", end="")
    strategies = []
    if ENABLE_EMA_CROSSOVER:
        strategies.append("EMA Crossover (5/8)")
    if ENABLE_CHOCH:
        strategies.append(f"CHoCH (swing={CHOCH_SWING_SIZE})")
    print(" + ".join(strategies) if strategies else "None")
    cloud_status = "ON" if ENABLE_EMA_TREND_CLOUD else "OFF"
    print(f"EMA Trend Cloud Filter: {cloud_status} (EMA {EMA_TREND_FAST}/{EMA_TREND_MEDIUM}/{EMA_TREND_SLOW})")
    print(f"Check interval IN position: {CHECK_INTERVAL_IN_POSITION}s")
    print(f"Check interval NO position: {CHECK_INTERVAL_NO_POSITION}s")
    print("=" * 60)
    print("⚠️  TO STOP THE BOT:")
    print("   Option 1: Press Ctrl+C")
    print("   Option 2: Create a file named 'STOP_BOT.txt' in this folder")
    print("=" * 60)

    init_log_file()

    balance = get_account_balance()
    print(f"💰 Initial USDC Balance: {balance:.2f} USDC")
    print("=" * 60)

    last_candle_check = None
    check_counter = 0

    while True:
        try:
            # Vérifier si fichier STOP existe
            if os.path.exists('STOP_BOT.txt'):
                print("\n\n🛑 STOP_BOT.txt detected - Shutting down gracefully...")
                os.remove('STOP_BOT.txt')  # Supprimer le fichier
                print(analyze_performance())
                print("\n✅ Bot stopped successfully")
                break

            current_time = datetime.now(timezone.utc)
            open_trades = trade_log[trade_log['status'] == 'OPEN']

            # ==================== MODE ACTIF ====================
            if len(open_trades) > 0:
                # Position ouverte → Monitoring temps réel
                current_price = get_current_price()

                trade = open_trades.iloc[0]
                signal = trade['signal']
                entry = trade['entry_price']
                sl = trade['sl']
                tp = trade['tp']

                pnl_current = ((current_price - entry) / entry * 100) if signal == "BUY" else ((entry - current_price) / entry * 100)

                print(f"📊 [{current_time.strftime('%H:%M:%S')}] {signal} Position | Price: {current_price:.2f} | P&L: {pnl_current:+.2f}% | SL: {sl:.2f} | TP: {tp:.2f}")

                # Vérifier SL/TP
                check_open_trades_realtime(current_price, current_time.strftime("%Y-%m-%d %H:%M:%S"))

                # Attendre intervalle court
                time.sleep(CHECK_INTERVAL_IN_POSITION)

            # ==================== MODE PASSIF ====================
            else:
                # Pas de position → Recherche de signaux sur nouveaux chandeliers

                # Déterminer l'intervalle selon timeframe
                if TIMEFRAME == Client.KLINE_INTERVAL_1MINUTE:
                    check_interval = 1
                elif TIMEFRAME == Client.KLINE_INTERVAL_15MINUTE:
                    check_interval = 15
                elif TIMEFRAME == Client.KLINE_INTERVAL_1HOUR:
                    check_interval = 60
                else:
                    check_interval = 1

                # Vérifier si nouveau chandelier
                if (current_time.minute % check_interval == 0 and
                    current_time.second < 10 and
                    last_candle_check != current_time.minute):

                    check_counter += 1
                    print(f"\n{'='*60}")
                    print(f"🔍 Searching for signals #{check_counter} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}")

                    # Récupérer données et calculer indicateurs
                    price = get_candles()
                    price = calculate_indicators(price)

                    # Exécuter les stratégies activées
                    if ENABLE_EMA_CROSSOVER:
                        ema_crossover(price)

                    if ENABLE_CHOCH:
                        strategy_choch(price)

                    # Rapport périodique
                    if check_counter % 20 == 0:
                        print(analyze_performance())

                    last_candle_check = current_time.minute

                # Attendre 1 seconde entre checks d'horloge
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n🛑 Ctrl+C detected - Stopping bot...")
            print(analyze_performance())
            print("\n✅ Bot stopped successfully")
            break
        except Exception as e:
            print(f"\n⚠️  Error in main loop: {e}")
            time.sleep(5)


# ==================== EXÉCUTION ====================

if __name__ == "__main__":
    run_bot()
