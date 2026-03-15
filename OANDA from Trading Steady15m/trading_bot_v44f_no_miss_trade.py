# trading_bot_v44f_no_miss_trade.py
# V44 - Price Log CSV + Chart PNG automatique à chaque trade
#
# AMELIORATIONS V44 vs V43 :
# 1. price_log_v44.csv : enregistrement de chaque tick de prix pendant un trade ouvert
#       colonnes : trade_id, timestamp, price, pnl_usdc, pnl_percent, signal, entry, sl, tp
# 2. Génération automatique d'un graphique PNG à l'ouverture de chaque trade :
#       - 100 dernières bougies M15 en chandeliers japonais
#       - EMA 50 / 200 / 250 tracées
#       - Point d'entrée (vert/rouge), SL (rouge), TP (vert) marqués
#       - Sauvegardé dans /charts/trade_YYYYMMDD_HHMMSS_BUY.png
#       - Compatible PythonAnywhere (matplotlib backend Agg, sans écran)
# 3. trade_id ajouté dans trade_log pour lier les deux CSV
#
# INSTALLATION requise sur PythonAnywhere (une seule fois) :
#   pip install mplfinance --user
#
# STRUCTURE des fichiers générés :
#   trade_log_binance_v44.csv   → résumé par trade (1 ligne par trade)
#   price_log_v44.csv           → détail tick par tick (N lignes par trade)
#   charts/                     → dossier des graphiques PNG
#     trade_20260220_063000_BUY_T001.png
#     trade_20260220_071500_BUY_T002.png
#     ...

# ==================== IMPORTS ====================

import config
from binance import Client
from binance.enums import *
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone
import time
import os

# matplotlib en mode non-interactif (compatible PythonAnywhere sans écran)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    print("⚠️  mplfinance non installé. Graphiques désactivés.")
    print("   Pour activer : pip install mplfinance --user")

# ==================== CONFIGURATION ====================

client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

SYMBOL               = "BTCUSDC"
BASE_ASSET           = "BTC"
QUOTE_ASSET          = "USDC"
TIMEFRAME            = Client.KLINE_INTERVAL_15MINUTE
POSITION_SIZE_PERCENT = 15
BINANCE_FEE          = 0.001
TP_RATIO             = 1.5

ENABLE_EMA_CROSSOVER = False
ENABLE_CHOCH         = True

CHOCH_SWING_SIZE     = 12
CHOCH_MICRO_SWING    = 3
CHOCH_SEARCH_WINDOW  = 25
CHOCH_CONF_TYPE      = 'close'

EMA_FAST   = 50
EMA_MEDIUM = 200
EMA_SLOW   = 250

BTC_PIP_SIZE = 0.1

# Nombre de bougies affichées sur le graphique
CHART_CANDLES = 100

# Dossier de sauvegarde des charts
CHARTS_DIR = "charts"

# Auto-adaptation intervalles
if TIMEFRAME == Client.KLINE_INTERVAL_1MINUTE:
    CHECK_INTERVAL_IN_POSITION = 10
    CHECK_INTERVAL_NO_POSITION = 60
elif TIMEFRAME == Client.KLINE_INTERVAL_15MINUTE:
    CHECK_INTERVAL_IN_POSITION = 30
    CHECK_INTERVAL_NO_POSITION = 60
elif TIMEFRAME == Client.KLINE_INTERVAL_1HOUR:
    CHECK_INTERVAL_IN_POSITION = 60
    CHECK_INTERVAL_NO_POSITION = 300
else:
    CHECK_INTERVAL_IN_POSITION = 10
    CHECK_INTERVAL_NO_POSITION = 60

# Fichiers de log
log_file       = "trade_log_binance_v44.csv"
price_log_file = "price_log_v44.csv"

# Compteur global de trades pour trade_id unique
trade_counter = 0

# DataFrames en mémoire
trade_log = pd.DataFrame(columns=[
    'trade_id',
    'timestamp_open', 'day_of_week_name', 'day_of_week_number', 'hour', 'minute', 'second',
    'week_number', 'month', 'is_weekend',
    'signal', 'entry_price', 'sl', 'tp', 'quantity_btc',
    'buy_win', 'buy_loss', 'sell_win', 'sell_loss',
    'choch_bull', 'choch_bear', 'bos_a', 'choch_a',
    'zone',
    'zone_abs',         # V44d : zone du plus bas absolu entre a et b
    'd1_price', 'd1_pips', 'd1_points',
    'd2_price', 'd2_pips', 'd2_points',
    'd3_candles',
    'ema_trend_cloud',
    'timestamp_close', 'exit_price', 'pnl_usdc', 'pnl_percent',
    'duration_minutes', 'status',
    'ema_5', 'ema_8', 'ema_30', 'ema_50', 'ema_200', 'ema_250',
    'atr_14', 'fees_usdc',
    'chart_file',   # V44 : nom du fichier PNG généré
    'notes'
])

price_log = pd.DataFrame(columns=[
    'trade_id',       # Lien avec trade_log
    'timestamp',      # Heure du tick
    'price',          # Prix actuel
    'pnl_usdc',       # P&L en USDC à cet instant
    'pnl_percent',    # P&L en % à cet instant
    'signal',         # BUY ou SELL
    'entry_price',    # Prix d'entrée du trade
    'sl',             # Stop Loss
    'tp',             # Take Profit
])


# ==================== FONCTIONS UTILITAIRES ====================

def get_account_balance():
    return float(client.get_asset_balance(asset=QUOTE_ASSET)['free'])

def get_btc_balance():
    return float(client.get_asset_balance(asset=BASE_ASSET)['free'])

def get_current_price():
    return float(client.get_symbol_ticker(symbol=SYMBOL)['price'])


# ==================== EMA TREND CLOUD ====================

def calculate_ema_trend_cloud(df):
    df[f'EMA_{EMA_FAST}']   = ta.ema(df['close'], length=EMA_FAST)
    df[f'EMA_{EMA_MEDIUM}'] = ta.ema(df['close'], length=EMA_MEDIUM)
    df[f'EMA_{EMA_SLOW}']   = ta.ema(df['close'], length=EMA_SLOW)
    df['ema_cloud_up']   = df[f'EMA_{EMA_FAST}'] > df[f'EMA_{EMA_SLOW}']
    df['ema_cloud_down'] = df[f'EMA_{EMA_FAST}'] < df[f'EMA_{EMA_SLOW}']
    return df


def find_last_ema_crossover(df):
    """
    Détection par changement de signe de (EMA50 - EMA200).
    Robuste aux arrondis flottants.
    """
    col_fast = f'EMA_{EMA_FAST}'
    col_med  = f'EMA_{EMA_MEDIUM}'

    for i in range(len(df) - 1, 0, -1):
        curr_diff = df[col_fast].iloc[i]     - df[col_med].iloc[i]
        prev_diff = df[col_fast].iloc[i - 1] - df[col_med].iloc[i - 1]

        if pd.isna(curr_diff) or pd.isna(prev_diff):
            continue
        if prev_diff <= 0 and curr_diff > 0:
            return i, 'GOLDEN'
        if prev_diff >= 0 and curr_diff < 0:
            return i, 'DEATH'

    return None, None


def determine_zone(point_c_price, ema50_at_c, ema200_at_c, is_bullish):
    if pd.isna(ema50_at_c) or pd.isna(ema200_at_c) or pd.isna(point_c_price):
        return 0
    if is_bullish:
        if point_c_price >= ema50_at_c:  return 1
        elif point_c_price >= ema200_at_c: return 2
        else: return 3
    else:
        if point_c_price <= ema50_at_c:  return 1
        elif point_c_price <= ema200_at_c: return 2
        else: return 3


def compute_d1_d2(df, point_a_index, point_b_index):
    col_fast = f'EMA_{EMA_FAST}'
    col_med  = f'EMA_{EMA_MEDIUM}'

    def _dist(idx):
        if idx is None or idx >= len(df):
            return {'price': 0.0, 'pips': 0.0, 'points': 0.0}
        e50  = df[col_fast].iloc[idx]
        e200 = df[col_med].iloc[idx]
        if pd.isna(e50) or pd.isna(e200):
            return {'price': 0.0, 'pips': 0.0, 'points': 0.0}
        dist_price = abs(e50 - e200)
        return {
            'price':  round(dist_price, 2),
            'pips':   round(dist_price / BTC_PIP_SIZE, 1),
            'points': round(dist_price, 2)
        }

    return _dist(point_a_index), _dist(point_b_index)


def compute_d3(point_a_index, point_e_index):
    if point_a_index is None or point_e_index is None:
        return 0
    return abs(point_a_index - point_e_index)


# ==================== V44 : PRICE LOG ====================

def init_price_log():
    """
    Initialise le fichier price_log_v44.csv.
    Charge l'historique existant ou crée un nouveau fichier.
    """
    global price_log
    if os.path.exists(price_log_file):
        price_log = pd.read_csv(price_log_file)
        print(f"Loading existing price log: {price_log_file} ({len(price_log)} ticks)")
    else:
        price_log.to_csv(price_log_file, index=False)
        print(f"Creating new price log: {price_log_file}")


def log_price_tick(trade_id, current_price, current_time, signal, entry_price, sl, tp, qty):
    """
    Enregistre un tick de prix dans price_log_v44.csv.
    Appelé à chaque itération du mode ACTIF (position ouverte).
    """
    global price_log

    # Calcul P&L instantané (sans frais, juste pour monitoring)
    if signal == "BUY":
        pnl_usdc    = (current_price - entry_price) * qty
        pnl_percent = (current_price - entry_price) / entry_price * 100
    else:
        pnl_usdc    = (entry_price - current_price) * qty
        pnl_percent = (entry_price - current_price) / entry_price * 100

    new_tick = {
        'trade_id':    trade_id,
        'timestamp':   current_time,
        'price':       current_price,
        'pnl_usdc':    round(pnl_usdc, 6),
        'pnl_percent': round(pnl_percent, 4),
        'signal':      signal,
        'entry_price': entry_price,
        'sl':          sl,
        'tp':          tp,
    }

    price_log = pd.concat([price_log, pd.DataFrame([new_tick])], ignore_index=True)

    # Sauvegarde CSV — append ligne par ligne pour éviter la perte en cas de crash
    pd.DataFrame([new_tick]).to_csv(price_log_file, mode='a', header=False, index=False)


# ==================== V44 : CHART PNG ====================

def generate_trade_chart(df, trade_id, signal, entry_price, sl_price, tp_price,
                          ema_cloud, zone, timestamp_str,
                          point_a_index=None, point_b_index=None,
                          point_c_index=None, point_c_price=None,
                          point_e_index=None):
    """
    Génère un graphique PNG du trade au moment du signal.
    Utilise mplfinance si disponible, sinon matplotlib basique.

    Paramètres :
    - df              : DataFrame complet avec colonnes OHLCV + EMAs
    - trade_id        : identifiant du trade (ex: T001)
    - signal          : 'BUY' ou 'SELL'
    - entry_price     : prix d'entrée
    - sl_price        : stop loss
    - tp_price        : take profit
    - ema_cloud       : 'UP' ou 'DOWN'
    - zone            : 1, 2 ou 3
    - timestamp_str   : horodatage pour le nom de fichier
    - point_a_index   : index absolu dans df du pivot majeur (point a) — triangle rouge
    - point_b_index   : index absolu dans df de la bougie BOS (point b) — triangle vert/rouge
    - point_c_index   : index absolu dans df du micro-pivot (point c) — cercle jaune
    - point_c_price   : prix du point c (pour positionner le marqueur)
    - point_e_index   : index absolu dans df du croisement EMA50/200 (point e) — losange blanc

    Retourne le chemin du fichier PNG créé, ou None si erreur.
    """

    # Créer le dossier charts s'il n'existe pas
    if not os.path.exists(CHARTS_DIR):
        os.makedirs(CHARTS_DIR)
        print(f"📁 Created charts directory: {CHARTS_DIR}/")

    # Nom du fichier PNG
    ts_clean   = timestamp_str.replace(" ", "_").replace(":", "").replace("-", "")[:15]
    chart_name = f"trade_{ts_clean}_{signal}_{trade_id}.png"
    chart_path = os.path.join(CHARTS_DIR, chart_name)

    # Prendre les CHART_CANDLES dernières bougies
    df_chart = df.tail(CHART_CANDLES).copy()
    df_chart = df_chart.reset_index(drop=True)

    # Convertir les index absolus en index relatifs dans df_chart
    # df_chart contient les indices [len(df)-CHART_CANDLES .. len(df)-1] du df original
    offset = len(df) - CHART_CANDLES  # index absolu de la première bougie affichée

    def to_relative(abs_idx):
        """Convertit un index absolu df -> index relatif df_chart. None si hors fenetre."""
        if abs_idx is None:
            return None
        rel = abs_idx - offset
        if 0 <= rel < CHART_CANDLES:
            return rel
        return None  # hors fenêtre d'affichage

    rel_a = to_relative(point_a_index)
    rel_b = to_relative(point_b_index)   # = dernière bougie (CHART_CANDLES-1) normalement
    rel_c = to_relative(point_c_index)
    rel_e = to_relative(point_e_index)

    try:
        if MPLFINANCE_AVAILABLE:
            _generate_chart_mplfinance(df_chart, chart_path, trade_id, signal,
                                        entry_price, sl_price, tp_price, ema_cloud, zone,
                                        rel_a, rel_b, rel_c, point_c_price, rel_e)
        else:
            _generate_chart_matplotlib(df_chart, chart_path, trade_id, signal,
                                        entry_price, sl_price, tp_price, ema_cloud, zone,
                                        rel_a, rel_b, rel_c, point_c_price, rel_e)

        print(f"📸 Chart saved: {chart_path}")
        return chart_path

    except Exception as e:
        print(f"⚠️  Chart generation failed: {e}")
        return None


def _generate_chart_mplfinance(df_chart, chart_path, trade_id, signal,
                                 entry_price, sl_price, tp_price, ema_cloud, zone,
                                 rel_a=None, rel_b=None, rel_c=None,
                                 point_c_price=None, rel_e=None):
    """
    Génération avec mplfinance + marqueurs des points a, b, c, e.

    Marqueurs :
    - Point a : triangle ROUGE  au-dessus de la bougie (pivot majeur cassé)
    - Point b : triangle VERT (BUY) ou ROUGE (SELL) en-dessous (bougie BOS)
    - Point c : cercle JAUNE  au niveau du micro-pivot HL/LH
    - Point e : losange BLANC au niveau du croisement EMA50/EMA200
    """
    import numpy as np

    n = len(df_chart)

    # mplfinance requiert un DatetimeIndex
    df_mpf = df_chart.set_index('time')
    df_mpf.index = pd.DatetimeIndex(df_mpf.index)

    # EMAs
    ema50_line  = df_chart[f'EMA_{EMA_FAST}'].values
    ema200_line = df_chart[f'EMA_{EMA_MEDIUM}'].values
    ema250_line = df_chart[f'EMA_{EMA_SLOW}'].values

    # ── Séries de marqueurs (NaN partout sauf à l'index voulu) ──
    # mplfinance addplot utilise des séries de même longueur que df
    # Les NaN sont invisibles, seule la valeur non-NaN est tracée

    # Point a — triangle rouge AU-DESSUS du high de la bougie
    markers_a = [float('nan')] * n
    if rel_a is not None:
        # Placer le marqueur légèrement au-dessus du high
        markers_a[rel_a] = df_chart['high'].iloc[rel_a] * 1.0008

    # Point b — triangle vert/rouge EN-DESSOUS du low (bougie BOS = dernière bougie)
    markers_b = [float('nan')] * n
    if rel_b is not None:
        markers_b[rel_b] = df_chart['low'].iloc[rel_b] * 0.9992

    # Point c — cercle jaune au prix du micro-pivot
    markers_c = [float('nan')] * n
    if rel_c is not None and point_c_price is not None:
        markers_c[rel_c] = point_c_price

    # Point e — losange blanc au niveau de EMA50 au moment du croisement
    markers_e = [float('nan')] * n
    if rel_e is not None:
        markers_e[rel_e] = ema50_line[rel_e] if not pd.isna(ema50_line[rel_e]) else float('nan')

    # ── Addplots ──
    apds = [
        mpf.make_addplot(ema50_line,  color='blue',   width=1.2),
        mpf.make_addplot(ema200_line, color='orange',  width=1.2),
        mpf.make_addplot(ema250_line, color='purple',  width=1.2),
    ]

    # Ajouter les marqueurs seulement si au moins une valeur non-NaN existe
    import numpy as np
    b_color = 'lime' if signal == 'BUY' else 'red'

    if any(not pd.isna(v) for v in markers_a):
        apds.append(mpf.make_addplot(markers_a, type='scatter',
                                     markersize=120, marker='v', color='red'))
    if any(not pd.isna(v) for v in markers_b):
        apds.append(mpf.make_addplot(markers_b, type='scatter',
                                     markersize=120, marker='^' if signal=='BUY' else 'v',
                                     color=b_color))
    if any(not pd.isna(v) for v in markers_c):
        apds.append(mpf.make_addplot(markers_c, type='scatter',
                                     markersize=100, marker='o', color='yellow'))
    if any(not pd.isna(v) for v in markers_e):
        apds.append(mpf.make_addplot(markers_e, type='scatter',
                                     markersize=100, marker='D', color='white'))

    # Lignes horizontales Entry / SL / TP
    hlines = dict(
        hlines=[entry_price, sl_price, tp_price],
        colors=['cyan', 'red', 'lime'],
        linestyle=['--', '--', '--'],
        linewidths=[1.5, 1.5, 1.5]
    )

    cloud_emoji = "UP" if ema_cloud == 'UP' else "DOWN"
    zone_labels = {1: "Zone 1 (above EMA50)", 2: "Zone 2 (EMA50-EMA200)", 3: "Zone 3 (below EMA200)"}
    zone_label  = zone_labels.get(zone, "Zone ?")

    # Légende des points dans le titre
    pts_info = []
    if rel_a is not None: pts_info.append(f"a={rel_a}")
    if rel_b is not None: pts_info.append(f"b={rel_b}")
    if rel_c is not None: pts_info.append(f"c={rel_c}")
    if rel_e is not None: pts_info.append(f"e={rel_e}")
    pts_str = " | " + " ".join(pts_info) if pts_info else ""

    title = (f"{SYMBOL} {TIMEFRAME} | {signal} {trade_id} | "
             f"Entry:{entry_price:.2f} SL:{sl_price:.2f} TP:{tp_price:.2f}\n"
             f"Cloud:{cloud_emoji} | {zone_label}{pts_str}\n"
             f"[RED▼=a pivot] [YELLOW●=c micro] [WHITE◆=e cross] [{b_color}▲=b BOS]")

    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        rc={'font.size': 8}
    )

    mpf.plot(
        df_mpf,
        type='candle',
        style=style,
        title=title,
        addplot=apds,
        hlines=hlines,
        figsize=(18, 10),
        savefig=dict(fname=chart_path, dpi=120, bbox_inches='tight')
    )
    plt.close('all')


def _generate_chart_matplotlib(df_chart, chart_path, trade_id, signal,
                                 entry_price, sl_price, tp_price, ema_cloud, zone,
                                 rel_a=None, rel_b=None, rel_c=None,
                                 point_c_price=None, rel_e=None):
    """
    Fallback matplotlib avec marqueurs des points a, b, c, e.
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    x = list(range(len(df_chart)))

    # Prix (ligne close)
    ax.plot(x, df_chart['close'].values, color='white', linewidth=1, label='Close')

    # EMAs
    col50  = f'EMA_{EMA_FAST}'
    col200 = f'EMA_{EMA_MEDIUM}'
    col250 = f'EMA_{EMA_SLOW}'
    if col50  in df_chart.columns: ax.plot(x, df_chart[col50].values,  color='blue',   linewidth=1.2, label=f'EMA{EMA_FAST}')
    if col200 in df_chart.columns: ax.plot(x, df_chart[col200].values, color='orange', linewidth=1.2, label=f'EMA{EMA_MEDIUM}')
    if col250 in df_chart.columns: ax.plot(x, df_chart[col250].values, color='purple', linewidth=1.2, label=f'EMA{EMA_SLOW}')

    # Niveaux Entry / SL / TP
    ax.axhline(y=entry_price, color='cyan', linestyle='--', linewidth=1.5, label=f'Entry {entry_price:.2f}')
    ax.axhline(y=sl_price,    color='red',  linestyle='--', linewidth=1.5, label=f'SL {sl_price:.2f}')
    ax.axhline(y=tp_price,    color='lime', linestyle='--', linewidth=1.5, label=f'TP {tp_price:.2f}')

    b_color = 'lime' if signal == 'BUY' else 'red'

    # Point a — triangle rouge au-dessus du high (pivot majeur cassé)
    if rel_a is not None:
        y_a = df_chart['high'].iloc[rel_a] * 1.0008
        ax.scatter([rel_a], [y_a], color='red', s=200, marker='v', zorder=6, label='a (pivot)')
        ax.annotate('a', xy=(rel_a, y_a), color='red', fontsize=9, fontweight='bold',
                    ha='center', va='bottom')

    # Point b — triangle vert/rouge sous le low (bougie BOS)
    if rel_b is not None:
        y_b = df_chart['low'].iloc[rel_b] * 0.9992
        ax.scatter([rel_b], [y_b], color=b_color, s=200,
                   marker='^' if signal=='BUY' else 'v', zorder=6, label='b (BOS)')
        ax.annotate('b', xy=(rel_b, y_b), color=b_color, fontsize=9, fontweight='bold',
                    ha='center', va='top')

    # Point c — cercle jaune au prix du micro-pivot
    if rel_c is not None and point_c_price is not None:
        ax.scatter([rel_c], [point_c_price], color='yellow', s=180, marker='o',
                   zorder=6, label='c (micro)')
        ax.annotate('c', xy=(rel_c, point_c_price), color='yellow', fontsize=9,
                    fontweight='bold', ha='center', va='bottom')

    # Point e — losange blanc au niveau EMA50 (croisement)
    if rel_e is not None and col50 in df_chart.columns:
        y_e = df_chart[col50].iloc[rel_e]
        if not pd.isna(y_e):
            ax.scatter([rel_e], [y_e], color='white', s=180, marker='D',
                       zorder=6, label='e (EMA cross)')
            ax.annotate('e', xy=(rel_e, y_e), color='white', fontsize=9,
                        fontweight='bold', ha='center', va='bottom')

    zone_labels = {1: "Zone 1 (above EMA50)", 2: "Zone 2 (EMA50-EMA200)", 3: "Zone 3 (below EMA200)"}
    zone_label  = zone_labels.get(zone, "Zone ?")

    ax.set_title(
        f"{SYMBOL} {TIMEFRAME} | {signal} {trade_id} | "
        f"Entry:{entry_price:.2f} SL:{sl_price:.2f} TP:{tp_price:.2f}\n"
        f"Cloud:{ema_cloud} | {zone_label}\n"
        f"[RED v=a pivot] [YELLOW o=c micro] [WHITE D=e cross] [{b_color} ^=b BOS]",
        color='white', fontsize=9
    )
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')

    legend = ax.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close('all')


# ==================== FONCTIONS DE LOGGING ====================

def init_log_file():
    global trade_log, trade_counter

    if os.path.exists(log_file):
        print(f"Loading existing log file: {log_file}")
        loaded_log = pd.read_csv(log_file)

        required_columns = list(trade_log.columns)
        for col in required_columns:
            if col not in loaded_log.columns:
                if col in ['bos_a', 'choch_a', 'choch_bull', 'choch_bear',
                           'day_of_week_number', 'hour', 'minute', 'second',
                           'week_number', 'month', 'is_weekend', 'zone', 'zone_abs', 'd3_candles']:
                    loaded_log[col] = 0
                elif col in ['d1_price', 'd1_pips', 'd1_points',
                             'd2_price', 'd2_pips', 'd2_points']:
                    loaded_log[col] = 0.0
                else:
                    loaded_log[col] = None

        trade_log = loaded_log[required_columns]

        # Reprendre le compteur depuis le dernier trade_id enregistré
        if 'trade_id' in trade_log.columns and len(trade_log) > 0:
            try:
                last_id = trade_log['trade_id'].dropna().iloc[-1]
                trade_counter = int(str(last_id).replace('T', ''))
                print(f"   Resuming trade counter at {trade_counter}")
            except Exception:
                trade_counter = 0
    else:
        print(f"Creating new log file: {log_file}")
        trade_log.to_csv(log_file, index=False)


def generate_trade_id():
    """Génère un identifiant unique de trade : T001, T002, ..."""
    global trade_counter
    trade_counter += 1
    return f"T{trade_counter:03d}"


def add_trade_entry(trade_id, timestamp, signal, entry, sl, tp, qty,
                    ema5, ema8, ema30, ema50, ema200, ema250, atr,
                    choch_bull=0, choch_bear=0, bos_a=0, choch_a=0,
                    zone=0, zone_abs=0,
                    d1_price=0.0, d1_pips=0.0, d1_points=0.0,
                    d2_price=0.0, d2_pips=0.0, d2_points=0.0,
                    d3_candles=0, ema_trend_cloud=None,
                    chart_file=None, notes=""):
    global trade_log

    ts = pd.to_datetime(timestamp)
    new_entry = {
        'trade_id':              trade_id,
        'timestamp_open':        timestamp,
        'day_of_week_name':      ts.strftime('%A'),
        'day_of_week_number':    ts.dayofweek,
        'hour':                  ts.hour,
        'minute':                ts.minute,
        'second':                ts.second,
        'week_number':           ts.isocalendar()[1],
        'month':                 ts.month,
        'is_weekend':            1 if ts.dayofweek >= 5 else 0,
        'signal':                signal,
        'entry_price':           entry,
        'sl':                    sl,
        'tp':                    tp,
        'quantity_btc':          qty,
        'buy_win':  0, 'buy_loss':  0,
        'sell_win': 0, 'sell_loss': 0,
        'choch_bull': choch_bull, 'choch_bear': choch_bear,
        'bos_a': bos_a, 'choch_a': choch_a,
        'zone':          zone,
        'zone_abs':      zone_abs,
        'd1_price':      d1_price,  'd1_pips':  d1_pips,  'd1_points':  d1_points,
        'd2_price':      d2_price,  'd2_pips':  d2_pips,  'd2_points':  d2_points,
        'd3_candles':    d3_candles,
        'ema_trend_cloud': ema_trend_cloud,
        'timestamp_close': None, 'exit_price': None,
        'pnl_usdc': None, 'pnl_percent': None,
        'duration_minutes': None, 'status': 'OPEN',
        'ema_5': ema5, 'ema_8': ema8, 'ema_30': ema30,
        'ema_50': ema50, 'ema_200': ema200, 'ema_250': ema250,
        'atr_14': atr, 'fees_usdc': None,
        'chart_file': chart_file,
        'notes': notes
    }

    trade_log = pd.concat([trade_log, pd.DataFrame([new_entry])], ignore_index=True)
    trade_log.to_csv(log_file, index=False)

    quality_info = ""
    if bos_a == 1:   quality_info = " | ⭐ BOS_A"
    elif choch_a == 1: quality_info = " | ⭐⭐ CHoCH_A"

    zone_labels = {1: "Zone 1 (above EMA50)", 2: "Zone 2 (EMA50-EMA200)",
                   3: "Zone 3 (below EMA200)", 0: "Zone unknown"}

    chart_info = f" | 📸 {os.path.basename(chart_file)}" if chart_file else ""

    print(f"✓ [{trade_id}] Trade logged: {signal} {qty:.6f} BTC at {entry:.2f} USDC (OPEN)"
          f"{quality_info} | {zone_labels.get(zone, '?')} | "
          f"D1={d1_price:.1f}$ D2={d2_price:.1f}$ D3={d3_candles}c"
          f"{chart_info}")


def close_trade(index, exit_price, result, current_time):
    global trade_log

    trade  = trade_log.iloc[index]
    signal = trade['signal']
    entry  = trade['entry_price']
    qty    = trade['quantity_btc']

    trade_log.at[index, 'buy_win']   = 0
    trade_log.at[index, 'buy_loss']  = 0
    trade_log.at[index, 'sell_win']  = 0
    trade_log.at[index, 'sell_loss'] = 0

    if signal == "BUY"  and result == "WIN":  trade_log.at[index, 'buy_win']   = 1
    if signal == "BUY"  and result == "LOSS": trade_log.at[index, 'buy_loss']  = 1
    if signal == "SELL" and result == "WIN":  trade_log.at[index, 'sell_win']  = 1
    if signal == "SELL" and result == "LOSS": trade_log.at[index, 'sell_loss'] = 1

    pnl_usdc = (exit_price - entry) * qty if signal == "BUY" else (entry - exit_price) * qty
    fees_total = (entry * qty * BINANCE_FEE) + (exit_price * qty * BINANCE_FEE)
    pnl_usdc  -= fees_total
    pnl_percent = (pnl_usdc / (entry * qty)) * 100

    time_open  = pd.to_datetime(trade['timestamp_open'])
    time_close = pd.to_datetime(current_time)
    duration   = (time_close - time_open).total_seconds() / 60

    trade_log.at[index, 'timestamp_close']  = current_time
    trade_log.at[index, 'exit_price']       = exit_price
    trade_log.at[index, 'pnl_usdc']         = pnl_usdc
    trade_log.at[index, 'pnl_percent']      = pnl_percent
    trade_log.at[index, 'fees_usdc']        = fees_total
    trade_log.at[index, 'duration_minutes'] = duration
    trade_log.at[index, 'status']           = f'CLOSED_{result}'

    trade_log.to_csv(log_file, index=False)

    symbol = "✅" if result == "WIN" else "❌"
    tid    = trade.get('trade_id', '?')
    print(f"\n{symbol} [{tid}] {signal} CLOSED: {result}")
    print(f"   Entry: {entry:.2f} → Exit: {exit_price:.2f}")
    print(f"   P&L: {pnl_usdc:+.2f} USDC ({pnl_percent:+.2f}%)")
    print(f"   Duration: {duration:.1f} minutes | Fees: {fees_total:.4f} USDC\n")


def check_open_trades_realtime(current_price, current_time):
    global trade_log
    open_trades = trade_log[trade_log['status'] == 'OPEN']
    if len(open_trades) == 0:
        return

    for idx, trade in open_trades.iterrows():
        signal = trade['signal']
        sl, tp = trade['sl'], trade['tp']
        entry  = trade['entry_price']
        qty    = trade['quantity_btc']
        tid    = trade.get('trade_id', '?')

        # --- V44 : enregistrer le tick dans price_log ---
        log_price_tick(tid, current_price, current_time,
                       signal, entry, sl, tp, qty)

        # Vérifier SL/TP
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
    closed_trades = trade_log[trade_log['status'].str.contains('CLOSED', na=False)]
    if len(closed_trades) == 0:
        return "No closed trades yet"

    total_trades = len(closed_trades)
    total_pnl    = closed_trades['pnl_usdc'].sum()
    total_fees   = closed_trades['fees_usdc'].sum()

    buy_wins   = int(closed_trades['buy_win'].sum())
    buy_losses = int(closed_trades['buy_loss'].sum())
    total_buys = buy_wins + buy_losses
    buy_wr     = (buy_wins / total_buys * 100) if total_buys > 0 else 0

    sell_wins   = int(closed_trades['sell_win'].sum())
    sell_losses = int(closed_trades['sell_loss'].sum())
    total_sells = sell_wins + sell_losses
    sell_wr     = (sell_wins / total_sells * 100) if total_sells > 0 else 0

    total_wins      = buy_wins + sell_wins
    global_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    current_balance = get_account_balance()

    zone_stats = ""
    for z in [1, 2, 3]:
        z_trades = closed_trades[closed_trades['zone'] == z]
        if len(z_trades) > 0:
            z_wins = int(z_trades['buy_win'].sum()) + int(z_trades['sell_win'].sum())
            zone_stats += f"    Zone {z}: {len(z_trades)} trades | WR: {z_wins/len(z_trades)*100:.1f}%\n"

    total_ticks = len(price_log)

    return f"""
    ╔══════════════════════════════════════════════════════════╗
    ║        BINANCE BOT V44 - PERFORMANCE REPORT              ║
    ╚══════════════════════════════════════════════════════════╝

    💰 ACCOUNT BALANCE
    ─────────────────────────────────────────────────────────
    Current USDC Balance: {current_balance:.2f} USDC

    📊 GLOBAL STATISTICS
    ─────────────────────────────────────────────────────────
    Total Trades: {total_trades} | Wins: {total_wins} | Losses: {total_trades - total_wins}
    Win Rate: {global_win_rate:.1f}%
    Net P&L: {total_pnl:+.2f} USDC | Total Fees: {total_fees:.2f} USDC

    🔵 BUY : {total_buys} trades | WR: {buy_wr:.1f}%
    🔴 SELL: {total_sells} trades | WR: {sell_wr:.1f}%

    📍 ZONE ANALYSIS
    ─────────────────────────────────────────────────────────
{zone_stats}
    📋 LOGS
    ─────────────────────────────────────────────────────────
    Price log ticks : {total_ticks}
    Charts folder   : {CHARTS_DIR}/
    Open positions  : {len(trade_log[trade_log['status'] == 'OPEN'])}
    """


# ==================== FONCTIONS DE TRADING ====================

def get_candles():
    klines = client.get_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=500)
    data = []
    for k in klines:
        data.append({
            'time':   pd.to_datetime(k[0], unit='ms'),
            'open':   float(k[1]),
            'high':   float(k[2]),
            'low':    float(k[3]),
            'close':  float(k[4]),
            'volume': float(k[5])
        })
    return pd.DataFrame(data)


def calculate_indicators(df):
    df['EMA_5']  = ta.ema(df['close'], length=5)
    df['EMA_8']  = ta.ema(df['close'], length=8)
    df['EMA_30'] = ta.ema(df['close'], length=30)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df = calculate_ema_trend_cloud(df)
    return df


def detect_choch(df, swing_size=12, micro_swing=3, search_window=25, conf_type='close'):
    point_a_bull_index = None
    point_a_bear_index = None
    point_c_bull_price = None
    point_c_bull_index = None
    point_c_bear_price = None
    point_c_bear_index = None

    if len(df) < swing_size * 2 + 1:
        return (0, 0, False, False, False, False,
                None, None, None, None, None, None)

    prev_high = prev_low = None
    prev_high_index = prev_low_index = None
    prev_breakout_dir = 0
    bull_distance = bear_distance = 0
    is_choch_bull = is_choch_bear = False
    is_quality_bull = is_quality_bear = False

    for i in range(swing_size, len(df) - swing_size):
        window_high = df['high'].iloc[i - swing_size:i + swing_size + 1]
        if df['high'].iloc[i] == window_high.max():
            prev_high = df['high'].iloc[i]
            prev_high_index = i
        window_low = df['low'].iloc[i - swing_size:i + swing_size + 1]
        if df['low'].iloc[i] == window_low.min():
            prev_low = df['low'].iloc[i]
            prev_low_index = i

    last_index = len(df) - 1

    if prev_high is not None:
        price_check = df['close'].iloc[last_index] if conf_type == 'close' else df['high'].iloc[last_index]
        if price_check > prev_high:
            bull_distance = last_index - prev_high_index
            is_choch_bull = (prev_breakout_dir == -1)
            point_a_bull_index = prev_high_index
            start = max(micro_swing, last_index - search_window)
            end   = max(start + micro_swing, last_index - micro_swing)
            micro_lows = []
            for i in range(start, end + 1):
                if i >= micro_swing and i < len(df) - micro_swing:
                    mw = df['low'].iloc[i - micro_swing:i + micro_swing + 1]
                    if df['low'].iloc[i] == mw.min():
                        micro_lows.append((i, df['low'].iloc[i]))
            if micro_lows and 'EMA_30' in df.columns:
                mi, mv = micro_lows[-1]
                ema30v = df['EMA_30'].iloc[mi]
                if not pd.isna(ema30v) and mv > ema30v:
                    is_quality_bull = True
                point_c_bull_price = mv
                point_c_bull_index = mi
            prev_breakout_dir = 1

    if prev_low is not None:
        price_check = df['close'].iloc[last_index] if conf_type == 'close' else df['low'].iloc[last_index]
        if price_check < prev_low:
            bear_distance = last_index - prev_low_index
            is_choch_bear = (prev_breakout_dir == 1)
            point_a_bear_index = prev_low_index
            start = max(micro_swing, last_index - search_window)
            end   = max(start + micro_swing, last_index - micro_swing)
            micro_highs = []
            for i in range(start, end + 1):
                if i >= micro_swing and i < len(df) - micro_swing:
                    mw = df['high'].iloc[i - micro_swing:i + micro_swing + 1]
                    if df['high'].iloc[i] == mw.max():
                        micro_highs.append((i, df['high'].iloc[i]))
            if micro_highs and 'EMA_30' in df.columns:
                mi, mv = micro_highs[-1]
                ema30v = df['EMA_30'].iloc[mi]
                if not pd.isna(ema30v) and mv < ema30v:
                    is_quality_bear = True
                point_c_bear_price = mv
                point_c_bear_index = mi
            prev_breakout_dir = -1

    return (bull_distance, bear_distance,
            is_choch_bull, is_choch_bear,
            is_quality_bull, is_quality_bear,
            point_a_bull_index, point_a_bear_index,
            point_c_bull_price, point_c_bull_index,
            point_c_bear_price, point_c_bear_index)


def compute_zone_abs(df, point_a_index, point_b_index, is_bullish):
    """
    V44d — Zone du plus bas/haut ABSOLU entre point a et point b.

    Bullish : cherche le LOW absolu minimum entre a et b
              → compare ce prix avec EMA50 et EMA200 à cet index
    Bearish : cherche le HIGH absolu maximum entre a et b
              → compare ce prix avec EMA50 et EMA200 à cet index

    Retourne 1, 2, 3 ou 0 si données insuffisantes.
    Zone bullish :
      1 : plus bas absolu >= EMA50   (au-dessus de EMA50)
      2 : EMA200 <= plus bas < EMA50 (entre EMA200 et EMA50)
      3 : plus bas < EMA200          (sous EMA200)
    Zone bearish (miroir) :
      1 : plus haut absolu <= EMA50
      2 : EMA50 < plus haut <= EMA200
      3 : plus haut > EMA200
    """
    col50  = f'EMA_{EMA_FAST}'
    col200 = f'EMA_{EMA_MEDIUM}'

    if point_a_index is None or point_b_index is None:
        return 0
    if point_a_index >= len(df) or point_b_index >= len(df):
        return 0

    # Fenêtre entre a et b (inclusif)
    idx_start = min(point_a_index, point_b_index)
    idx_end   = max(point_a_index, point_b_index)
    window    = df.iloc[idx_start:idx_end + 1]

    if len(window) == 0:
        return 0

    if is_bullish:
        # Plus bas absolu (low minimum) entre a et b
        abs_idx_in_window = window['low'].idxmin()
        abs_price = window['low'].min()
    else:
        # Plus haut absolu (high maximum) entre a et b
        abs_idx_in_window = window['high'].idxmax()
        abs_price = window['high'].max()

    # Lire EMA50 et EMA200 à cet index
    e50  = df[col50].iloc[abs_idx_in_window]  if col50  in df.columns else float('nan')
    e200 = df[col200].iloc[abs_idx_in_window] if col200 in df.columns else float('nan')

    if pd.isna(e50) or pd.isna(e200) or pd.isna(abs_price):
        return 0

    return determine_zone(abs_price, e50, e200, is_bullish)


# ==================== ORDRES ====================

def place_buy_order(df, entry_price, sl_price, tp_price,
                    ema5, ema8, ema30, ema50, ema200, ema250, atr,
                    choch_bull=0, choch_bear=0, bos_a=0, choch_a=0,
                    zone=0, zone_abs=0,
                    d1_price=0.0, d1_pips=0.0, d1_points=0.0,
                    d2_price=0.0, d2_pips=0.0, d2_points=0.0,
                    d3_candles=0, ema_trend_cloud=None,
                    strategy_note="BUY", **kwargs):
    try:
        balance_usdc  = get_account_balance()
        position_usdc = balance_usdc * (POSITION_SIZE_PERCENT / 100)
        quantity_btc  = position_usdc / entry_price
        quantity_str  = f"{quantity_btc:.6f}"

        print(f"💰 Using {position_usdc:.2f} USDC to buy ~{quantity_str} BTC")

        # Générer l'ID et le timestamp AVANT l'ordre
        trade_id     = generate_trade_id()
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # --- V44 : générer le chart PNG (avec points a/b/c/e passés en kwargs) ---
        chart_path = generate_trade_chart(
            df, trade_id, "BUY", entry_price, sl_price, tp_price,
            ema_trend_cloud or 'N/A', zone, current_time,
            point_a_index=kwargs.get('point_a_index'),
            point_b_index=kwargs.get('point_b_index'),
            point_c_index=kwargs.get('point_c_index'),
            point_c_price=kwargs.get('point_c_price_val'),
            point_e_index=kwargs.get('point_e_index'),
        )

        # Logger le trade
        add_trade_entry(
            trade_id, current_time, "BUY", entry_price, sl_price, tp_price,
            float(quantity_str), ema5, ema8, ema30, ema50, ema200, ema250, atr,
            choch_bull, choch_bear, bos_a, choch_a,
            zone, zone_abs, d1_price, d1_pips, d1_points,
            d2_price, d2_pips, d2_points, d3_candles,
            ema_trend_cloud, chart_path, strategy_note
        )

        order = client.create_order(
            symbol=SYMBOL, side='BUY', type='MARKET', quantity=quantity_str
        )
        print(f"✓ BUY MARKET executed: {quantity_str} BTC | Order ID: {order['orderId']}")
        return True, float(quantity_str)

    except Exception as e:
        print(f"✗ BUY order failed: {e}")
        return False, 0


def place_sell_order(df, entry_price, sl_price, tp_price,
                     ema5, ema8, ema30, ema50, ema200, ema250, atr,
                     choch_bull=0, choch_bear=0, bos_a=0, choch_a=0,
                     zone=0, zone_abs=0,
                     d1_price=0.0, d1_pips=0.0, d1_points=0.0,
                     d2_price=0.0, d2_pips=0.0, d2_points=0.0,
                     d3_candles=0, ema_trend_cloud=None,
                     strategy_note="SELL", **kwargs):
    try:
        btc_balance = get_btc_balance()
        if btc_balance < 0.00001:
            print(f"✗ Insufficient BTC balance: {btc_balance:.8f} BTC")
            return False, 0

        quantity_btc = btc_balance * 0.9
        quantity_str = f"{quantity_btc:.6f}"

        print(f"💰 Selling {quantity_str} BTC")

        trade_id     = generate_trade_id()
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # --- V44 : générer le chart PNG (avec points a/b/c/e passés en kwargs) ---
        chart_path = generate_trade_chart(
            df, trade_id, "SELL", entry_price, sl_price, tp_price,
            ema_trend_cloud or 'N/A', zone, current_time,
            point_a_index=kwargs.get('point_a_index'),
            point_b_index=kwargs.get('point_b_index'),
            point_c_index=kwargs.get('point_c_index'),
            point_c_price=kwargs.get('point_c_price_val'),
            point_e_index=kwargs.get('point_e_index'),
        )

        add_trade_entry(
            trade_id, current_time, "SELL", entry_price, sl_price, tp_price,
            float(quantity_str), ema5, ema8, ema30, ema50, ema200, ema250, atr,
            choch_bull, choch_bear, bos_a, choch_a,
            zone, zone_abs, d1_price, d1_pips, d1_points,
            d2_price, d2_pips, d2_points, d3_candles,
            ema_trend_cloud, chart_path, strategy_note
        )

        order = client.create_order(
            symbol=SYMBOL, side='SELL', type='MARKET', quantity=quantity_str
        )
        print(f"✓ SELL MARKET executed: {quantity_str} BTC | Order ID: {order['orderId']}")
        return True, float(quantity_str)

    except Exception as e:
        print(f"✗ SELL order failed: {e}")
        return False, 0


# ==================== STRATEGIES ====================

def ema_crossover(df):
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    ema5  = last['EMA_5'];  ema8  = last['EMA_8']
    ema30 = last['EMA_30']; ema50 = last[f'EMA_{EMA_FAST}']
    ema200= last[f'EMA_{EMA_MEDIUM}']; ema250= last[f'EMA_{EMA_SLOW}']
    atr   = last['ATR_14']
    entry = last['close']
    cloud = 'UP' if last.get('ema_cloud_up', False) else 'DOWN'

    if last['EMA_5'] > last['EMA_8'] and prev['EMA_5'] <= prev['EMA_8']:
        buy_sl = entry - atr
        buy_tp = entry + (entry - buy_sl) * TP_RATIO
        print(f"\n🔵 EMA CROSSOVER BUY SIGNAL!")
        place_buy_order(df, entry, buy_sl, buy_tp, ema5, ema8, ema30, ema50, ema200, ema250, atr,
                        ema_trend_cloud=cloud, strategy_note="BUY - EMA Crossover 5/8")

    elif last['EMA_5'] < last['EMA_8'] and prev['EMA_5'] >= prev['EMA_8']:
        sell_sl = entry + atr
        sell_tp = entry - (sell_sl - entry) * TP_RATIO
        print(f"\n🔴 EMA CROSSOVER SELL SIGNAL!")
        place_sell_order(df, entry, sell_sl, sell_tp, ema5, ema8, ema30, ema50, ema200, ema250, atr,
                         ema_trend_cloud=cloud, strategy_note="SELL - EMA Crossover 5/8")
    else:
        print("⚪ No EMA crossover detected")


def strategy_choch(df):
    last  = df.iloc[-1]
    last_index = len(df) - 1

    ema5  = last.get('EMA_5',  0);  ema8  = last.get('EMA_8',  0)
    ema30 = last.get('EMA_30', 0);  ema50 = last.get(f'EMA_{EMA_FAST}',   0)
    ema200= last.get(f'EMA_{EMA_MEDIUM}', 0); ema250= last.get(f'EMA_{EMA_SLOW}', 0)
    atr   = last.get('ATR_14', last['high'] - last['low'])
    entry = last['close']
    cloud = 'UP' if last.get('ema_cloud_up', False) else 'DOWN'

    (bull_distance, bear_distance,
     is_choch_bull, is_choch_bear,
     is_quality_bull, is_quality_bear,
     point_a_bull_index, point_a_bear_index,
     point_c_bull_price, point_c_bull_index,
     point_c_bear_price, point_c_bear_index) = detect_choch(
        df, CHOCH_SWING_SIZE, CHOCH_MICRO_SWING, CHOCH_SEARCH_WINDOW, CHOCH_CONF_TYPE
    )

    point_e_index, cross_type = find_last_ema_crossover(df)

    # ============================================================
    # FILTRES DE TENDANCE V44c
    # BUY  : EMA50 > EMA200 > EMA250 ET points a, b, c tous > EMA50
    # SELL : EMA50 < EMA200 < EMA250 ET points a, b, c tous < EMA50
    # ============================================================

    col50  = f'EMA_{EMA_FAST}'
    col200 = f'EMA_{EMA_MEDIUM}'
    col250 = f'EMA_{EMA_SLOW}'

    def ema50_at(idx):
        """Valeur de EMA50 à l'index idx, ou None si NaN/invalide."""
        if idx is None or idx >= len(df): return None
        v = df[col50].iloc[idx]
        return None if pd.isna(v) else v

    def check_buy_filters():
        """
        Retourne (True, "") si tous les filtres BUY sont validés.
        Retourne (False, "raison") sinon.
        V44c — Conditions :
          1. Cloud UP    : EMA50 > EMA250
          2. Tendance    : EMA50 > EMA200
          3. Point a     : prix du pivot (high) > EMA50 à l'index a
          4. Point b     : close de la bougie BOS > EMA50 à l'index b
          5. Point c     : prix du micro-pivot HL > EMA50 à l'index c  (= Zone 1 only)
        """
        # 1. Cloud UP
        if ema50 <= ema250:
            return False, f"Cloud DOWN (EMA50={ema50:.2f} <= EMA250={ema250:.2f})"
        # 2. EMA50 > EMA200
        if ema50 <= ema200:
            return False, f"EMA50={ema50:.2f} <= EMA200={ema200:.2f}"

        # 3. Point a au-dessus de EMA50
        e50_a = ema50_at(point_a_bull_index)
        if e50_a is None:
            return False, "Point a introuvable"
        price_a = df['high'].iloc[point_a_bull_index]
        if price_a <= e50_a:
            return False, f"Point a ({price_a:.2f}) <= EMA50 à a ({e50_a:.2f})"

        # 4. Point b (bougie BOS = last_index) close > EMA50
        e50_b = ema50_at(last_index)
        if e50_b is None:
            return False, "EMA50 indisponible à b"
        price_b = df['close'].iloc[last_index]
        if price_b <= e50_b:
            return False, f"Point b close ({price_b:.2f}) <= EMA50 à b ({e50_b:.2f})"

        # 5. Point c (micro-pivot HL) — Zone 1 ou 2 requises
        # Si point c absent : filtre ignoré, trade validé sur filtres 1-4
        if point_c_bull_price is not None and point_c_bull_index is not None:
            e50_c = ema50_at(point_c_bull_index)
            if e50_c is not None:
                e200_c = df[f'EMA_{EMA_MEDIUM}'].iloc[point_c_bull_index]
                if pd.isna(e200_c) or point_c_bull_price < e200_c:
                    return False, f"Point c ({point_c_bull_price:.2f}) < EMA200 à c ({e200_c:.2f}) — Zone 3 rejeté"
        # else : pas de point c → filtre 5 ignoré, trade accepté

        return True, "OK"

    def check_sell_filters():
        """
        Retourne (True, "") si tous les filtres SELL sont validés.
        V44c — Conditions (miroir inversé du BUY) :
          1. Cloud DOWN  : EMA50 < EMA250
          2. Tendance    : EMA50 < EMA200
          3. Point a     : prix du pivot (low) < EMA50 à l'index a
          4. Point b     : close de la bougie BOS < EMA50 à l'index b
          5. Point c     : prix du micro-pivot LH < EMA50 à l'index c  (= Zone 1 bearish only)
        """
        # 1. Cloud DOWN
        if ema50 >= ema250:
            return False, f"Cloud UP (EMA50={ema50:.2f} >= EMA250={ema250:.2f})"
        # 2. EMA50 < EMA200
        if ema50 >= ema200:
            return False, f"EMA50={ema50:.2f} >= EMA200={ema200:.2f}"

        # 3. Point a en-dessous de EMA50
        e50_a = ema50_at(point_a_bear_index)
        if e50_a is None:
            return False, "Point a introuvable"
        price_a = df['low'].iloc[point_a_bear_index]
        if price_a >= e50_a:
            return False, f"Point a ({price_a:.2f}) >= EMA50 à a ({e50_a:.2f})"

        # 4. Point b close < EMA50
        e50_b = ema50_at(last_index)
        if e50_b is None:
            return False, "EMA50 indisponible à b"
        price_b = df['close'].iloc[last_index]
        if price_b >= e50_b:
            return False, f"Point b close ({price_b:.2f}) >= EMA50 à b ({e50_b:.2f})"

        # 5. Point c (micro-pivot LH) — Zone 1 ou 2 requises
        # Si point c absent : filtre ignoré, trade validé sur filtres 1-4
        if point_c_bear_price is not None and point_c_bear_index is not None:
            e50_c = ema50_at(point_c_bear_index)
            if e50_c is not None:
                e200_c = df[f'EMA_{EMA_MEDIUM}'].iloc[point_c_bear_index]
                if pd.isna(e200_c) or point_c_bear_price > e200_c:
                    return False, f"Point c ({point_c_bear_price:.2f}) > EMA200 à c ({e200_c:.2f}) — Zone 3 rejeté"
        # else : pas de point c → filtre 5 ignoré, trade accepté

        return True, "OK"

    # ---- BUY ----
    if bull_distance > 0:
        buy_ok, buy_reason = check_buy_filters()

        if not buy_ok:
            print(f"⛔ BUY signal ignoré — filtre V44c : {buy_reason}")
        else:
            buy_sl = entry - atr
            buy_tp = entry + (entry - buy_sl) * TP_RATIO

            stype  = "CHoCH" if is_choch_bull else "BOS"
            qsufx  = "_A"   if is_quality_bull else ""
            bos_a  = 1 if (not is_choch_bull and is_quality_bull) else 0
            choch_a= 1 if (    is_choch_bull and is_quality_bull) else 0

            zone_val = 0
            if point_c_bull_price is not None and point_c_bull_index is not None:
                zone_val = determine_zone(
                    point_c_bull_price,
                    df[col50].iloc[point_c_bull_index],
                    df[col200].iloc[point_c_bull_index],
                    is_bullish=True
                )

            d1, d2 = compute_d1_d2(df, point_a_bull_index, last_index)
            d3     = compute_d3(point_a_bull_index, point_e_index)

            # V44d : zone du plus bas absolu entre a et b
            zone_abs_val = compute_zone_abs(df, point_a_bull_index, last_index, is_bullish=True)

            print(f"\n🔵 {'🔄' if is_choch_bull else '📈'} {stype}{qsufx} BULLISH SIGNAL! ✅ Tous filtres OK")
            print(f"   Cloud: {cloud} | Zone: {zone_val} | EMA50>{ema50:.0f} > EMA200>{ema200:.0f} > EMA250>{ema250:.0f}")
            print(f"   D1={d1['price']:.1f}$ ({d1['pips']:.0f}p) | D2={d2['price']:.1f}$ ({d2['pips']:.0f}p) | D3={d3}c [{cross_type}]")
            print(f"   Entry: {entry:.2f} | SL: {buy_sl:.2f} | TP: {buy_tp:.2f}")

            place_buy_order(
                df, entry, buy_sl, buy_tp,
                ema5, ema8, ema30, ema50, ema200, ema250, atr,
                bull_distance, 0, bos_a, choch_a,
                zone_val, zone_abs_val,
                d1['price'], d1['pips'], d1['points'],
                d2['price'], d2['pips'], d2['points'],
                d3, cloud,
                f"BUY - {stype}{qsufx} Bullish (swing={CHOCH_SWING_SIZE}, micro={CHOCH_MICRO_SWING})",
                point_a_index=point_a_bull_index,
                point_b_index=last_index,
                point_c_index=point_c_bull_index,
                point_c_price_val=point_c_bull_price,
                point_e_index=point_e_index,
            )

    # ---- SELL ----
    elif bear_distance > 0:
        sell_ok, sell_reason = check_sell_filters()

        if not sell_ok:
            print(f"⛔ SELL signal ignoré — filtre V44c : {sell_reason}")
        else:
            sell_sl = entry + atr
            sell_tp = entry - (sell_sl - entry) * TP_RATIO

            stype  = "CHoCH" if is_choch_bear else "BOS"
            qsufx  = "_A"   if is_quality_bear else ""
            bos_a  = 1 if (not is_choch_bear and is_quality_bear) else 0
            choch_a= 1 if (    is_choch_bear and is_quality_bear) else 0

            zone_val = 0
            if point_c_bear_price is not None and point_c_bear_index is not None:
                zone_val = determine_zone(
                    point_c_bear_price,
                    df[col50].iloc[point_c_bear_index],
                    df[col200].iloc[point_c_bear_index],
                    is_bullish=False
                )

            d1, d2 = compute_d1_d2(df, point_a_bear_index, last_index)
            d3     = compute_d3(point_a_bear_index, point_e_index)

            # V44d : zone du plus haut absolu entre a et b
            zone_abs_val = compute_zone_abs(df, point_a_bear_index, last_index, is_bullish=False)

            print(f"\n🔴 {'🔄' if is_choch_bear else '📉'} {stype}{qsufx} BEARISH SIGNAL! ✅ Tous filtres OK")
            print(f"   Cloud: {cloud} | Zone: {zone_val} | EMA50<{ema50:.0f} < EMA200<{ema200:.0f} < EMA250<{ema250:.0f}")
            print(f"   D1={d1['price']:.1f}$ ({d1['pips']:.0f}p) | D2={d2['price']:.1f}$ ({d2['pips']:.0f}p) | D3={d3}c [{cross_type}]")
            print(f"   Entry: {entry:.2f} | SL: {sell_sl:.2f} | TP: {sell_tp:.2f}")

            place_sell_order(
                df, entry, sell_sl, sell_tp,
                ema5, ema8, ema30, ema50, ema200, ema250, atr,
                0, bear_distance, bos_a, choch_a,
                zone_val, zone_abs_val,
                d1['price'], d1['pips'], d1['points'],
                d2['price'], d2['pips'], d2['points'],
                d3, cloud,
                f"SELL - {stype}{qsufx} Bearish (swing={CHOCH_SWING_SIZE}, micro={CHOCH_MICRO_SWING})",
                point_a_index=point_a_bear_index,
                point_b_index=last_index,
                point_c_index=point_c_bear_index,
                point_c_price_val=point_c_bear_price,
                point_e_index=point_e_index,
            )

    else:
        print("⚪ No BOS/CHoCH detected")


# ==================== BOUCLE PRINCIPALE ====================

def run_bot():
    print("=" * 65)
    print("🤖 BINANCE ALGORITHMIC TRADING BOT - V44f")
    print("=" * 65)
    print(f"Symbol:        {SYMBOL}")
    print(f"Timeframe:     {TIMEFRAME}")
    print(f"Position Size: {POSITION_SIZE_PERCENT}% of balance")
    print(f"EMA Cloud:     {EMA_FAST} / {EMA_MEDIUM} / {EMA_SLOW}")
    strategies = []
    if ENABLE_EMA_CROSSOVER: strategies.append("EMA Crossover (5/8)")
    if ENABLE_CHOCH:         strategies.append(f"CHoCH (swing={CHOCH_SWING_SIZE})")
    print(f"Strategies:    {' + '.join(strategies) if strategies else 'None'}")
    print(f"Charts:        {'mplfinance ✅' if MPLFINANCE_AVAILABLE else 'matplotlib fallback ⚠️'} → {CHARTS_DIR}/")
    print(f"Price log:     {price_log_file}")
    print(f"Trade log:     {log_file}")
    print("=" * 65)
    print("⚠️  STOP: Ctrl+C  or  create STOP_BOT.txt")
    print("=" * 65)

    init_log_file()
    init_price_log()

    balance = get_account_balance()
    print(f"💰 Initial USDC Balance: {balance:.2f} USDC")
    print("=" * 65)

    last_candle_check = None
    check_counter     = 0

    while True:
        try:
            if os.path.exists('STOP_BOT.txt'):
                print("\n🛑 STOP_BOT.txt detected - Shutting down...")
                os.remove('STOP_BOT.txt')
                print(analyze_performance())
                print("\n✅ Bot stopped successfully")
                break

            current_time = datetime.now(timezone.utc)
            open_trades  = trade_log[trade_log['status'] == 'OPEN']

            # ---- MODE ACTIF ----
            if len(open_trades) > 0:
                current_price = get_current_price()
                trade  = open_trades.iloc[0]
                signal = trade['signal']
                entry  = trade['entry_price']
                sl, tp = trade['sl'], trade['tp']
                tid    = trade.get('trade_id', '?')

                pnl_pct = ((current_price - entry) / entry * 100) if signal == "BUY" \
                          else ((entry - current_price) / entry * 100)

                print(f"📊 [{current_time.strftime('%H:%M:%S')}] [{tid}] {signal} | "
                      f"Price: {current_price:.2f} | P&L: {pnl_pct:+.2f}% | "
                      f"SL: {sl:.2f} | TP: {tp:.2f}")

                # Vérifier SL/TP + enregistrer tick price_log
                check_open_trades_realtime(
                    current_price,
                    current_time.strftime("%Y-%m-%d %H:%M:%S")
                )
                time.sleep(CHECK_INTERVAL_IN_POSITION)

            # ---- MODE PASSIF ----
            else:
                if TIMEFRAME == Client.KLINE_INTERVAL_1MINUTE:    check_interval = 1
                elif TIMEFRAME == Client.KLINE_INTERVAL_15MINUTE: check_interval = 15
                elif TIMEFRAME == Client.KLINE_INTERVAL_1HOUR:    check_interval = 60
                else: check_interval = 1

                if (current_time.minute % check_interval == 0 and
                    current_time.second < 10 and
                    last_candle_check != current_time.minute):

                    check_counter += 1
                    print(f"\n{'='*65}")
                    print(f"🔍 Signal search #{check_counter} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*65}")

                    price = get_candles()
                    price = calculate_indicators(price)

                    last = price.iloc[-1]
                    cloud_state = "🟢 UP" if last.get('ema_cloud_up', False) else "🔴 DOWN"
                    print(f"   EMA Cloud: {cloud_state} | "
                          f"EMA50={last[f'EMA_{EMA_FAST}']:.2f} | "
                          f"EMA200={last[f'EMA_{EMA_MEDIUM}']:.2f} | "
                          f"EMA250={last[f'EMA_{EMA_SLOW}']:.2f}")

                    if ENABLE_EMA_CROSSOVER: ema_crossover(price)
                    if ENABLE_CHOCH:         strategy_choch(price)

                    if check_counter % 20 == 0:
                        print(analyze_performance())

                    last_candle_check = current_time.minute

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C - Stopping...")
            print(analyze_performance())
            print("\n✅ Bot stopped successfully")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            time.sleep(5)


# ==================== EXECUTION ====================

if __name__ == "__main__":
    run_bot()
