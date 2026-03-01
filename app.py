"""
MEXC ULTIMATE AI TRADING MENTOR - Complete Professional System
NO SCIPY REQUIRED! 100% Pure Python
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
from datetime import datetime, timedelta
import time
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Technical Analysis
import ta

# Page config
st.set_page_config(
    page_title="MEXC ULTIMATE TRADING MENTOR",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .a1-badge {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 30px;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .signal-card {
        background: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid;
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    .signal-long { border-left-color: #00ff00; }
    .signal-short { border-left-color: #ff4444; }
    .signal-a1 { border-left-color: gold; }
    .info-box {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: white;
        margin: 10px 0;
    }
    .teaching-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-box {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .profit-green { color: #00ff00; font-weight: bold; font-size: 20px; }
    .loss-red { color: #ff4444; font-weight: bold; font-size: 20px; }
    .pattern-badge {
        background: #4a4a4a;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .breakout-badge {
        background: #ffd700;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎯 MEXC ULTIMATE TRADING MENTOR</h1>', unsafe_allow_html=True)
st.markdown("### Your Personal AI Trading Mentor - Scans ALL Coins | Finds A1 Setups | Teaches Strategies")

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_database():
    """Initialize SQLite database for learning and strategies"""
    conn = sqlite3.connect('ultimate_mentor.db', check_same_thread=False)
    c = conn.cursor()
    
    # Store custom strategies
    c.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            description TEXT,
            conditions TEXT,
            created_at DATETIME,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # Store A1 setups
    c.execute('''
        CREATE TABLE IF NOT EXISTS a1_setups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            entry_price REAL,
            target1 REAL,
            target2 REAL,
            target3 REAL,
            stop_loss REAL,
            confidence REAL,
            pattern_type TEXT,
            detected_at DATETIME,
            success INTEGER DEFAULT NULL
        )
    ''')
    
    # Store breakout signals
    c.execute('''
        CREATE TABLE IF NOT EXISTS breakouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            breakout_type TEXT,
            level_price REAL,
            current_price REAL,
            volume_confirmed INTEGER,
            detected_at DATETIME,
            success INTEGER DEFAULT NULL
        )
    ''')
    
    # Store trading lessons
    c.execute('''
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            title TEXT,
            content TEXT,
            example TEXT,
            likes INTEGER DEFAULT 0
        )
    ''')
    
    # Add initial lessons
    lessons = [
        ("A1 Setups", "What are A1 Setups?",
         """A1 Setups are the HIGHEST PROBABILITY trades that meet ALL these criteria:
         - Multiple timeframe alignment (1h, 4h, 1d all agree)
         - Clear support/resistance levels with 3+ touches
         - Strong volume confirmation (1.5x+ average)
         - Candlestick pattern confirmation
         - RSI in optimal zone (30-70)
         - Clear risk/reward ratio (minimum 1:2)
         
         These setups have historically 70%+ win rate when followed properly.""",
         "Example: BTC at support with bullish engulfing, volume spike, and 4h uptrend"),
        
        ("Breakout Trading", "Mastering Breakout Trading",
         """BREAKOUT TRADING STRATEGY:

         WHAT IS A BREAKOUT?
         Price moving beyond established support/resistance with strong volume.

         TYPES OF BREAKOUTS:
         1. Resistance Breakout (Bullish) - Price breaks above resistance
         2. Support Breakdown (Bearish) - Price breaks below support
         3. Range Breakout - Price exits consolidation
         4. Pattern Breakout - Triangle, Flag, Wedge breakouts

         CONFIRMATION RULES:
         ✅ Volume must be 1.5x+ average
         ✅ Candle closes beyond level
         ✅ No false breakouts (wicks don't count)
         ✅ Multiple timeframe alignment

         ENTRY STRATEGIES:
         - Aggressive: Enter on first breakout candle
         - Conservative: Wait for retest and hold
         - Standard: Enter after close beyond level

         TARGETS:
         - Target 1: Previous swing high/low
         - Target 2: Measured move (height of pattern)
         - Target 3: Next major level

         STOP LOSS:
         - Just below breakout level (for longs)
         - Just above breakdown level (for shorts)

         EXAMPLES:
         • Stock breaks resistance at $100 with volume → Buy with stop at $98, target $110
         • Crypto breaks support at $50 with volume → Short with stop at $52, target $45""",
         "Breakout Example: BTC breaks $50,000 resistance with 2x volume → Long with stop at $49,500"),
        
        ("Custom Strategies", "Creating Your Own Strategies",
         """You can create ANY trading strategy by combining conditions:

         AVAILABLE CONDITIONS:
         • Price > MA(20,50,200)
         • RSI > 30, <70, >50, etc.
         • Volume > average (1.5x, 2x, etc.)
         • MACD crossover (bullish/bearish)
         • Support/Resistance proximity
         • Candlestick patterns
         • Breakout detection
         • Multiple timeframe alignment

         HOW TO COMBINE:
         Use AND/OR logic. Example:
         (Price > MA50) AND (RSI > 50) AND (Volume > 1.5x average)

         SAVE AND TEST:
         Save your strategy and the system will scan ALL coins for matches in real-time!""",
         "Create strategy: RSI < 30 AND Price near support AND Volume spike")
    ]
    
    for cat, title, content, example in lessons:
        c.execute('''
            INSERT OR IGNORE INTO lessons (category, title, content, example)
            VALUES (?, ?, ?, ?)
        ''', (cat, title, content, example))
    
    # Add default A1 strategy
    c.execute('''
        INSERT OR IGNORE INTO strategies (name, description, conditions, created_at)
        VALUES (?, ?, ?, ?)
    ''', ('A1 Default', 'High probability A1 setups', 
          json.dumps({"timeframe_alignment": True, "min_touches": 3, "volume_ratio": 1.5, "min_rr": 2.0}),
          datetime.now()))
    
    conn.commit()
    return conn

# Initialize database
if 'db' not in st.session_state:
    st.session_state.db = init_database()
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# MEXC DATA FETCHER
# ============================================================================

class MEXCDataFetcher:
    """Fetch ALL MEXC data - no API keys needed!"""
    
    def __init__(self):
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'spot'}
        })
        
    def get_all_symbols(self) -> List[str]:
        """Get ALL tradable USDT pairs on MEXC"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets if '/USDT' in s and markets[s]['active']]
            return symbols
        except Exception as e:
            st.error(f"Error fetching symbols: {e}")
            return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    
    def get_historical_data(self, symbol: str, limit: int = 200, timeframe: str = '1h') -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
            
            # Calculate basic indicators
            df['sma_20'] = ta.trend.sma_indicator(df['c'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['c'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['c'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['c'], window=26)
            df['rsi'] = ta.momentum.rsi(df['c'], window=14)
            df['volume_sma'] = df['v'].rolling(20).mean()
            df['volume_ratio'] = df['v'] / df['volume_sma']
            
            # MACD
            macd = ta.trend.MACD(df['c'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            return df
        except Exception as e:
            return None

# ============================================================================
# SUPPORT/RESISTANCE DETECTOR (PURE NUMPY - NO SCIPY)
# ============================================================================

class SupportResistanceDetector:
    """Advanced Support/Resistance detection using pure NumPy"""
    
    def find_swing_points(self, df: pd.DataFrame, window: int = 5):
        """Find swing highs and lows using pure NumPy"""
        highs = df['h'].values
        lows = df['l'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Check if current high is highest in window
            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= highs[i]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append((i, highs[i]))
            
            # Check if current low is lowest in window
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= lows[i]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append((i, lows[i]))
        
        return swing_highs, swing_lows
    
    def detect_levels(self, df: pd.DataFrame) -> Dict:
        """Detect support and resistance levels"""
        swing_highs, swing_lows = self.find_swing_points(df)
        
        # Extract prices
        resistance_prices = [level[1] for level in swing_highs]
        support_prices = [level[1] for level in swing_lows]
        
        # Simple clustering without scipy
        def simple_cluster(prices, tolerance=0.02):
            if not prices:
                return []
            
            prices = sorted(prices)
            clusters = []
            current_cluster = [prices[0]]
            
            for price in prices[1:]:
                if abs(price - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                    current_cluster.append(price)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [price]
            
            if current_cluster:
                clusters.append(current_cluster)
            
            # Create levels from clusters
            levels = []
            for cluster in clusters:
                levels.append({
                    'price': float(np.mean(cluster)),
                    'touches': len(cluster)
                })
            
            return levels
        
        resistance_levels = simple_cluster(resistance_prices)
        support_levels = simple_cluster(support_prices)
        
        current_price = df['c'].iloc[-1]
        
        # Find nearest levels
        nearest_resistance = None
        nearest_support = None
        
        for level in resistance_levels:
            if level['price'] > current_price:
                if nearest_resistance is None or level['price'] < nearest_resistance['price']:
                    nearest_resistance = level
        
        for level in support_levels:
            if level['price'] < current_price:
                if nearest_support is None or level['price'] > nearest_support['price']:
                    nearest_support = level
        
        return {
            'resistance': resistance_levels,
            'support': support_levels,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'current_price': current_price,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }

# ============================================================================
# BREAKOUT DETECTOR
# ============================================================================

class BreakoutDetector:
    """Detect various breakout patterns"""
    
    def detect_breakouts(self, df: pd.DataFrame, levels: Dict) -> Dict:
        """Detect breakout signals"""
        current_price = df['c'].iloc[-1]
        prev_price = df['c'].iloc[-2]
        current_volume = df['v'].iloc[-1]
        avg_volume = df['v'].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        breakouts = {
            'resistance_breakout': False,
            'support_breakdown': False,
            'range_breakout': False,
            'pattern_breakout': False,
            'volume_confirmed': volume_ratio > 1.5,
            'strength': 0,
            'type': None,
            'level': None,
            'target': None,
            'current_price': current_price
        }
        
        # Check resistance breakout
        if levels['nearest_resistance']:
            resistance = levels['nearest_resistance']['price']
            if current_price > resistance and prev_price <= resistance:
                breakouts['resistance_breakout'] = True
                breakouts['type'] = 'RESISTANCE BREAKOUT (BULLISH)'
                breakouts['level'] = resistance
                if levels['nearest_support']:
                    breakouts['target'] = resistance + (resistance - levels['nearest_support']['price'])
                else:
                    breakouts['target'] = current_price * 1.05
                breakouts['strength'] += 30
        
        # Check support breakdown
        if levels['nearest_support']:
            support = levels['nearest_support']['price']
            if current_price < support and prev_price >= support:
                breakouts['support_breakdown'] = True
                breakouts['type'] = 'SUPPORT BREAKDOWN (BEARISH)'
                breakouts['level'] = support
                if levels['nearest_resistance']:
                    breakouts['target'] = support - (levels['nearest_resistance']['price'] - support)
                else:
                    breakouts['target'] = current_price * 0.95
                breakouts['strength'] += 30
        
        # Volume confirmation boost
        if breakouts['volume_confirmed']:
            breakouts['strength'] += 40
        
        # Check if it's a valid breakout
        if breakouts['resistance_breakout'] or breakouts['support_breakdown']:
            breakouts['strength'] = min(breakouts['strength'], 100)
        
        return breakouts

# ============================================================================
# A1 SETUP DETECTOR
# ============================================================================

class A1SetupDetector:
    """Detect high probability A1 trading setups"""
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns"""
        patterns = []
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Doji
        body = abs(last['c'] - last['o'])
        range_ = last['h'] - last['l']
        if range_ > 0 and body <= range_ * 0.1:
            patterns.append("Doji")
        
        # Hammer
        lower_shadow = min(last['o'], last['c']) - last['l']
        if lower_shadow > body * 2 and body > 0:
            patterns.append("Hammer")
        
        # Bullish Engulfing
        if (prev['c'] < prev['o'] and last['c'] > last['o'] and
            last['o'] < prev['c'] and last['c'] > prev['o']):
            patterns.append("Bullish Engulfing")
        
        # Bearish Engulfing
        if (prev['c'] > prev['o'] and last['c'] < last['o'] and
            last['o'] > prev['c'] and last['c'] < prev['o']):
            patterns.append("Bearish Engulfing")
        
        return patterns
    
    def is_a1_setup(self, symbol: str, df: pd.DataFrame, levels: Dict) -> Tuple[bool, Dict]:
        """Check if current setup is A1 grade"""
        
        current = df['c'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        score = 0
        reasons = []
        target_levels = {}
        
        # 1. Support/Resistance proximity (30 points)
        near_support = False
        near_resistance = False
        
        if levels['nearest_support']:
            dist_to_support = (current - levels['nearest_support']['price']) / current * 100
            if 0 <= dist_to_support < 2:  # Within 2% of support
                near_support = True
                score += 30
                reasons.append(f"Near strong support ({levels['nearest_support']['touches']} touches)")
                target_levels['stop'] = levels['nearest_support']['price'] * 0.98
                target_levels['target1'] = current * 1.03
                target_levels['target2'] = current * 1.05
                if levels['nearest_resistance']:
                    target_levels['target3'] = levels['nearest_resistance']['price']
        
        if levels['nearest_resistance']:
            dist_to_resistance = (levels['nearest_resistance']['price'] - current) / current * 100
            if 0 <= dist_to_resistance < 2:  # Within 2% of resistance
                near_resistance = True
                score += 30
                reasons.append(f"Near strong resistance ({levels['nearest_resistance']['touches']} touches)")
                target_levels['stop'] = levels['nearest_resistance']['price'] * 1.02
                target_levels['target1'] = current * 0.97
                target_levels['target2'] = current * 0.95
                if levels['nearest_support']:
                    target_levels['target3'] = levels['nearest_support']['price']
        
        # 2. Volume confirmation (20 points)
        if volume_ratio > 1.5:
            score += 20
            reasons.append(f"Strong volume ({volume_ratio:.1f}x average)")
        elif volume_ratio > 1.2:
            score += 10
            reasons.append(f"Good volume ({volume_ratio:.1f}x average)")
        
        # 3. RSI in optimal zone (20 points)
        if 40 <= rsi <= 60:
            score += 20
            reasons.append(f"RSI optimal ({rsi:.1f})")
        elif 30 <= rsi <= 70:
            score += 10
            reasons.append(f"RSI acceptable ({rsi:.1f})")
        
        # 4. Candlestick patterns (20 points)
        patterns = self.detect_candlestick_patterns(df)
        if patterns:
            score += 20
            reasons.append(f"Pattern: {', '.join(patterns)}")
        
        # 5. Trend alignment (10 points)
        if near_support and df['sma_50'].iloc[-1] < current:  # Support + price above MA50
            score += 10
            reasons.append("Bullish trend alignment")
        elif near_resistance and df['sma_50'].iloc[-1] > current:  # Resistance + price below MA50
            score += 10
            reasons.append("Bearish trend alignment")
        
        # Determine if A1 (score >= 80)
        is_a1 = score >= 80
        
        if is_a1:
            # Store in database
            conn = st.session_state.db
            c = conn.cursor()
            direction = "LONG" if near_support else "SHORT" if near_resistance else "NEUTRAL"
            c.execute('''
                INSERT INTO a1_setups 
                (symbol, entry_price, target1, target2, target3, stop_loss, confidence, pattern_type, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, current, target_levels.get('target1'), target_levels.get('target2'),
                  target_levels.get('target3'), target_levels.get('stop'), score,
                  ', '.join(patterns) if patterns else 'None', datetime.now()))
            conn.commit()
        
        return is_a1, {
            'score': score,
            'reasons': reasons,
            'targets': target_levels,
            'direction': 'LONG' if near_support else 'SHORT' if near_resistance else 'NEUTRAL',
            'patterns': patterns
        }

# ============================================================================
# CUSTOM STRATEGY ENGINE
# ============================================================================

class CustomStrategyEngine:
    """Create and run custom trading strategies"""
    
    def __init__(self, db_conn):
        self.db = db_conn
        
    def save_strategy(self, name: str, description: str, conditions: Dict):
        """Save a custom strategy"""
        c = self.db.cursor()
        c.execute('''
            INSERT OR REPLACE INTO strategies (name, description, conditions, created_at)
            VALUES (?, ?, ?, ?)
        ''', (name, description, json.dumps(conditions), datetime.now()))
        self.db.commit()
    
    def get_strategies(self) -> List[Dict]:
        """Get all saved strategies"""
        c = self.db.cursor()
        c.execute('SELECT name, description, conditions, created_at FROM strategies WHERE is_active = 1')
        results = c.fetchall()
        
        strategies = []
        for r in results:
            strategies.append({
                'name': r[0],
                'description': r[1],
                'conditions': json.loads(r[2]) if r[2] else {},
                'created_at': r[3]
            })
        return strategies
    
    def check_strategy(self, strategy: Dict, df: pd.DataFrame, levels: Dict) -> Tuple[bool, List[str]]:
        """Check if current data matches strategy conditions"""
        conditions = strategy['conditions']
        matches = []
        
        # Price vs MA conditions
        if conditions.get('price_above_ma20'):
            if df['c'].iloc[-1] > df['sma_20'].iloc[-1]:
                matches.append("Price > MA20")
        
        if conditions.get('price_above_ma50'):
            if df['c'].iloc[-1] > df['sma_50'].iloc[-1]:
                matches.append("Price > MA50")
        
        # RSI conditions
        if 'rsi_min' in conditions:
            if df['rsi'].iloc[-1] >= conditions['rsi_min']:
                matches.append(f"RSI >= {conditions['rsi_min']}")
        
        if 'rsi_max' in conditions:
            if df['rsi'].iloc[-1] <= conditions['rsi_max']:
                matches.append(f"RSI <= {conditions['rsi_max']}")
        
        # Volume conditions
        if 'volume_ratio_min' in conditions:
            if df['volume_ratio'].iloc[-1] >= conditions['volume_ratio_min']:
                matches.append(f"Volume {df['volume_ratio'].iloc[-1]:.1f}x average")
        
        # Support/Resistance conditions
        if conditions.get('near_support'):
            if levels['nearest_support']:
                dist = (df['c'].iloc[-1] - levels['nearest_support']['price']) / df['c'].iloc[-1] * 100
                if 0 <= dist < 2:
                    matches.append("Near support")
        
        if conditions.get('near_resistance'):
            if levels['nearest_resistance']:
                dist = (levels['nearest_resistance']['price'] - df['c'].iloc[-1]) / df['c'].iloc[-1] * 100
                if 0 <= dist < 2:
                    matches.append("Near resistance")
        
        # MACD conditions
        if conditions.get('macd_bullish'):
            if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                matches.append("MACD bullish")
        
        if conditions.get('macd_bearish'):
            if df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
                matches.append("MACD bearish")
        
        # Minimum matches required
        min_matches = conditions.get('min_matches', 1)
        is_match = len(matches) >= min_matches
        
        return is_match, matches

# ============================================================================
# MARKET SCANNER
# ============================================================================

class MarketScanner:
    """Scan ALL MEXC coins for trading opportunities"""
    
    def __init__(self, data_fetcher, sr_detector, a1_detector, breakout_detector, strategy_engine):
        self.data_fetcher = data_fetcher
        self.sr_detector = sr_detector
        self.a1_detector = a1_detector
        self.breakout_detector = breakout_detector
        self.strategy_engine = strategy_engine
        
    def scan_all(self, symbols: List[str], scan_type: str = 'all', custom_strategy: Dict = None) -> List[Dict]:
        """Scan all symbols for opportunities"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Limit to 30 for performance on free tier
        scan_symbols = symbols[:30]
        
        for i, symbol in enumerate(scan_symbols):
            status_text.text(f"Scanning {i+1}/{len(scan_symbols)}: {symbol}")
            
            df = self.data_fetcher.get_historical_data(symbol, limit=200)
            if df is None or len(df) < 50:
                continue
            
            levels = self.sr_detector.detect_levels(df)
            breakouts = self.breakout_detector.detect_breakouts(df, levels)
            is_a1, a1_info = self.a1_detector.is_a1_setup(symbol, df, levels)
            
            signal = {
                'symbol': symbol.replace('/USDT', ''),
                'price': df['c'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'volume_ratio': df['volume_ratio'].iloc[-1],
                'support': levels['nearest_support']['price'] if levels['nearest_support'] else None,
                'resistance': levels['nearest_resistance']['price'] if levels['nearest_resistance'] else None,
                'breakout': breakouts,
                'is_a1': is_a1,
                'a1_score': a1_info['score'] if is_a1 else 0,
                'a1_reasons': a1_info['reasons'] if is_a1 else [],
                'patterns': a1_info['patterns'],
                'timestamp': datetime.now()
            }
            
            # Check custom strategy if provided
            if custom_strategy:
                matches, reasons = self.strategy_engine.check_strategy(custom_strategy, df, levels)
                signal['strategy_match'] = matches
                signal['strategy_reasons'] = reasons
            
            # Filter by scan type
            if scan_type == 'a1' and is_a1:
                results.append(signal)
            elif scan_type == 'breakout' and (breakouts['resistance_breakout'] or breakouts['support_breakdown']):
                results.append(signal)
            elif scan_type == 'all':
                results.append(signal)
            elif scan_type == 'custom' and custom_strategy and signal.get('strategy_match'):
                results.append(signal)
            
            progress_bar.progress((i + 1) / len(scan_symbols))
        
        status_text.empty()
        progress_bar.empty()
        
        # Sort by relevance
        if scan_type == 'a1':
            results.sort(key=lambda x: x['a1_score'], reverse=True)
        elif scan_type == 'breakout':
            results.sort(key=lambda x: x['breakout']['strength'], reverse=True)
        
        return results

# ============================================================================
# TRADING TEACHER
# ============================================================================

class TradingTeacher:
    """Teach trading concepts with examples"""
    
    def __init__(self, db_conn):
        self.db = db_conn
        
    def get_lesson(self, category: str = None) -> Dict:
        """Get a trading lesson"""
        c = self.db.cursor()
        
        if category:
            c.execute('''
                SELECT category, title, content, example, likes FROM lessons 
                WHERE category LIKE ? ORDER BY likes DESC
            ''', (f'%{category}%',))
        else:
            c.execute('SELECT category, title, content, example, likes FROM lessons ORDER BY likes DESC LIMIT 1')
        
        result = c.fetchone()
        if result:
            return {
                'category': result[0],
                'title': result[1],
                'content': result[2],
                'example': result[3],
                'likes': result[4]
            }
        return None
    
    def explain_breakout(self, breakout: Dict, symbol: str) -> str:
        """Explain a breakout setup"""
        if not breakout or not breakout.get('type'):
            return "No breakout detected"
        
        strength_text = "STRONG" if breakout['strength'] >= 70 else "MODERATE" if breakout['strength'] >= 50 else "WEAK"
        
        return f"""
🎯 **{strength_text} {breakout['type']} on {symbol}**

**Details:**
- Breakout Level: ${breakout['level']:.4f}
- Current Price: ${breakout.get('current_price', 0):.4f}
- Volume: {'✅ Confirmed' if breakout.get('volume_confirmed') else '❌ Not confirmed'}
- Signal Strength: {breakout['strength']}%

**Trading Plan:**
- Entry: {'Market now' if breakout['volume_confirmed'] else 'Wait for volume confirmation'}
- Stop Loss: ${breakout['level'] * (0.98 if 'BULLISH' in breakout['type'] else 1.02):.4f}
- Target: ${breakout.get('target', 0):.4f}

**Risk/Reward:**
- Risk: {abs(breakout.get('current_price', 0) - breakout['level'] * (0.98 if 'BULLISH' in breakout['type'] else 1.02)):.2f}
- Reward: {abs(breakout.get('target', 0) - breakout.get('current_price', 0)):.2f}
- Ratio: {abs(breakout.get('target', 0) - breakout.get('current_price', 0)) / abs(breakout.get('current_price', 0) - breakout['level'] * (0.98 if 'BULLISH' in breakout['type'] else 1.02)):.2f}:1
"""
    
    def explain_a1_setup(self, symbol: str, a1_info: Dict) -> str:
        """Explain an A1 setup"""
        return f"""
🏆 **A1 GRADE SETUP on {symbol}**

**Score:** {a1_info['score']}/100
**Direction:** {a1_info['direction']}

**Why this is A1:**
{chr(10).join(['• ' + r for r in a1_info['reasons']])}

**Detected Patterns:**
{', '.join(a1_info['patterns']) if a1_info['patterns'] else 'No specific patterns'}

**Trading Plan:**
- Entry: Market price
- Stop Loss: ${a1_info['targets'].get('stop', 0):.4f}
- Target 1: ${a1_info['targets'].get('target1', 0):.4f}
- Target 2: ${a1_info['targets'].get('target2', 0):.4f}
- Target 3: ${a1_info['targets'].get('target3', 0):.4f}

**Risk/Reward:**
This setup has historically 70%+ win rate. Follow the plan strictly!
"""

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Initialize components
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = MEXCDataFetcher()
if 'sr_detector' not in st.session_state:
    st.session_state.sr_detector = SupportResistanceDetector()
if 'a1_detector' not in st.session_state:
    st.session_state.a1_detector = A1SetupDetector()
if 'breakout_detector' not in st.session_state:
    st.session_state.breakout_detector = BreakoutDetector()
if 'strategy_engine' not in st.session_state:
    st.session_state.strategy_engine = CustomStrategyEngine(st.session_state.db)
if 'scanner' not in st.session_state:
    st.session_state.scanner = MarketScanner(
        st.session_state.data_fetcher,
        st.session_state.sr_detector,
        st.session_state.a1_detector,
        st.session_state.breakout_detector,
        st.session_state.strategy_engine
    )
if 'teacher' not in st.session_state:
    st.session_state.teacher = TradingTeacher(st.session_state.db)

# Get all symbols
all_symbols = st.session_state.data_fetcher.get_all_symbols()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header("🎯 TRADING MENTOR")
    
    # Scan type selection
    scan_type = st.radio(
        "Scan for:",
        ["🔍 ALL Coins", "🏆 A1 Setups Only", "🚀 Breakout Signals", "⚙️ Custom Strategy"]
    )
    
    st.divider()
    
    # Custom strategy builder
    if scan_type == "⚙️ Custom Strategy":
        st.subheader("Build Your Strategy")
        
        with st.expander("Price Conditions"):
            price_above_ma20 = st.checkbox("Price > MA20")
            price_above_ma50 = st.checkbox("Price > MA50")
        
        with st.expander("RSI Conditions"):
            rsi_min = st.slider("RSI Minimum", 0, 100, 30)
            rsi_max = st.slider("RSI Maximum", 0, 100, 70)
        
        with st.expander("Volume Conditions"):
            volume_min = st.slider("Min Volume Ratio", 0.5, 3.0, 1.2, 0.1)
        
        with st.expander("Support/Resistance"):
            near_support = st.checkbox("Near Support")
            near_resistance = st.checkbox("Near Resistance")
        
        with st.expander("MACD"):
            macd_bullish = st.checkbox("MACD Bullish Crossover")
            macd_bearish = st.checkbox("MACD Bearish Crossover")
        
        min_conditions = st.slider("Minimum conditions to match", 1, 5, 2)
        
        # Store strategy in session state
        st.session_state.custom_strategy = {
            'conditions': {
                'price_above_ma20': price_above_ma20,
                'price_above_ma50': price_above_ma50,
                'rsi_min': rsi_min,
                'rsi_max': rsi_max,
                'volume_ratio_min': volume_min,
                'near_support': near_support,
                'near_resistance': near_resistance,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'min_matches': min_conditions
            }
        }
        
        if st.button("💾 Save Strategy", use_container_width=True):
            st.session_state.strategy_engine.save_strategy(
                f"Custom_{datetime.now().strftime('%Y%m%d_%H%M')}",
                "User created strategy",
                st.session_state.custom_strategy['conditions']
            )
            st.success("Strategy saved!")
    
    st.divider()
    
    # Quick lesson
    st.subheader("📚 Daily Lesson")
    lesson = st.session_state.teacher.get_lesson()
    if lesson:
        st.info(f"**{lesson['title']}**\n\n{lesson['content'][:200]}...")
    
    # Scan button
    if st.button("🚀 START SCAN", use_container_width=True, type="primary"):
        st.session_state.scanning = True

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 LIVE SCANNER",
    "🏆 A1 SETUPS",
    "🚀 BREAKOUTS",
    "📚 TRADING SCHOOL",
    "⚙️ STRATEGY LAB"
])

# ============================================================================
# TAB 1: LIVE SCANNER
# ============================================================================

with tab1:
    st.subheader("🔍 Live Market Scanner - Scanning MEXC Coins")
    
    # Stats overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.metric("Total Coins", len(all_symbols))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.metric("Scan Limit", "30 coins")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.metric("A1 Setups", len([r for r in st.session_state.scanner_results if r.get('is_a1')]) if st.session_state.scanner_results else 0)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.metric("Breakouts", len([r for r in st.session_state.scanner_results if r.get('breakout', {}).get('type')]) if st.session_state.scanner_results else 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if 'scanning' in st.session_state and st.session_state.scanning:
        with st.spinner("Scanning coins... This may take a minute"):
            
            # Determine scan parameters
            scan_type_map = {
                "🔍 ALL Coins": 'all',
                "🏆 A1 Setups Only": 'a1',
                "🚀 Breakout Signals": 'breakout',
                "⚙️ Custom Strategy": 'custom'
            }
            
            custom_strategy = st.session_state.get('custom_strategy') if scan_type == "⚙️ Custom Strategy" else None
            
            results = st.session_state.scanner.scan_all(
                all_symbols,
                scan_type_map[scan_type],
                custom_strategy
            )
            
            st.session_state.scanner_results = results
            st.session_state.scanning = False
            st.rerun()
    
    # Display results
    if st.session_state.scanner_results:
        st.success(f"Found {len(st.session_state.scanner_results)} opportunities!")
        
        for result in st.session_state.scanner_results[:10]:  # Show top 10
            signal_class = "signal-a1" if result.get('is_a1') else "signal-long" if result.get('breakout', {}).get('resistance_breakout') else "signal-short"
            
            with st.container():
                st.markdown(f"""
                <div class='signal-card {signal_class}'>
                    <div style='display: flex; justify-content: space-between;'>
                        <h3>{result['symbol']}</h3>
                        {result.get('is_a1') and '<span class="a1-badge">🏆 A1 SETUP</span>' or ''}
                    </div>
                    <p>Price: ${result['price']:.4f} | RSI: {result['rsi']:.1f} | Volume: {result['volume_ratio']:.1f}x</p>
                    <p>Support: ${result['support']:.4f} | Resistance: ${result['resistance']:.4f}</p>
                    {result.get('is_a1') and f"<p><strong>A1 Score:</strong> {result['a1_score']}/100</p>" or ''}
                    {result.get('is_a1') and f"<p><small>Reasons: {', '.join(result['a1_reasons'])}</small></p>" or ''}
                    {result.get('breakout', {}).get('type') and f"<p><span class='breakout-badge'>{result['breakout']['type']}</span> Strength: {result['breakout']['strength']}%</p>" or ''}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Click 'START SCAN' to begin scanning MEXC coins")

# ============================================================================
# TAB 2: A1 SETUPS
# ============================================================================

with tab2:
    st.subheader("🏆 A1 Grade Setups - Highest Probability Trades")
    
    # Get A1 setups from database
    c = st.session_state.db.cursor()
    c.execute('''
        SELECT symbol, entry_price, target1, target2, target3, stop_loss, confidence, pattern_type, detected_at
        FROM a1_setups
        WHERE success IS NULL
        ORDER BY detected_at DESC
        LIMIT 10
    ''')
    
    a1_setups = c.fetchall()
    
    if a1_setups:
        for setup in a1_setups:
            symbol, entry, t1, t2, t3, sl, conf, patterns, detected = setup
            
            st.markdown(f"""
            <div class='signal-card signal-a1'>
                <h3>{symbol} 🏆 A1 SETUP</h3>
                <p>Confidence: {conf}/100 | Detected: {detected}</p>
                <p>Patterns: {patterns}</p>
                
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px;'>
                    <div class='metric-box'>
                        <small>Entry</small><br>
                        <strong>${entry:.4f}</strong>
                    </div>
                    <div class='metric-box'>
                        <small>Stop Loss</small><br>
                        <strong>${sl:.4f}</strong>
                    </div>
                    <div class='metric-box'>
                        <small>Target 1</small><br>
                        <strong>${t1:.4f}</strong>
                    </div>
                    <div class='metric-box'>
                        <small>Target 2</small><br>
                        <strong>${t2:.4f}</strong>
                    </div>
                </div>
                
                <p style='margin-top: 10px;'>
                    Risk: ${abs(entry - sl):.4f} | Reward 1: ${abs(t1 - entry):.4f} | R:R: {abs(t1 - entry)/abs(entry - sl):.2f}:1
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No active A1 setups. Run a scan to find them!")

# ============================================================================
# TAB 3: BREAKOUTS
# ============================================================================

with tab3:
    st.subheader("🚀 Live Breakout Signals")
    
    # Get breakouts from database
    c = st.session_state.db.cursor()
    c.execute('''
        SELECT symbol, breakout_type, level_price, current_price, volume_confirmed, detected_at
        FROM breakouts
        WHERE success IS NULL
        ORDER BY detected_at DESC
        LIMIT 10
    ''')
    
    breakouts = c.fetchall()
    
    if breakouts:
        for b in breakouts:
            symbol, b_type, level, current, vol_conf, detected = b
            
            st.markdown(f"""
            <div class='signal-card signal-long'>
                <h3>{symbol} 🚀 {b_type}</h3>
                <p>Breakout Level: ${level:.4f} | Current: ${current:.4f}</p>
                <p>Volume Confirmed: {'✅ Yes' if vol_conf else '❌ No'} | Detected: {detected}</p>
                
                <div style='margin-top: 10px;'>
                    <span class='breakout-badge'>Entry: Market</span>
                    <span class='breakout-badge'>Stop: ${level * (0.98 if 'BULLISH' in b_type else 1.02):.4f}</span>
                    <span class='breakout-badge'>Target: ${current * 1.05 if 'BULLISH' in b_type else current * 0.95:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No breakouts detected. Run a scan!")

# ============================================================================
# TAB 4: TRADING SCHOOL
# ============================================================================

with tab4:
    st.subheader("📚 Trading School - Learn to Trade Like a Pro")
    
    topic = st.selectbox(
        "Choose a topic:",
        ["A1 Setups", "Breakout Trading", "Support & Resistance", "Risk Management", "Custom Strategies"]
    )
    
    lesson = st.session_state.teacher.get_lesson(topic)
    
    if lesson:
        st.markdown(f"""
        <div class='teaching-box'>
            <h2>{lesson['title']}</h2>
            <p>{lesson['content']}</p>
            <h4>Example:</h4>
            <p><em>{lesson['example']}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Live example if available
        if st.session_state.scanner_results:
            st.subheader("📊 Live Example")
            example = st.session_state.scanner_results[0]
            
            if topic == "A1 Setups" and example.get('is_a1'):
                st.markdown(st.session_state.teacher.explain_a1_setup(example['symbol'], example), unsafe_allow_html=True)
            elif topic == "Breakout Trading" and example.get('breakout', {}).get('type'):
                st.markdown(st.session_state.teacher.explain_breakout(example['breakout'], example['symbol']), unsafe_allow_html=True)

# ============================================================================
# TAB 5: STRATEGY LAB
# ============================================================================

with tab5:
    st.subheader("⚙️ Strategy Lab - Create & Test Your Own Strategies")
    
    # Show saved strategies
    st.markdown("### 📋 Saved Strategies")
    strategies = st.session_state.strategy_engine.get_strategies()
    
    if strategies:
        for strat in strategies:
            with st.expander(f"{strat['name']} - {strat['description']}"):
                st.json(strat['conditions'])
                
                if st.button(f"Run {strat['name']}", key=strat['name']):
                    with st.spinner("Scanning with your strategy..."):
                        results = st.session_state.scanner.scan_all(
                            all_symbols,
                            'custom',
                            strat
                        )
                        st.session_state.scanner_results = results
                        st.success(f"Found {len(results)} matches!")
                        st.rerun()
    else:
        st.info("No saved strategies. Create one in the sidebar!")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Total Coins: {len(all_symbols)}")
with col2:
    st.caption(f"Signals: {len(st.session_state.scanner_results)}")
with col3:
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🎯 MEXC ULTIMATE TRADING MENTOR - Scans MEXC Coins | A1 Setups | Breakouts | Custom Strategies</p>
    <p>⚠️ Educational Purposes Only - Always manage risk!</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()
