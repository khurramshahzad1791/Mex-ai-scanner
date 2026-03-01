import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
import warnings
warnings.filterwarnings('ignore')

# Simplified ML Libraries (lighter than full ensemble)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import sqlite3
import hashlib
import json
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="MEXC Quantum AI Scanner", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .signal-long {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px;
    }
    .signal-short {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px;
    }
    .explanation-box {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        color: white;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('mexc_ai.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            symbol TEXT PRIMARY KEY,
            model_data BLOB,
            accuracy REAL,
            last_trained TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS patterns (
            symbol TEXT,
            pattern_name TEXT,
            occurrences INTEGER,
            success_rate REAL,
            timestamp TIMESTAMP,
            PRIMARY KEY (symbol, pattern_name)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT,
            signal_type TEXT,
            price REAL,
            confidence REAL,
            timestamp TIMESTAMP,
            outcome REAL
        )
    ''')
    
    conn.commit()
    return conn

# Initialize session state
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'all_symbols' not in st.session_state:
    st.session_state.all_symbols = []
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

# MEXC Handler
class MEXCHandler:
    def __init__(self):
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
            'rateLimit': 1000,
            'options': {'defaultType': 'future'}
        })
    
    def get_all_usdt_pairs(self):
        """Get all USDT perpetual pairs"""
        try:
            markets = self.exchange.load_markets()
            pairs = [
                symbol for symbol in markets 
                if '/USDT:USDT' in symbol and markets[symbol]['active']
            ]
            return pairs
        except Exception as e:
            logger.error(f"Error fetching pairs: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=500):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

# AI Engine
class QuantumAI:
    def __init__(self):
        self.patterns_cache = defaultdict(list)
    
    def extract_features(self, df):
        """Extract key features for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}'] = df['c'].pct_change(period)
            features[f'volatility_{period}'] = df['c'].pct_change().rolling(period).std()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['c'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = (df['c'] - features[f'sma_{period}']) / features[f'sma_{period}']
        
        # Volume features
        features['volume'] = df['v']
        features['volume_sma'] = df['v'].rolling(20).mean()
        features['volume_ratio'] = df['v'] / features['volume_sma']
        
        # RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['c'].ewm(span=12).mean()
        exp2 = df['c'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_mid = df['c'].rolling(20).mean()
        bb_std = df['c'].rolling(20).std()
        features['bb_position'] = (df['c'] - bb_mid) / (2 * bb_std)
        
        # Target (next period direction)
        features['target'] = (df['c'].shift(-1) > df['c']).astype(int)
        
        return features
    
    def detect_patterns(self, df):
        """Detect candlestick patterns"""
        patterns = {}
        
        # Doji
        patterns['doji'] = abs(df['c'] - df['o']) <= (df['h'] - df['l']) * 0.1
        
        # Hammer
        body = abs(df['c'] - df['o'])
        lower_shadow = df[['o', 'c']].min(axis=1) - df['l']
        upper_shadow = df['h'] - df[['o', 'c']].max(axis=1)
        patterns['hammer'] = (lower_shadow > body * 2) & (upper_shadow < body * 0.3)
        
        # Engulfing
        patterns['bullish_engulfing'] = (
            (df['c'].shift(1) < df['o'].shift(1)) & 
            (df['c'] > df['o']) & 
            (df['o'] < df['c'].shift(1)) & 
            (df['c'] > df['o'].shift(1))
        )
        
        patterns['bearish_engulfing'] = (
            (df['c'].shift(1) > df['o'].shift(1)) & 
            (df['c'] < df['o']) & 
            (df['o'] > df['c'].shift(1)) & 
            (df['c'] < df['o'].shift(1))
        )
        
        # Morning/Evening Star
        patterns['morning_star'] = (
            (df['c'].shift(2) < df['o'].shift(2)) & 
            (abs(df['c'].shift(1) - df['o'].shift(1)) < abs(df['c'].shift(2) - df['o'].shift(2)) * 0.3) & 
            (df['c'] > df['o']) & 
            (df['c'] > (df['c'].shift(2) + df['o'].shift(2)) / 2)
        )
        
        return patterns
    
    def train_model(self, symbol, df, features):
        """Train ML model for a symbol"""
        # Prepare data
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols].dropna().values
        y = features['target'].dropna().values
        
        if len(X) < 50:
            return None, None
        
        # Align X and y
        min_len = min(len(X), len(y))
        X = X[-min_len:]
        y = y[-min_len:]
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest (lighter than full ensemble)
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        # Train on all data
        model.fit(X_scaled, y)
        accuracy = np.mean(scores)
        
        return model, scaler, accuracy
    
    def predict(self, model, scaler, features):
        """Make prediction"""
        if model is None:
            return "WAIT", 0
        
        feature_cols = [col for col in features.columns if col != 'target']
        X_latest = features[feature_cols].iloc[-1:].values
        X_scaled = scaler.transform(X_latest)
        
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = max(proba) * 100
        
        signal = "LONG" if pred == 1 else "SHORT"
        return signal, confidence
    
    def generate_explanation(self, symbol, df, signal, confidence, patterns):
        """Generate trade explanation"""
        last = df.iloc[-1]
        
        explanation = f"""
## 🎯 TRADE SIGNAL: {signal} - {symbol}
**Confidence:** {confidence:.1f}% | **Price:** ${last['c']:.4f}

### 📊 Technical Analysis
- **RSI (14):** {df['rsi'].iloc[-1]:.1f} - {'Oversold' if df['rsi'].iloc[-1] < 30 else 'Overbought' if df['rsi'].iloc[-1] > 70 else 'Neutral'}
- **MACD:** {'Bullish' if df['macd_hist'].iloc[-1] > 0 else 'Bearish'}
- **Volume Ratio:** {df['volume_ratio'].iloc[-1]:.2f}x average

### 🔍 Detected Patterns
"""
        
        active_patterns = [p for p, active in patterns.items() 
                          if hasattr(active, 'iloc') and active.iloc[-1]]
        
        for pattern in active_patterns[:5]:
            explanation += f"- ✅ {pattern.replace('_', ' ').title()}\n"
        
        if not active_patterns:
            explanation += "- No strong patterns detected\n"
        
        # Risk management
        if signal == "LONG":
            sl = last['c'] * 0.98
            tp1 = last['c'] * 1.03
            tp2 = last['c'] * 1.05
        else:
            sl = last['c'] * 1.02
            tp1 = last['c'] * 0.97
            tp2 = last['c'] * 0.95
        
        explanation += f"""
### 💰 Risk Management
- **Entry:** ${last['c']:.4f}
- **Stop Loss:** ${sl:.4f} (2% risk)
- **Take Profit 1:** ${tp1:.4f} (R:R = 1:1.5)
- **Take Profit 2:** ${tp2:.4f} (R:R = 1:2.5)

### ⚠️ Recommendation
"""
        
        if confidence >= 80:
            explanation += "**STRONG SIGNAL** - Consider full position with tight stops"
        elif confidence >= 65:
            explanation += "**MODERATE SIGNAL** - Consider half position"
        else:
            explanation += "**WEAK SIGNAL** - Wait for confirmation or pass"
        
        return explanation

# Sidebar
with st.sidebar:
    st.markdown("## 🎮 Controls")
    
    # AI Mode
    ai_mode = st.radio(
        "AI Mode",
        ["A1 Mode (High Accuracy)", "Trading Mode (Balanced)", "Learning Mode"],
        index=1
    )
    
    confidence_threshold = {
        "A1 Mode (High Accuracy)": 80,
        "Trading Mode (Balanced)": 65,
        "Learning Mode": 50
    }[ai_mode]
    
    # Settings
    st.markdown("### ⚙️ Settings")
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    max_pairs = st.slider("Max Pairs to Scan", 10, 200, 50, 
                          help="Limit for Streamlit Cloud performance")
    
    if st.button("🚀 Start Scan", use_container_width=True):
        st.session_state.scanning = True
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.models = {}
        st.success("Cache cleared!")

# Main title
st.markdown('<h1 class="main-title">🤖 MEXC Quantum AI Scanner</h1>', unsafe_allow_html=True)

# Initialize
mexc = MEXCHandler()
ai = QuantumAI()

# Get all pairs on first load
if not st.session_state.all_symbols:
    with st.spinner("Fetching MEXC trading pairs..."):
        st.session_state.all_symbols = mexc.get_all_usdt_pairs()
        st.success(f"Found {len(st.session_state.all_symbols)} trading pairs")

# Main scanning logic
if st.session_state.scanning:
    # Limit pairs for performance
    pairs_to_scan = st.session_state.all_symbols[:max_pairs]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    signals = []
    
    for idx, symbol in enumerate(pairs_to_scan):
        try:
            status_text.text(f"Analyzing {idx+1}/{len(pairs_to_scan)}: {symbol}")
            
            # Fetch data
            df = mexc.fetch_ohlcv(symbol, timeframe, limit=200)
            if df is None or len(df) < 50:
                continue
            
            # Extract features
            features = ai.extract_features(df)
            
            # Train or load model
            if symbol not in st.session_state.models:
                model, scaler, accuracy = ai.train_model(symbol, df, features)
                if model:
                    st.session_state.models[symbol] = model
                    st.session_state.scalers[symbol] = scaler
            else:
                model = st.session_state.models.get(symbol)
                scaler = st.session_state.scalers.get(symbol)
            
            # Detect patterns
            patterns = ai.detect_patterns(df)
            
            # Make prediction
            if model and scaler:
                signal, confidence = ai.predict(model, scaler, features)
                
                # Check confidence threshold
                if confidence >= confidence_threshold:
                    signals.append({
                        'symbol': symbol.replace('/USDT:USDT', ''),
                        'price': df['c'].iloc[-1],
                        'signal': signal,
                        'confidence': confidence,
                        'patterns': [p for p, a in patterns.items() 
                                   if hasattr(a, 'iloc') and a.iloc[-1]][:3]
                    })
            
        except Exception as e:
            logger.error(f"Error with {symbol}: {e}")
        
        progress_bar.progress((idx + 1) / len(pairs_to_scan))
    
    status_text.empty()
    progress_bar.empty()
    st.session_state.scanning = False
    
    # Display results
    if signals:
        st.success(f"✅ Found {len(signals)} trading signals!")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Display top signals
        st.subheader("🔥 Top Signals")
        cols = st.columns(3)
        
        for i, signal in enumerate(signals[:6]):
            with cols[i % 3]:
                if signal['signal'] == 'LONG':
                    st.markdown(f"""
                    <div class="signal-long">
                        <h3>{signal['symbol']}</h3>
                        <h2>{signal['signal']}</h2>
                        <p>${signal['price']:.4f}</p>
                        <p>Confidence: {signal['confidence']:.1f}%</p>
                        <p>Patterns: {', '.join(signal['patterns'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-short">
                        <h3>{signal['symbol']}</h3>
                        <h2>{signal['signal']}</h2>
                        <p>${signal['price']:.4f}</p>
                        <p>Confidence: {signal['confidence']:.1f}%</p>
                        <p>Patterns: {', '.join(signal['patterns'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Full table
        st.subheader("📊 All Signals")
        signals_df = pd.DataFrame(signals)
        signals_df['price'] = signals_df['price'].apply(lambda x: f"${x:.4f}")
        signals_df['confidence'] = signals_df['confidence'].apply(lambda x: f"{x:.1f}%")
        signals_df['patterns'] = signals_df['patterns'].apply(lambda x: ', '.join(x))
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
        
        # Detailed explanation for first signal
        if signals:
            st.subheader("📝 Sample Trade Explanation")
            first = signals[0]
            
            # Get full data for explanation
            df = mexc.fetch_ohlcv(first['symbol'] + "/USDT:USDT", timeframe, limit=100)
            features = ai.extract_features(df)
            patterns = ai.detect_patterns(df)
            
            explanation = ai.generate_explanation(
                first['symbol'], 
                df, 
                first['signal'], 
                first['confidence'],
                patterns
            )
            
            st.markdown(f'<div class="explanation-box">{explanation}</div>', 
                       unsafe_allow_html=True)
    else:
        st.info("No signals found. Try lowering confidence threshold or changing timeframe.")

# Stats row
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Pairs", len(st.session_state.all_symbols))
with col2:
    st.metric("Trained Models", len(st.session_state.models))
with col3:
    st.metric("Timeframe", timeframe)
with col4:
    st.metric("AI Mode", ai_mode)

# Auto-refresh option
if st.checkbox("Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🤖 Free AI Trading Scanner | No API Keys Required | All ML Runs Locally</p>
    <p>⚠️ Warning: Never trade more than you can afford to lose</p>
</div>
""", unsafe_allow_html=True)
