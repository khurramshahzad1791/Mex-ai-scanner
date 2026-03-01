"""
MEXC AI Trading Assistant - Complete Streamlit Solution
NO API KEYS REQUIRED - 100% FREE!
Self-learning ML + Interactive Q&A + Profit Calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ccxt
import requests
from datetime import datetime, timedelta
import time
import sqlite3
import pickle
import json
import hashlib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Technical Analysis
import ta

# Page config
st.set_page_config(
    page_title="MEXC AI Trading Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
        font-family: 'Arial Black', sans-serif;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #888;
        margin-bottom: 30px;
    }
    .signal-long {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-short {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-hold {
        background: linear-gradient(135deg, #808080, #404040);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: white;
        margin: 10px 0;
    }
    .profit-green {
        color: #00ff00;
        font-weight: bold;
        font-size: 24px;
    }
    .loss-red {
        color: #ff4444;
        font-weight: bold;
        font-size: 24px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background: #1e1e1e;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background: #2d2d2d;
        border-left: 4px solid #00b09b;
    }
    .assistant-message {
        background: #1e1e1e;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        border-radius: 5px;
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

# ============================================================================
# DATABASE SETUP - Self-learning memory
# ============================================================================

def init_database():
    """Initialize SQLite database for learning memory"""
    conn = sqlite3.connect('trading_assistant.db')
    c = conn.cursor()
    
    # Store learned patterns
    c.execute('''
        CREATE TABLE IF NOT EXISTS learned_patterns (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            pattern_name TEXT,
            feature_vector BLOB,
            outcome INTEGER,
            profit_loss REAL,
            timestamp DATETIME,
            confidence REAL
        )
    ''')
    
    # Store model performance
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            symbol TEXT,
            model_type TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            last_trained DATETIME,
            num_samples INTEGER,
            PRIMARY KEY (symbol, model_type)
        )
    ''')
    
    # Store Q&A history for learning
    c.execute('''
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT,
            symbol TEXT,
            context TEXT,
            user_feedback INTEGER,
            timestamp DATETIME
        )
    ''')
    
    # Store trade history
    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            signal TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            profit_loss REAL,
            profit_loss_pct REAL,
            confidence REAL,
            entry_time DATETIME,
            exit_time DATETIME,
            notes TEXT
        )
    ''')
    
    conn.commit()
    return conn

# Initialize database
if 'db' not in st.session_state:
    st.session_state.db = init_database()
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# FREE DATA SOURCES (No API Keys Required)
# ============================================================================

class FreeDataFetcher:
    """Fetch market data from free public sources"""
    
    def __init__(self):
        # Initialize free exchange connections
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'mexc': ccxt.mexc({'enableRateLimit': True}),
            'kucoin': ccxt.kucoin({'enableRateLimit': True}),
            'bybit': ccxt.bybit({'enableRateLimit': True})
        }
        
        # Free API endpoints
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.coinpaprika_api = "https://api.coinpaprika.com/v1"
        
    def get_mexc_symbols(self) -> List[str]:
        """Get all MEXC trading pairs (free)"""
        try:
            markets = self.exchanges['mexc'].load_markets()
            symbols = [s for s in markets if '/USDT' in s]
            return symbols[:100]  # Limit for performance
        except:
            # Fallback to common pairs
            return [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
                "XRP/USDT", "ADA/USDT", "DOGE/USDT", "DOT/USDT",
                "LINK/USDT", "MATIC/USDT", "AVAX/USDT", "UNI/USDT"
            ]
    
    def get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get historical data from multiple free sources"""
        
        # Try MEXC first
        try:
            ohlcv = self.exchanges['mexc'].fetch_ohlcv(
                symbol, '1h', limit=min(days * 24, 1000)
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
            return df
        except:
            pass
        
        # Try CoinGecko as backup
        try:
            coin_id = symbol.split('/')[0].lower()
            url = f"{self.coingecko_api}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['o'] = df['h'] = df['l'] = df['c'] = df['close']
                df['v'] = [v[1] for v in data['total_volumes']] if 'total_volumes' in data else 0
                return df
        except:
            pass
        
        return None
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from free sources"""
        try:
            ticker = self.exchanges['mexc'].fetch_ticker(symbol)
            return ticker['last']
        except:
            try:
                # Try CoinGecko
                coin_id = symbol.split('/')[0].lower()
                url = f"{self.coingecko_api}/simple/price"
                params = {
                    'ids': coin_id,
                    'vs_currencies': 'usd'
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return data[coin_id]['usd']
            except:
                pass
        return 0

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Create 50+ technical features for ML"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df = df.copy()
        
        # Price-based features
        for period in [1, 3, 5, 10, 20, 50]:
            df[f'return_{period}'] = df['c'].pct_change(period)
            df[f'volatility_{period}'] = df['c'].pct_change().rolling(period).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['c'], period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['c'], period)
            df[f'price_to_sma_{period}'] = df['c'] / df[f'sma_{period}'] - 1
        
        # RSI (multiple periods)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['c'], period)
        
        # MACD
        macd = ta.trend.MACD(df['c'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['c'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['c']
        df['bb_position'] = (df['c'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volume
        df['volume_sma'] = df['v'].rolling(20).mean()
        df['volume_ratio'] = df['v'] / df['volume_sma']
        df['volume_change'] = df['v'].pct_change()
        
        # Price action
        df['high_low_ratio'] = (df['h'] - df['l']) / df['c']
        df['close_position'] = (df['c'] - df['l']) / (df['h'] - df['l'])
        df['candle_body'] = abs(df['c'] - df['o']) / (df['h'] - df['l'])
        
        # Target (next period direction)
        df['target'] = (df['c'].shift(-1) > df['c']).astype(int)
        df['target_return'] = df['c'].shift(-1) / df['c'] - 1
        
        return df.dropna()

# ============================================================================
# SELF-LEARNING ML ENGINE
# ============================================================================

class SelfLearningEngine:
    """ML models that learn from market data and improve over time"""
    
    def __init__(self, db_conn):
        self.db = db_conn
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_or_update_model(self, symbol: str, df: pd.DataFrame, features: pd.DataFrame):
        """Train or update model with new data"""
        
        feature_cols = [col for col in features.columns if col not in ['target', 'target_return']]
        X = features[feature_cols].values[:-1]
        y = features['target'].values[:-1]
        
        if len(X) < 50:
            return None
        
        # Remove NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            return None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train multiple models
        models = {}
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_scaled, y)
        rf_score = rf.score(X_scaled, y)
        models['RandomForest'] = {'model': rf, 'accuracy': rf_score}
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_scaled, y)
        xgb_score = xgb_model.score(X_scaled, y)
        models['XGBoost'] = {'model': xgb_model, 'accuracy': xgb_score}
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_scaled, y)
        gb_score = gb.score(X_scaled, y)
        models['GradientBoosting'] = {'model': gb, 'accuracy': gb_score}
        
        # Store models
        self.models[symbol] = models
        self.scalers[symbol] = scaler
        
        # Save feature importance
        if hasattr(rf, 'feature_importances_'):
            self.feature_importance[symbol] = dict(zip(feature_cols, rf.feature_importances_))
        
        # Save to database
        cursor = self.db.cursor()
        for model_name, model_data in models.items():
            cursor.execute('''
                INSERT OR REPLACE INTO model_performance 
                (symbol, model_type, accuracy, last_trained, num_samples)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, model_name, model_data['accuracy'], datetime.now(), len(X)))
        
        self.db.commit()
        
        return models
    
    def predict(self, symbol: str, features: pd.DataFrame) -> Tuple[str, float, float, Dict]:
        """Make prediction with confidence and expected return"""
        
        if symbol not in self.models:
            return "HOLD", 0, 0, {}
        
        feature_cols = [col for col in features.columns if col not in ['target', 'target_return']]
        X_latest = features[feature_cols].iloc[-1:].values
        
        if symbol in self.scalers:
            X_scaled = self.scalers[symbol].transform(X_latest)
        else:
            return "HOLD", 0, 0, {}
        
        predictions = []
        confidences = []
        expected_returns = []
        
        for model_name, model_data in self.models[symbol].items():
            model = model_data['model']
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                pred = 1 if proba[1] > 0.5 else 0
                conf = max(proba) * 100
            else:
                pred = model.predict(X_scaled)[0]
                conf = model_data['accuracy'] * 100
            
            predictions.append(pred)
            confidences.append(conf)
            
            # Estimate expected return based on historical patterns
            expected_return = features['target_return'].mean() * (conf / 100)
            expected_returns.append(expected_return)
        
        # Ensemble voting
        final_pred = 1 if np.mean(predictions) > 0.5 else 0
        final_conf = np.mean(confidences)
        final_expected_return = np.mean(expected_returns) * 100  # as percentage
        
        signal = "LONG" if final_pred == 1 else "SHORT" if final_pred == 0 else "HOLD"
        
        model_votes = {
            name: ('LONG' if pred == 1 else 'SHORT') 
            for name, pred in zip(self.models[symbol].keys(), predictions)
        }
        
        return signal, final_conf, final_expected_return, model_votes

# ============================================================================
# AI ASSISTANT FOR Q&A
# ============================================================================

class TradingAssistant:
    """AI assistant that answers questions about market conditions"""
    
    def __init__(self, db_conn, ml_engine):
        self.db = db_conn
        self.ml_engine = ml_engine
        self.data_fetcher = FreeDataFetcher()
        
    def analyze_market(self, symbol: str, question: str) -> str:
        """Analyze market and answer questions"""
        
        # Get data
        df = self.data_fetcher.get_historical_data(symbol, days=30)
        if df is None:
            return "I couldn't fetch data for this symbol. Please try another."
        
        # Create features
        fe = FeatureEngineer()
        features = fe.create_features(df)
        
        # Get ML prediction
        signal, confidence, expected_return, votes = self.ml_engine.predict(symbol, features)
        
        # Calculate key metrics
        last_price = df['c'].iloc[-1]
        price_change_24h = (df['c'].iloc[-1] / df['c'].iloc[-24] - 1) * 100 if len(df) > 24 else 0
        volume_ratio = df['v'].iloc[-1] / df['v'].tail(20).mean()
        rsi = features['rsi_14'].iloc[-1] if 'rsi_14' in features.columns else 50
        
        # Support/resistance levels
        recent_high = df['h'].tail(20).max()
        recent_low = df['l'].tail(20).min()
        
        # Generate answer based on question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['buy', 'entry', 'enter', 'long']):
            if signal == "LONG" and confidence > 60:
                return self._generate_buy_answer(symbol, last_price, confidence, expected_return, recent_low, recent_high)
            elif signal == "SHORT":
                return f"📉 I wouldn't buy {symbol} right now. The model predicts a downtrend with {confidence:.1f}% confidence. Consider waiting for a better entry."
            else:
                return f"🤔 It's uncertain for {symbol} right now. Confidence is only {confidence:.1f}%. Better to wait for clearer signals."
        
        elif any(word in question_lower for word in ['sell', 'exit', 'short']):
            if signal == "SHORT" and confidence > 60:
                return self._generate_sell_answer(symbol, last_price, confidence, expected_return, recent_low, recent_high)
            elif signal == "LONG":
                return f"📈 I wouldn't sell {symbol} right now. The model predicts an uptrend with {confidence:.1f}% confidence. Consider holding."
            else:
                return f"🤔 It's uncertain for {symbol} right now. Confidence is only {confidence:.1f}%. Better to wait."
        
        elif any(word in question_lower for word in ['profit', 'target', 'gain', 'make']):
            return self._generate_profit_answer(symbol, last_price, signal, confidence, expected_return)
        
        elif any(word in question_lower for word in ['risk', 'loss', 'stop']):
            return self._generate_risk_answer(symbol, last_price, recent_low, recent_high, rsi)
        
        else:
            # General analysis
            return self._generate_general_analysis(symbol, last_price, signal, confidence, 
                                                  price_change_24h, volume_ratio, rsi, 
                                                  recent_low, recent_high)
    
    def _generate_buy_answer(self, symbol, price, confidence, expected_return, support, resistance):
        return f"""
🎯 **BUY SIGNAL for {symbol}**

✅ **Action:** Consider BUYING at current price
📊 **Confidence:** {confidence:.1f}%
💰 **Expected Return:** +{expected_return:.2f}% in next period

**Key Levels:**
- Current Price: ${price:.4f}
- Support Level: ${support:.4f}
- Resistance: ${resistance:.4f}

**Entry Strategy:**
- Aggressive: Buy now at ${price:.4f}
- Conservative: Wait for pullback to ${support:.4f}

**Risk Management:**
- Stop Loss: ${support * 0.98:.4f} (2% below support)
- Take Profit 1: ${price * 1.03:.4f} (+3%)
- Take Profit 2: ${price * 1.05:.4f} (+5%)

**If you invest $100:**
- At TP1: You'd have ${100 * 1.03:.2f} (+$3 profit)
- At TP2: You'd have ${100 * 1.05:.2f} (+$5 profit)
"""
    
    def _generate_sell_answer(self, symbol, price, confidence, expected_return, support, resistance):
        return f"""
📉 **SELL SIGNAL for {symbol}**

⚠️ **Action:** Consider SELLING/Shorting at current price
📊 **Confidence:** {confidence:.1f}%
💰 **Expected Return:** {expected_return:.2f}% in next period

**Key Levels:**
- Current Price: ${price:.4f}
- Support: ${support:.4f}
- Resistance: ${resistance:.4f}

**Exit Strategy:**
- Aggressive: Sell now at ${price:.4f}
- Conservative: Wait for bounce to ${resistance:.4f}

**Risk Management:**
- Stop Loss: ${resistance * 1.02:.4f} (2% above resistance)
- Take Profit 1: ${price * 0.97:.4f} (-3%)
- Take Profit 2: ${price * 0.95:.4f} (-5%)

**If you short $100:**
- At TP1: You'd have ${100 * 1.03:.2f} (+$3 profit)
- At TP2: You'd have ${100 * 1.05:.2f} (+$5 profit)
"""
    
    def _generate_profit_answer(self, symbol, price, signal, confidence, expected_return):
        scenarios = []
        for investment in [100, 500, 1000]:
            profit = investment * (expected_return / 100)
            scenarios.append(f"- ${investment} → ${investment + profit:.2f} (${profit:.2f} profit)")
        
        return f"""
💰 **PROFIT POTENTIAL for {symbol}**

**Current Signal:** {signal} ({confidence:.1f}% confidence)
**Expected Return:** {expected_return:.2f}%

**Profit Scenarios:**
{chr(10).join(scenarios)}

**Risk-Adjusted Returns:**
- If signal correct ({confidence:.1f}% chance): +{expected_return:.2f}%
- If signal wrong ({100-confidence:.1f}% chance): -2.0% (stop loss)

**Expected Value:**
{((confidence/100) * expected_return - ((100-confidence)/100) * 2):.2f}%
"""
    
    def _generate_risk_answer(self, symbol, price, support, resistance, rsi):
        risk_level = "HIGH" if rsi > 70 or rsi < 30 else "MEDIUM" if rsi > 60 or rsi < 40 else "LOW"
        
        return f"""
⚠️ **RISK ANALYSIS for {symbol}**

**Current Risk Level:** {risk_level}

**Key Risk Metrics:**
- RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
- Distance to Support: {(price - support)/price * 100:.2f}%
- Distance to Resistance: {(resistance - price)/price * 100:.2f}%

**Recommended Position Size:**
- Conservative: 1% of portfolio
- Moderate: 2% of portfolio
- Aggressive: 3% of portfolio

**Stop Loss Levels:**
- Tight: ${price * 0.98:.4f} (2% loss)
- Normal: ${price * 0.95:.4f} (5% loss)
- Wide: ${price * 0.90:.4f} (10% loss)
"""
    
    def _generate_general_analysis(self, symbol, price, signal, confidence, change_24h, volume_ratio, rsi, support, resistance):
        trend = "BULLISH 📈" if signal == "LONG" else "BEARISH 📉" if signal == "SHORT" else "NEUTRAL ➡️"
        
        return f"""
📊 **MARKET ANALYSIS for {symbol}**

**Current Status:**
- Price: ${price:.4f}
- 24h Change: {change_24h:.2f}%
- Volume: {volume_ratio:.2f}x average
- RSI: {rsi:.1f}

**AI Prediction:**
- Trend: {trend}
- Confidence: {confidence:.1f}%
- Support: ${support:.4f}
- Resistance: ${resistance:.4f}

**Market Sentiment:** {
    'Strong Buying Pressure' if rsi > 60 and signal == 'LONG' else
    'Strong Selling Pressure' if rsi < 40 and signal == 'SHORT' else
    'Mixed/Neutral'
}

**What to Watch:**
- Break above ${resistance:.4f} → Bullish
- Break below ${support:.4f} → Bearish
- Volume spike > 2x → Trend confirmation
"""

# ============================================================================
# PROFIT CALCULATOR
# ============================================================================

class ProfitCalculator:
    """Calculate potential profits based on signals"""
    
    @staticmethod
    def calculate_profit(investment: float, entry_price: float, exit_price: float, 
                         side: str = "LONG") -> Dict:
        """Calculate profit/loss for a trade"""
        
        if side == "LONG":
            profit_pct = (exit_price - entry_price) / entry_price * 100
            profit_amount = investment * (profit_pct / 100)
            exit_value = investment + profit_amount
        else:  # SHORT
            profit_pct = (entry_price - exit_price) / entry_price * 100
            profit_amount = investment * (profit_pct / 100)
            exit_value = investment + profit_amount
        
        return {
            'investment': investment,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'exit_value': exit_value,
            'side': side
        }
    
    @staticmethod
    def scenario_analysis(investment: float, entry_price: float, 
                         target_pcts: List[float], stop_pct: float) -> pd.DataFrame:
        """Analyze different profit/loss scenarios"""
        
        scenarios = []
        
        for target_pct in target_pcts:
            if target_pct > 0:  # Profit scenarios
                exit_price = entry_price * (1 + target_pct/100)
                result = ProfitCalculator.calculate_profit(investment, entry_price, exit_price, "LONG")
                scenarios.append({
                    'Scenario': f'Target +{target_pct}%',
                    'Exit Price': f'${exit_price:.4f}',
                    'Profit/Loss': f'+{result["profit_amount"]:.2f}',
                    'ROI': f'+{result["profit_pct"]:.1f}%'
                })
        
        # Stop loss scenario
        stop_price = entry_price * (1 - stop_pct/100)
        stop_result = ProfitCalculator.calculate_profit(investment, entry_price, stop_price, "LONG")
        scenarios.append({
            'Scenario': f'Stop Loss -{stop_pct}%',
            'Exit Price': f'${stop_price:.4f}',
            'Profit/Loss': f'{stop_result["profit_amount"]:.2f}',
            'ROI': f'{stop_result["profit_pct"]:.1f}%'
        })
        
        return pd.DataFrame(scenarios)

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Initialize components
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = FreeDataFetcher()
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'ml_engine' not in st.session_state:
    st.session_state.ml_engine = SelfLearningEngine(st.session_state.db)
if 'assistant' not in st.session_state:
    st.session_state.assistant = TradingAssistant(st.session_state.db, st.session_state.ml_engine)

# Main title
st.markdown('<h1 class="main-title">🤖 MEXC AI Trading Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Self-Learning ML | No API Keys Required | Ask Me Anything About Crypto</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header("📊 Controls")
    
    # Symbol selection
    symbols = st.session_state.data_fetcher.get_mexc_symbols()
    selected_symbol = st.selectbox(
        "Select Trading Pair",
        symbols,
        index=0,
        format_func=lambda x: x.replace('/USDT', '')
    )
    
    st.divider()
    
    # Investment amount
    investment = st.number_input(
        "Investment Amount (USDT)",
        min_value=10,
        max_value=100000,
        value=100,
        step=10
    )
    
    st.divider()
    
    # Timeframe
    timeframe = st.selectbox(
        "Analysis Timeframe",
        ["1h", "4h", "1d", "1w"],
        index=1
    )
    
    st.divider()
    
    # Model stats
    st.subheader("🧠 Model Stats")
    st.metric("Trained Models", len(st.session_state.ml_engine.models))
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 LIVE SIGNALS", 
    "💬 AI ASSISTANT", 
    "💰 PROFIT CALCULATOR",
    "📚 LEARNING DASHBOARD"
])

# ============================================================================
# TAB 1: LIVE SIGNALS
# ============================================================================

with tab1:
    st.subheader(f"🎯 Live Analysis for {selected_symbol}")
    
    with st.spinner("Fetching data and running AI models..."):
        # Get data
        df = st.session_state.data_fetcher.get_historical_data(selected_symbol, days=30)
        
        if df is not None and len(df) > 50:
            # Create features
            features = st.session_state.feature_engineer.create_features(df)
            
            # Train or update model
            if selected_symbol not in st.session_state.ml_engine.models:
                st.session_state.ml_engine.train_or_update_model(selected_symbol, df, features)
            
            # Get prediction
            signal, confidence, expected_return, votes = st.session_state.ml_engine.predict(selected_symbol, features)
            
            # Display signal
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if signal == "LONG":
                    st.markdown(f"""
                    <div class="signal-long">
                        <h2>{signal}</h2>
                        <h3>Confidence: {confidence:.1f}%</h3>
                        <p>Expected Return: +{expected_return:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif signal == "SHORT":
                    st.markdown(f"""
                    <div class="signal-short">
                        <h2>{signal}</h2>
                        <h3>Confidence: {confidence:.1f}%</h3>
                        <p>Expected Return: {expected_return:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-hold">
                        <h2>{signal}</h2>
                        <h3>Confidence: {confidence:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Current Price", f"${df['c'].iloc[-1]:.4f}")
                st.metric("24h Change", f"{(df['c'].iloc[-1]/df['c'].iloc[-24]-1)*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Volume Ratio", f"{df['v'].iloc[-1]/df['v'].tail(20).mean():.2f}x")
                st.metric("RSI", f"{features['rsi_14'].iloc[-1]:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Model votes
            st.subheader("🤖 AI Model Votes")
            votes_df = pd.DataFrame([
                {"Model": model, "Vote": vote}
                for model, vote in votes.items()
            ])
            st.dataframe(votes_df, use_container_width=True, hide_index=True)
            
            # Price chart
            st.subheader("📈 Price Chart")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3])
            
            fig.add_trace(
                go.Candlestick(
                    x=df['ts'].tail(100),
                    open=df['o'].tail(100),
                    high=df['h'].tail(100),
                    low=df['l'].tail(100),
                    close=df['c'].tail(100),
                    name="Price"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df['ts'].tail(100),
                    y=df['v'].tail(100),
                    name="Volume"
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Unable to fetch data. Please try another symbol.")

# ============================================================================
# TAB 2: AI ASSISTANT (Q&A)
# ============================================================================

with tab2:
    st.subheader("💬 Ask Me Anything About the Market")
    
    st.markdown("""
    <div class="info-box">
        <p>🤖 I can answer questions like:</p>
        <ul>
            <li>"Should I buy BTC now?"</li>
            <li>"What's the profit potential for ETH?"</li>
            <li>"Is it risky to enter SOL?"</li>
            <li>"How much can I make with $100 on XRP?"</li>
            <li>"When should I sell my DOGE?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., Should I buy BTC now?",
        key="question_input"
    )
    
    if st.button("Ask AI Assistant", use_container_width=True):
        if user_question:
            with st.spinner("Analyzing market data..."):
                # Get answer
                answer = st.session_state.assistant.analyze_market(selected_symbol, user_question)
                
                # Display chat
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {user_question}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Store in history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': answer,
                    'symbol': selected_symbol,
                    'timestamp': datetime.now()
                })
    
    # Show chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📜 Chat History")
        
        for chat in st.session_state.chat_history[-5:]:  # Show last 5
            st.markdown(f"""
            <div class="chat-message user-message">
                <small>{chat['timestamp'].strftime('%H:%M:%S')}</small><br>
                <strong>Q:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: PROFIT CALCULATOR
# ============================================================================

with tab3:
    st.subheader("💰 Profit Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        calc_investment = st.number_input(
            "Investment (USDT)",
            min_value=10,
            max_value=10000,
            value=investment,
            step=10,
            key="calc_invest"
        )
        
        entry_price = st.number_input(
            "Entry Price (USDT)",
            min_value=0.000001,
            value=0.0,
            format="%.4f",
            key="entry_price"
        )
        
        exit_price = st.number_input(
            "Exit Price (USDT)",
            min_value=0.000001,
            value=0.0,
            format="%.4f",
            key="exit_price"
        )
        
        trade_side = st.radio("Trade Side", ["LONG", "SHORT"], horizontal=True)
        
        if st.button("Calculate Profit", use_container_width=True):
            if entry_price > 0 and exit_price > 0:
                result = ProfitCalculator.calculate_profit(
                    calc_investment, entry_price, exit_price, trade_side
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    
                    if result['profit_amount'] >= 0:
                        st.markdown(f"<p class='profit-green'>+${result['profit_amount']:.2f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='profit-green'>+{result['profit_pct']:.2f}% ROI</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='loss-red'>${result['profit_amount']:.2f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='loss-red'>{result['profit_pct']:.2f}% ROI</p>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Exit Value:** ${result['exit_value']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Scenario analysis
    st.subheader("📊 Scenario Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        target_pcts_input = st.text_input(
            "Target Profits (% - comma separated)",
            value="3,5,10",
            help="e.g., 3,5,10 for 3%, 5%, 10% targets"
        )
    
    with col4:
        stop_pct = st.number_input(
            "Stop Loss (%)",
            min_value=1.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )
    
    if entry_price > 0:
        target_pcts = [float(x.strip()) for x in target_pcts_input.split(',')]
        
        scenarios_df = ProfitCalculator.scenario_analysis(
            calc_investment, entry_price, target_pcts, stop_pct
        )
        
        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = go.Figure()
        
        for _, row in scenarios_df.iterrows():
            color = 'green' if '+' in row['Profit/Loss'] else 'red'
            fig.add_trace(go.Bar(
                x=[row['Scenario']],
                y=[float(row['Profit/Loss'].replace('+', ''))],
                name=row['Scenario'],
                marker_color=color
            ))
        
        fig.update_layout(
            title="Profit/Loss Scenarios",
            yaxis_title="Profit/Loss ($)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: LEARNING DASHBOARD
# ============================================================================

with tab4:
    st.subheader("📚 AI Learning Dashboard")
    
    # Model performance
    st.markdown("### 🤖 Model Performance")
    
    cursor = st.session_state.db.cursor()
    cursor.execute('''
        SELECT symbol, model_type, accuracy, last_trained, num_samples
        FROM model_performance
        ORDER BY last_trained DESC
        LIMIT 10
    ''')
    
    results = cursor.fetchall()
    
    if results:
        perf_df = pd.DataFrame(
            results,
            columns=['Symbol', 'Model', 'Accuracy', 'Last Trained', 'Samples']
        )
        perf_df['Accuracy'] = perf_df['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Accuracy chart
        fig = go.Figure()
        
        for model in perf_df['Model'].unique():
            model_data = perf_df[perf_df['Model'] == model]
            fig.add_trace(go.Bar(
                x=model_data['Symbol'],
                y=model_data['Accuracy'].str.rstrip('%').astype(float),
                name=model
            ))
        
        fig.update_layout(
            title="Model Accuracy by Symbol",
            yaxis_title="Accuracy (%)",
            template="plotly_dark",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model data yet. Run analyses to build the learning database.")
    
    # Feature importance
    if st.session_state.ml_engine.feature_importance:
        st.markdown("### 🔍 Feature Importance")
        
        symbol_for_features = st.selectbox(
            "Select Symbol",
            list(st.session_state.ml_engine.feature_importance.keys())
        )
        
        if symbol_for_features in st.session_state.ml_engine.feature_importance:
            importance = st.session_state.ml_engine.feature_importance[symbol_for_features]
            importance_df = pd.DataFrame([
                {'Feature': f, 'Importance': i}
                for f, i in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            ])
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='purple'
            ))
            
            fig.update_layout(
                title=f"Top 15 Features - {symbol_for_features}",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🤖 MEXC AI Trading Assistant - Self-Learning ML | No API Keys Required | 100% Free</p>
    <p>⚠️ Educational Purposes Only - Not Financial Advice</p>
    <p>Version 1.0 | Last Update: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()
