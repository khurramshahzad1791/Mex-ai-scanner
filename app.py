import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="MEXC AI Scanner", 
    layout="wide", 
    page_icon="🤖"
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
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🤖 MEXC AI SCANNER</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # AI Mode
    ai_mode = st.radio(
        "AI Mode",
        ["A1 Mode (High Accuracy)", "Trading Mode (Balanced)", "Learning Mode"],
        index=1
    )
    
    # Confidence threshold based on mode
    if ai_mode == "A1 Mode (High Accuracy)":
        min_confidence = 80
        st.info("🎯 Only showing high probability setups (80%+ confidence)")
    elif ai_mode == "Trading Mode (Balanced)":
        min_confidence = 65
        st.info("⚖️ Balanced approach - good signals")
    else:
        min_confidence = 50
        st.info("📚 Learning mode - collecting data")
    
    # Timeframe
    timeframe = st.selectbox(
        "Timeframe",
        ["15m", "1h", "4h", "1d"],
        index=1
    )
    
    # Number of coins
    max_coins = st.slider("Coins to scan", 10, 100, 50)
    
    # Start button
    scan_button = st.button("🚀 START SCAN", use_container_width=True)

# Initialize MEXC connection
@st.cache_resource
def init_exchange():
    return ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

exchange = init_exchange()

# Get top coins by volume
@st.cache_data(ttl=3600)
def get_top_coins(limit=100):
    try:
        markets = exchange.load_markets()
        usdt_pairs = [s for s in markets if '/USDT:USDT' in s]
        return usdt_pairs[:limit]
    except:
        # Fallback to common coins
        return [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
            "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
            "DOT/USDT:USDT", "LINK/USDT:USDT", "MATIC/USDT:USDT", "UNI/USDT:USDT"
        ][:limit]

# Feature engineering
def create_features(df):
    df = df.copy()
    
    # Price features
    df['returns_1'] = df['c'].pct_change(1)
    df['returns_5'] = df['c'].pct_change(5)
    df['returns_10'] = df['c'].pct_change(10)
    
    # Moving averages
    df['sma_20'] = df['c'].rolling(20).mean()
    df['sma_50'] = df['c'].rolling(50).mean()
    df['price_to_sma'] = df['c'] / df['sma_20'] - 1
    
    # Volume
    df['volume_sma'] = df['v'].rolling(20).mean()
    df['volume_ratio'] = df['v'] / df['volume_sma']
    
    # RSI
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['c'].ewm(span=12).mean()
    exp2 = df['c'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Target (next candle direction)
    df['target'] = (df['c'].shift(-1) > df['c']).astype(int)
    
    return df.dropna()

# Pattern detection
def detect_patterns(df):
    patterns = []
    
    # Doji
    if abs(df['c'].iloc[-1] - df['o'].iloc[-1]) <= (df['h'].iloc[-1] - df['l'].iloc[-1]) * 0.1:
        patterns.append("Doji")
    
    # Hammer
    body = abs(df['c'].iloc[-1] - df['o'].iloc[-1])
    lower_shadow = min(df['o'].iloc[-1], df['c'].iloc[-1]) - df['l'].iloc[-1]
    if lower_shadow > body * 2:
        patterns.append("Hammer")
    
    # Engulfing
    if (df['c'].iloc[-2] < df['o'].iloc[-2] and 
        df['c'].iloc[-1] > df['o'].iloc[-1] and
        df['o'].iloc[-1] < df['c'].iloc[-2] and
        df['c'].iloc[-1] > df['o'].iloc[-2]):
        patterns.append("Bullish Engulfing")
    
    if (df['c'].iloc[-2] > df['o'].iloc[-2] and 
        df['c'].iloc[-1] < df['o'].iloc[-1] and
        df['o'].iloc[-1] > df['c'].iloc[-2] and
        df['c'].iloc[-1] < df['o'].iloc[-2]):
        patterns.append("Bearish Engulfing")
    
    return patterns

# Train model for a coin
def train_model(df):
    features = create_features(df)
    
    if len(features) < 30:
        return None, None
    
    feature_cols = ['returns_1', 'returns_5', 'returns_10', 'price_to_sma', 
                    'volume_ratio', 'rsi', 'macd', 'macd_signal']
    
    X = features[feature_cols].values[:-1]
    y = features['target'].values[:-1]
    
    if len(X) < 20:
        return None, None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    return model, scaler

# Predict
def predict(model, scaler, df):
    if model is None:
        return "WAIT", 0
    
    features = create_features(df)
    feature_cols = ['returns_1', 'returns_5', 'returns_10', 'price_to_sma', 
                    'volume_ratio', 'rsi', 'macd', 'macd_signal']
    
    X_latest = features[feature_cols].iloc[-1:].values
    X_scaled = scaler.transform(X_latest)
    
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = max(proba) * 100
    
    signal = "LONG" if pred == 1 else "SHORT"
    return signal, confidence

# Generate explanation
def generate_explanation(symbol, df, signal, confidence, patterns):
    last = df.iloc[-1]
    
    explanation = f"""
## 🎯 {symbol} - {signal} SIGNAL
**Confidence:** {confidence:.1f}% | **Price:** ${last['c']:.4f}

### 📊 Technical Analysis
- **RSI:** {df['rsi'].iloc[-1]:.1f}
- **Volume:** {df['volume_ratio'].iloc[-1]:.2f}x average
- **MACD:** {'Bullish' if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else 'Bearish'}

### 🔍 Patterns Detected
"""
    for pattern in patterns:
        explanation += f"- ✅ {pattern}\n"
    
    if not patterns:
        explanation += "- No clear patterns\n"
    
    # Risk levels
    if signal == "LONG":
        sl = last['c'] * 0.98
        tp = last['c'] * 1.04
    else:
        sl = last['c'] * 1.02
        tp = last['c'] * 0.96
    
    explanation += f"""
### 💰 Trade Plan
- **Entry:** ${last['c']:.4f}
- **Stop Loss:** ${sl:.4f} (2% risk)
- **Take Profit:** ${tp:.4f} (4% target)

### ⚠️ Verdict
"""
    if confidence >= 80:
        explanation += "**STRONG SIGNAL** - Consider entering now"
    elif confidence >= 65:
        explanation += "**MODERATE SIGNAL** - Consider half position"
    else:
        explanation += "**WEAK SIGNAL** - Wait for confirmation"
    
    return explanation

# Main scanning logic
if scan_button:
    coins = get_top_coins(max_coins)
    
    st.subheader(f"🔍 Scanning {len(coins)} coins...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    signals = []
    models = {}  # Store trained models
    scalers = {}
    
    for i, symbol in enumerate(coins):
        try:
            status_text.text(f"Analyzing {symbol}...")
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            
            # Train or use existing model
            if symbol not in models:
                model, scaler = train_model(df)
                if model:
                    models[symbol] = model
                    scalers[symbol] = scaler
            else:
                model = models.get(symbol)
                scaler = scalers.get(symbol)
            
            # Make prediction
            if model:
                signal, confidence = predict(model, scaler, df)
                patterns = detect_patterns(df)
                
                if confidence >= min_confidence:
                    signals.append({
                        'symbol': symbol.replace('/USDT:USDT', ''),
                        'price': df['c'].iloc[-1],
                        'signal': signal,
                        'confidence': confidence,
                        'patterns': patterns,
                        'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                        'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1
                    })
            
        except Exception as e:
            continue
        
        progress_bar.progress((i + 1) / len(coins))
    
    status_text.empty()
    progress_bar.empty()
    
    # Display results
    if signals:
        st.success(f"✅ Found {len(signals)} trading signals!")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Show top signals in columns
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
                        <p>RSI: {signal['rsi']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-short">
                        <h3>{signal['symbol']}</h3>
                        <h2>{signal['signal']}</h2>
                        <p>${signal['price']:.4f}</p>
                        <p>Confidence: {signal['confidence']:.1f}%</p>
                        <p>RSI: {signal['rsi']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show table
        st.subheader("📊 All Signals")
        df_show = pd.DataFrame(signals)
        df_show['price'] = df_show['price'].apply(lambda x: f"${x:.4f}")
        df_show['confidence'] = df_show['confidence'].apply(lambda x: f"{x:.1f}%")
        df_show['patterns'] = df_show['patterns'].apply(lambda x: ', '.join(x) if x else 'None')
        st.dataframe(df_show[['symbol', 'signal', 'confidence', 'price', 'rsi', 'patterns']], 
                    use_container_width=True, hide_index=True)
        
        # Detailed explanation for first signal
        st.subheader("📝 Sample Trade Analysis")
        first = signals[0]
        
        # Get full data for explanation
        symbol_full = first['symbol'] + "/USDT:USDT"
        ohlcv = exchange.fetch_ohlcv(symbol_full, timeframe, limit=100)
        df_detail = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        
        explanation = generate_explanation(
            first['symbol'],
            df_detail,
            first['signal'],
            first['confidence'],
            first['patterns']
        )
        
        st.markdown(f'<div class="explanation-box">{explanation}</div>', 
                   unsafe_allow_html=True)
        
    else:
        st.warning("No signals found. Try lowering confidence threshold or changing timeframe.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🤖 Free AI Scanner | No API Keys Required | Runs on Streamlit Cloud</p>
    <p>⚠️ Always use proper risk management</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()
