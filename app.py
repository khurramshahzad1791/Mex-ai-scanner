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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-short {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .explanation-box {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: white;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🤖 MEXC AI SCANNER</h1>', unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # AI Mode
    ai_mode = st.radio(
        "🤖 AI Mode",
        ["🔴 A1 Mode (High Accuracy)", "🟡 Trading Mode (Balanced)", "🟢 Learning Mode"],
        index=1
    )
    
    # Confidence threshold based on mode
    if "A1" in ai_mode:
        min_confidence = 80
        st.info("🎯 Only high probability setups (80%+ confidence)")
    elif "Trading" in ai_mode:
        min_confidence = 65
        st.info("⚖️ Balanced approach - good signals")
    else:
        min_confidence = 50
        st.info("📚 Learning mode - collecting data")
    
    st.divider()
    
    # Timeframe
    timeframe = st.selectbox(
        "⏰ Timeframe",
        ["15m", "1h", "4h", "1d"],
        index=1
    )
    
    # Number of coins
    max_coins = st.slider("📊 Coins to scan", 10, 50, 30, 
                          help="Lower = faster scanning")
    
    st.divider()
    
    # Start button
    scan_button = st.button("🚀 START SCAN", use_container_width=True)
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.models = {}
        st.session_state.scalers = {}
        st.rerun()

# Initialize MEXC connection
@st.cache_resource
def init_exchange():
    return ccxt.mexc({
        'enableRateLimit': True,
        'rateLimit': 1200,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })

exchange = init_exchange()

# Get top coins
@st.cache_data(ttl=3600)
def get_top_coins(limit=50):
    try:
        markets = exchange.load_markets()
        usdt_pairs = [s for s in markets if '/USDT:USDT' in s and markets[s]['active']]
        # Sort by volume if available, otherwise return first N
        return usdt_pairs[:limit]
    except Exception as e:
        st.error(f"Error fetching coins: {e}")
        # Fallback to common coins
        return [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
            "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
            "DOT/USDT:USDT", "LINK/USDT:USDT"
        ][:limit]

# Feature engineering
def create_features(df):
    """Create technical features for ML"""
    df = df.copy()
    
    # Price features
    df['returns_1'] = df['c'].pct_change(1)
    df['returns_3'] = df['c'].pct_change(3)
    df['returns_5'] = df['c'].pct_change(5)
    df['returns_10'] = df['c'].pct_change(10)
    
    # Moving averages
    df['sma_20'] = df['c'].rolling(20).mean()
    df['sma_50'] = df['c'].rolling(50).mean()
    df['ema_12'] = df['c'].ewm(span=12).mean()
    df['ema_26'] = df['c'].ewm(span=26).mean()
    
    # Price to MA ratios
    df['price_to_sma20'] = df['c'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['c'] / df['sma_50'] - 1
    
    # Volume
    df['volume_sma'] = df['v'].rolling(20).mean()
    df['volume_ratio'] = df['v'] / df['volume_sma']
    df['volume_change'] = df['v'].pct_change()
    
    # Volatility
    df['volatility'] = df['returns_1'].rolling(20).std()
    
    # RSI
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Price position
    df['high_low_ratio'] = (df['h'] - df['l']) / df['c']
    df['close_position'] = (df['c'] - df['l']) / (df['h'] - df['l'])
    
    # Target (next candle direction)
    df['target'] = (df['c'].shift(-1) > df['c']).astype(int)
    
    return df.dropna()

# Pattern detection
def detect_patterns(df):
    """Detect common candlestick patterns"""
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Doji
    body = abs(last['c'] - last['o'])
    range_ = last['h'] - last['l']
    if body <= range_ * 0.1:
        patterns.append("Doji")
    
    # Hammer
    lower_shadow = min(last['o'], last['c']) - last['l']
    if lower_shadow > body * 2 and body > 0:
        patterns.append("Hammer")
    
    # Shooting Star
    upper_shadow = last['h'] - max(last['o'], last['c'])
    if upper_shadow > body * 2 and body > 0:
        patterns.append("Shooting Star")
    
    # Bullish Engulfing
    if (prev['c'] < prev['o'] and 
        last['c'] > last['o'] and
        last['o'] < prev['c'] and
        last['c'] > prev['o']):
        patterns.append("Bullish Engulfing")
    
    # Bearish Engulfing
    if (prev['c'] > prev['o'] and 
        last['c'] < last['o'] and
        last['o'] > prev['c'] and
        last['c'] < prev['o']):
        patterns.append("Bearish Engulfing")
    
    # Three White Soldiers (simplified)
    if len(df) >= 3:
        if (df['c'].iloc[-3] > df['o'].iloc[-3] and
            df['c'].iloc[-2] > df['o'].iloc[-2] and
            df['c'].iloc[-1] > df['o'].iloc[-1] and
            df['c'].iloc[-1] > df['c'].iloc[-2] > df['c'].iloc[-3]):
            patterns.append("Three White Soldiers")
    
    return patterns

# Train model
def train_model(df):
    """Train Random Forest model"""
    try:
        features = create_features(df)
        
        if len(features) < 30:
            return None, None
        
        feature_cols = ['returns_1', 'returns_3', 'returns_5', 'price_to_sma20', 
                        'volume_ratio', 'volatility', 'rsi', 'macd_hist', 
                        'close_position']
        
        X = features[feature_cols].values[:-1]
        y = features['target'].values[:-1]
        
        if len(X) < 20:
            return None, None
        
        # Remove any remaining NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 20:
            return None, None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, y)
        
        return model, scaler
        
    except Exception as e:
        return None, None

# Predict
def predict(model, scaler, df):
    """Make prediction using trained model"""
    if model is None or scaler is None:
        return "WAIT", 0
    
    try:
        features = create_features(df)
        feature_cols = ['returns_1', 'returns_3', 'returns_5', 'price_to_sma20', 
                        'volume_ratio', 'volatility', 'rsi', 'macd_hist', 
                        'close_position']
        
        X_latest = features[feature_cols].iloc[-1:].values
        
        # Check for NaN
        if np.isnan(X_latest).any():
            return "WAIT", 0
        
        X_scaled = scaler.transform(X_latest)
        
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(max(proba) * 100)
        
        signal = "LONG" if pred == 1 else "SHORT"
        return signal, confidence
        
    except Exception as e:
        return "WAIT", 0

# Generate explanation
def generate_explanation(symbol, df, signal, confidence, patterns):
    """Generate detailed trade explanation"""
    last = df.iloc[-1]
    
    # Calculate support/resistance levels
    recent_low = df['l'].tail(20).min()
    recent_high = df['h'].tail(20).max()
    
    # Risk levels
    if signal == "LONG":
        sl = last['c'] * 0.98
        tp1 = last['c'] * 1.03
        tp2 = last['c'] * 1.05
        risk = (last['c'] - sl) / last['c'] * 100
        reward1 = (tp1 - last['c']) / last['c'] * 100
        reward2 = (tp2 - last['c']) / last['c'] * 100
    else:
        sl = last['c'] * 1.02
        tp1 = last['c'] * 0.97
        tp2 = last['c'] * 0.95
        risk = (sl - last['c']) / last['c'] * 100
        reward1 = (last['c'] - tp1) / last['c'] * 100
        reward2 = (last['c'] - tp2) / last['c'] * 100
    
    explanation = f"""
## 🎯 {symbol} - {signal} SIGNAL
**Confidence:** {confidence:.1f}% | **Price:** ${last['c']:.4f}

### 📊 Technical Analysis
- **RSI (14):** {df['rsi'].iloc[-1]:.1f}
- **MACD:** {'Bullish' if df['macd_hist'].iloc[-1] > 0 else 'Bearish'}
- **Volume:** {df['volume_ratio'].iloc[-1]:.2f}x average
- **Volatility:** {df['volatility'].iloc[-1]*100:.2f}%

### 🔍 Detected Patterns
"""
    for pattern in patterns:
        explanation += f"- ✅ {pattern}\n"
    
    if not patterns:
        explanation += "- No clear patterns\n"
    
    explanation += f"""
### 🎯 Key Levels
- **Support:** ${recent_low:.4f}
- **Resistance:** ${recent_high:.4f}

### 💰 Trade Plan
- **Entry:** ${last['c']:.4f}
- **Stop Loss:** ${sl:.4f} ({risk:.1f}% risk)
- **Take Profit 1:** ${tp1:.4f} (R:R = {reward1/risk:.1f}:1)
- **Take Profit 2:** ${tp2:.4f} (R:R = {reward2/risk:.1f}:1)

### ⚠️ Recommendation
"""
    if confidence >= 80:
        explanation += "**🔥 STRONG SIGNAL** - Consider entering with full position"
    elif confidence >= 65:
        explanation += "**📊 MODERATE SIGNAL** - Consider half position"
    else:
        explanation += "**⚠️ WEAK SIGNAL** - Wait for confirmation or pass"
    
    return explanation

# Main scanning logic
if scan_button:
    st.session_state.scanning = True
    
if st.session_state.scanning:
    coins = get_top_coins(max_coins)
    
    st.subheader(f"🔍 Scanning {len(coins)} coins...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    signals = []
    
    for i, symbol in enumerate(coins):
        try:
            status_text.text(f"📊 Analyzing {i+1}/{len(coins)}: {symbol}")
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=150)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
            
            # Skip if not enough data
            if len(df) < 50:
                continue
            
            # Train or use existing model
            if symbol not in st.session_state.models:
                model, scaler = train_model(df)
                if model:
                    st.session_state.models[symbol] = model
                    st.session_state.scalers[symbol] = scaler
            else:
                model = st.session_state.models.get(symbol)
                scaler = st.session_state.scalers.get(symbol)
            
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
            # Silently skip errors
            pass
        
        progress_bar.progress((i + 1) / len(coins))
    
    status_text.empty()
    progress_bar.empty()
    st.session_state.scanning = False
    
    # Display results
    if signals:
        st.success(f"✅ Found {len(signals)} trading signals!")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Show top signals
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
                        <p>Patterns: {', '.join(signal['patterns'][:2]) if signal['patterns'] else 'None'}</p>
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
                        <p>Patterns: {', '.join(signal['patterns'][:2]) if signal['patterns'] else 'None'}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show table
        st.subheader("📊 All Signals")
        df_show = pd.DataFrame(signals)
        df_show['price'] = df_show['price'].apply(lambda x: f"${x:.4f}")
        df_show['confidence'] = df_show['confidence'].apply(lambda x: f"{x:.1f}%")
        df_show['patterns'] = df_show['patterns'].apply(lambda x: ', '.join(x[:3]) if x else 'None')
        st.dataframe(
            df_show[['symbol', 'signal', 'confidence', 'price', 'rsi', 'patterns']], 
            use_container_width=True, 
            hide_index=True
        )
        
        # Detailed explanation for first signal
        if signals:
            st.subheader("📝 Sample Trade Analysis")
            first = signals[0]
            
            # Get full data for explanation
            symbol_full = first['symbol'] + "/USDT:USDT"
            try:
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
            except:
                st.info("Detailed chart not available")
        
    else:
        st.warning("No signals found. Try:")
        st.markdown("""
        - Lowering confidence threshold
        - Changing timeframe
        - Scanning more coins
        """)

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Models Trained", len(st.session_state.models))
with col2:
    st.metric("Timeframe", timeframe)
with col3:
    st.metric("AI Mode", ai_mode.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", ""))
with col4:
    st.metric("Min Confidence", f"{min_confidence}%")

st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🤖 Free AI Scanner | No API Keys Required | Python 3.13 Compatible</p>
    <p>⚠️ Warning: Never trade more than you can afford to lose. Past performance doesn't guarantee future results.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh (optional)
if st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()
