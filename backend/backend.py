from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import hashlib
import requests
import time
from typing import Optional
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Alpha Vantage Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Setup logging for debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spungus-backend")

def calculate_ema(data, period):
    """Calculate Exponential Moving Average for a pandas Series or list of prices"""
    if isinstance(data, pd.Series):
        # For pandas Series, calculate EMA properly
        ema = data.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]  # Return the last value
    elif isinstance(data, list):
        # Convert list to pandas Series for calculation
        series = pd.Series(data)
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]  # Return the last value
    else:
        # If it's already a pandas Series or other format
        try:
            ema = data.ewm(span=period, adjust=False).mean()
            return ema.iloc[-1] if hasattr(ema, 'iloc') else ema
        except:
            # Fallback to simple average if EMA calculation fails
            return sum(data[-period:]) / min(period, len(data))

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    }

def get_real_news_sentiment(symbol: str) -> dict:
    """Get real news sentiment using NewsAPI"""
    
    # Get API key from environment variable (you'll need to set this)
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        raise Exception("NEWS_API_KEY not configured. Please set your NewsAPI key.")
    
    try:
        # Search for news about the company
        company_name = get_company_name(symbol)
        query = f"{symbol} OR {company_name}"
        
        # NewsAPI endpoint
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,  # Get last 20 articles
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] != 'ok' or not data.get('articles'):
            raise Exception("No news articles found")
        
        # Analyze sentiment from article titles and descriptions
        articles = data['articles']
        sentiment_scores = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if text.strip():
                score = analyze_text_sentiment(text)
                sentiment_scores.append(score)
        
        if not sentiment_scores:
            raise Exception("No sentiment could be analyzed from articles")
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Normalize to 0-1 scale
        normalized_sentiment = (avg_sentiment + 1) / 2  # Convert from -1,1 to 0,1
        
        # Determine confidence based on number of articles
        confidence = min(0.95, 0.5 + (len(articles) * 0.02))
        
        return {
            'score': round(normalized_sentiment, 3),
            'label': 'Bullish' if normalized_sentiment > 0.6 else 'Bearish' if normalized_sentiment < 0.4 else 'Neutral',
            'confidence': round(confidence, 2),
            'articles_analyzed': len(articles),
            'source': 'NewsAPI'
        }
        
    except Exception as e:
        raise Exception(f"NewsAPI error: {e}")

def get_company_name(symbol: str) -> str:
    """Get company name from symbol"""
    company_names = {
        'AAPL': 'Apple Inc',
        'TSLA': 'Tesla Inc',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc',
        'AMZN': 'Amazon.com Inc',
        'SNAP': 'Snap Inc',
        'NVDA': 'NVIDIA Corporation',
        'META': 'Meta Platforms Inc',
        'NFLX': 'Netflix Inc',
        'AMD': 'Advanced Micro Devices Inc',
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF'
    }
    return company_names.get(symbol, symbol)

def analyze_text_sentiment(text: str) -> float:
    """Simple sentiment analysis using keyword scoring"""
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Positive keywords (financial context)
    positive_words = [
        'bullish', 'surge', 'rally', 'gain', 'rise', 'up', 'positive', 'strong',
        'beat', 'exceed', 'growth', 'profit', 'earnings', 'revenue', 'upgrade',
        'buy', 'outperform', 'strong buy', 'positive outlook', 'growth',
        'innovation', 'breakthrough', 'success', 'win', 'leading', 'top'
    ]
    
    # Negative keywords (financial context)
    negative_words = [
        'bearish', 'drop', 'fall', 'decline', 'down', 'negative', 'weak',
        'miss', 'disappoint', 'loss', 'decline', 'downgrade', 'sell',
        'underperform', 'strong sell', 'negative outlook', 'concern',
        'risk', 'warning', 'trouble', 'problem', 'issue', 'challenge'
    ]
    
    # Count occurrences
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate sentiment score (-1 to 1)
    total_words = positive_count + negative_count
    if total_words == 0:
        return 0  # Neutral if no sentiment words found
    
    sentiment_score = (positive_count - negative_count) / total_words
    
    # Apply some weighting based on text length and keyword density
    keyword_density = total_words / len(text.split())
    sentiment_score *= min(1.0, keyword_density * 10)  # Scale by density
    
    return max(-1.0, min(1.0, sentiment_score))  # Clamp to -1, 1

def get_sentiment_score(symbol: str):
    """Get sentiment score using real news sentiment or throw error"""
    
    # Try to get real news sentiment first
    try:
        sentiment_data = get_real_news_sentiment(symbol)
        if sentiment_data['source'] != 'Simulated':
            return sentiment_data
    except Exception as e:
        print(f"Real news sentiment failed: {e}")
    
    # If no real sentiment available, throw error instead of using simulated data
    raise Exception(f"No sentiment data available for {symbol}. Please check your API configuration.")

def get_stock_data_with_retry(symbol: str, max_retries: int = 3, delay: float = 1.0):
    """Get stock data with retry logic using Alpha Vantage"""
    for attempt in range(max_retries):
        try:
            # Get daily data from Alpha Vantage
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
            
            if data.empty:
                raise Exception(f"No historical data available for {symbol}")
            
            # Get current quote
            quote_data, quote_meta = ts.get_quote_endpoint(symbol=symbol)
            
            # Validate quote data
            if not quote_data or '05. price' not in quote_data or '08. previous close' not in quote_data:
                raise Exception(f"Incomplete quote data for {symbol}")
            
            return data, quote_data, meta_data
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e) or "API call frequency" in str(e):
                if attempt < max_retries - 1:
                    print(f"Rate limited, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Alpha Vantage rate limit exceeded. Please upgrade to premium plan or wait for rate limit reset. Error: {str(e)}")
            else:
                raise Exception(f"Alpha Vantage error: {str(e)}")
    
    raise Exception(f"Failed to get data for {symbol} after {max_retries} attempts")

@app.get("/price")
def get_price(symbol: str = Query(...)):
    try:
        # Get stock data
        data, quote_data, meta_data = get_stock_data_with_retry(symbol)
        
        # Get real-time quote data (no fallbacks)
        current_price = float(quote_data['05. price'])
        prev_close = float(quote_data['08. previous close'])
        
        # Calculate change
        price_change = current_price - prev_close
        price_change_percent = ((price_change / prev_close) * 100) if prev_close > 0 else 0
        
        # Calculate EMAs
        data['EMA5'] = calculate_ema(data['4. close'], 5)
        data['EMA12'] = calculate_ema(data['4. close'], 12)
        data['EMA34'] = calculate_ema(data['4. close'], 34)
        data['EMA50'] = calculate_ema(data['4. close'], 50)
        
        # Get latest values
        latest = data.iloc[-1]
        
        # Determine trend based on EMA clouds
        ema_5_12_above_34_50 = latest['EMA5'] > latest['EMA34'] and latest['EMA12'] > latest['EMA50']
        trend = "Bullish" if ema_5_12_above_34_50 else "Bearish"
        
        # Calculate Fibonacci levels from recent high/low
        recent_high = float(data['2. high'].max())
        recent_low = float(data['3. low'].min())
        fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
        
        # Get the actual data timestamp from Alpha Vantage
        data_timestamp = meta_data.get('3. Last Refreshed', datetime.now().isoformat())
        if isinstance(data_timestamp, str):
            try:
                data_timestamp = datetime.fromisoformat(data_timestamp)
            except Exception:
                data_timestamp = datetime.now()
        else:
            data_timestamp = datetime.now()
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round(price_change, 2),
            "change_percent": round(price_change_percent, 2),
            "ema": {
                "ema5": round(latest['EMA5'], 2),
                "ema12": round(latest['EMA12'], 2),
                "ema34": round(latest['EMA34'], 2),
                "ema50": round(latest['EMA50'], 2)
            },
            "trend": trend,
            "fibonacci": fib_levels,
            "timestamp": data_timestamp.isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get price data for {symbol}: {str(e)}"}

@app.get("/sentiment")
def get_sentiment(symbol: str = Query(...)):
    try:
        sentiment_data = get_sentiment_score(symbol)
        return {
            "symbol": symbol,
            "sentiment": sentiment_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

def generate_trading_signal(price_data, sentiment_score):
    """Generate trading signal based on Spungus trading system methodology"""
    
    # Ensure we have enough data for EMA calculations (need at least 50 periods)
    if len(price_data) < 50:
        raise Exception(f"Insufficient historical data. Need at least 50 periods, got {len(price_data)}")
    
    # Convert to list if it's a pandas Series
    if isinstance(price_data, pd.Series):
        price_list = price_data.tolist()
    else:
        price_list = price_data
    
    # Calculate EMAs for Ripster clouds (5, 12, 34, 50)
    ema_5 = calculate_ema(price_list, 5)
    ema_12 = calculate_ema(price_list, 12)
    ema_34 = calculate_ema(price_list, 34)
    ema_50 = calculate_ema(price_list, 50)
    
    # Current price
    current_price = price_list[-1]
    
    # SPUNGUS METHODOLOGY: 
    # "When 5/12 is above 34/50, trend is bullish. When 5/12 is below 34/50, trend is bearish"
    
    # Calculate 5/12 cloud center (average of 5 and 12 EMA)
    cloud_5_12_center = (ema_5 + ema_12) / 2
    
    # Calculate 34/50 cloud center (average of 34 and 50 EMA)
    cloud_34_50_center = (ema_34 + ema_50) / 2
    
    # Calculate cloud thickness (Spungus uses this for trend strength)
    cloud_5_12_thickness = abs(ema_5 - ema_12)
    cloud_34_50_thickness = abs(ema_34 - ema_50)
    
    # Calculate cloud separation (distance between cloud centers)
    cloud_separation = abs(cloud_5_12_center - cloud_34_50_center)
    
    # SPUNGUS TREND DETERMINATION
    if cloud_5_12_center > cloud_34_50_center:
        # 5/12 cloud is above 34/50 cloud = BULLISH
        base_signal = "BULLISH"
        signal_score = 25  # Base bullish score
        
        # Spungus trend strength analysis
        if cloud_separation > (current_price * 0.02):  # Strong separation
            signal_score += 20  # Strong trend
        elif cloud_separation > (current_price * 0.01):  # Moderate separation
            signal_score += 10  # Moderate trend
        else:
            signal_score += 5   # Weak trend
            
    elif cloud_5_12_center < cloud_34_50_center:
        # 5/12 cloud is below 34/50 cloud = BEARISH
        base_signal = "BEARISH"
        signal_score = -25  # Base bearish score
        
        # Spungus trend strength analysis
        if cloud_separation > (current_price * 0.02):  # Strong separation
            signal_score -= 20  # Strong trend
        elif cloud_separation > (current_price * 0.01):  # Moderate separation
            signal_score -= 10  # Moderate trend
        else:
            signal_score -= 5   # Weak trend
    else:
        # Clouds are converging = NEUTRAL/REVERSING
        base_signal = "NEUTRAL"
        signal_score = 0
    
    # SPUNGUS PRICE ACTION ANALYSIS
    # Price position relative to clouds (for strength assessment)
    price_above_5_12 = current_price > ema_5 and current_price > ema_12
    price_below_5_12 = current_price < ema_5 and current_price < ema_12
    price_above_34_50 = current_price > ema_34 and current_price > ema_50
    price_below_34_50 = current_price < ema_34 and current_price < ema_50
    
    # Price within 5/12 cloud (Spungus considers this important)
    price_in_5_12_cloud = min(ema_5, ema_12) <= current_price <= max(ema_5, ema_12)
    price_in_34_50_cloud = min(ema_34, ema_50) <= current_price <= max(ema_34, ema_50)
    
    # Adjust signal strength based on price position (Spungus methodology)
    if base_signal == "BULLISH":
        if price_above_5_12:
            signal_score += 15  # Strong bullish - price riding the trend
        elif price_in_5_12_cloud:
            signal_score += 10  # Price within 5/12 cloud - good entry zone
        elif price_above_34_50:
            signal_score += 5   # Moderate bullish - price above support
    elif base_signal == "BEARISH":
        if price_below_5_12:
            signal_score -= 15  # Strong bearish - price below short-term EMAs
        elif price_in_5_12_cloud:
            signal_score -= 10  # Price within 5/12 cloud - good entry zone
        elif price_below_34_50:
            signal_score -= 5   # Moderate bearish - price below support
    
    # SPUNGUS MOMENTUM ANALYSIS
    # Calculate momentum using recent price changes
    if len(price_list) >= 5:
        recent_momentum = (price_list[-1] - price_list[-5]) / price_list[-5] * 100
        if base_signal == "BULLISH" and recent_momentum > 1:
            signal_score += 10  # Momentum confirms bullish trend
        elif base_signal == "BEARISH" and recent_momentum < -1:
            signal_score -= 10  # Momentum confirms bearish trend
    else:
        recent_momentum = 0
    
    # Fibonacci levels for entry/exit zones (Spungus uses these for targets)
    # Use last 20 periods for Fibonacci calculation
    recent_data = price_list[-20:] if len(price_list) >= 20 else price_list
    recent_high = max(recent_data)
    recent_low = min(recent_data)
    range_size = recent_high - recent_low
    
    fib_236 = recent_high - (range_size * 0.236)
    fib_382 = recent_high - (range_size * 0.382)
    fib_500 = recent_high - (range_size * 0.500)
    fib_618 = recent_high - (range_size * 0.618)
    fib_786 = recent_high - (range_size * 0.786)
    
    # SPUNGUS FIBONACCI ANALYSIS
    # Determine nearest Fibonacci levels for entry zones
    nearest_support = None
    nearest_resistance = None
    
    if current_price > fib_618:
        nearest_resistance = recent_high
        nearest_support = fib_618
    elif current_price > fib_500:
        nearest_resistance = fib_618
        nearest_support = fib_500
    elif current_price > fib_382:
        nearest_resistance = fib_500
        nearest_support = fib_382
    elif current_price > fib_236:
        nearest_resistance = fib_382
        nearest_support = fib_236
    else:
        nearest_resistance = fib_236
        nearest_support = recent_low
    
    # SPUNGUS SIGNAL STRENGTH CLASSIFICATION
    if abs(signal_score) > 50:
        signal = f"VERY STRONG {base_signal}"
    elif abs(signal_score) > 35:
        signal = f"STRONG {base_signal}"
    elif abs(signal_score) > 15:
        signal = base_signal
    else:
        signal = "NEUTRAL"
    
    # Component breakdown (strictly following Spungus methodology)
    components = {
        "ema_cloud": {
            "score": float(signal_score),
            "details": {
                "ema_5": round(ema_5, 2),
                "ema_12": round(ema_12, 2),
                "ema_34": round(ema_34, 2),
                "ema_50": round(ema_50, 2),
                "cloud_5_12_center": round(cloud_5_12_center, 2),
                "cloud_34_50_center": round(cloud_34_50_center, 2),
                "cloud_5_12_thickness": round(cloud_5_12_thickness, 2),
                "cloud_34_50_thickness": round(cloud_34_50_thickness, 2),
                "cloud_separation": round(cloud_separation, 2),
                "cloud_5_12_above_34_50": bool(cloud_5_12_center > cloud_34_50_center),
                "trend_strength": "Strong" if cloud_separation > (current_price * 0.02) else "Moderate" if cloud_separation > (current_price * 0.01) else "Weak",
                "price_above_5_12": bool(price_above_5_12),
                "price_in_5_12_cloud": bool(price_in_5_12_cloud),
                "price_above_34_50": bool(price_above_34_50),
                "price_in_34_50_cloud": bool(price_in_34_50_cloud)
            }
        },
        "fibonacci": {
            "score": calculate_fibonacci_score(current_price, fib_236, fib_382, fib_500, fib_618, fib_786),
            "details": {
                "fib_236": round(fib_236, 2),
                "fib_382": round(fib_382, 2),
                "fib_500": round(fib_500, 2),
                "fib_618": round(fib_618, 2),
                "fib_786": round(fib_786, 2),
                "nearest_support": round(nearest_support, 2),
                "nearest_resistance": round(nearest_resistance, 2),
                "price_position": "Above 0.618" if current_price > fib_618 else "Above 0.500" if current_price > fib_500 else "Above 0.382" if current_price > fib_382 else "Above 0.236" if current_price > fib_236 else "Below 0.236",
                "range_size": round(range_size, 2),
                "recent_high": round(recent_high, 2),
                "recent_low": round(recent_low, 2)
            }
        },
        "price_action": {
            "score": calculate_price_action_score(current_price, ema_5, ema_12, ema_34, ema_50),
            "details": {
                "current_price": round(current_price, 2),
                "position_vs_clouds": "Above 5/12" if price_above_5_12 else "In 5/12 Cloud" if price_in_5_12_cloud else "Below 5/12" if price_below_5_12 else "Between clouds",
                "momentum": round(recent_momentum, 2) if len(price_list) >= 5 else "N/A",
                "strength": "Strong" if abs(current_price - ema_5) / ema_5 > 0.02 else "Moderate" if abs(current_price - ema_5) / ema_5 > 0.01 else "Weak"
            }
        },
        "sentiment": {
            "score": sentiment_score,
            "details": {
                "note": "Local LLM sentiment analysis available",
                "current_sentiment": "Bullish" if sentiment_score > 10 else "Bearish" if sentiment_score < -10 else "Neutral"
            }
        }
    }
    
    return signal, signal_score, components

def generate_option_recommendation(signal, score, components):
    """Generate specific option trading recommendations based on Spungus trading system"""
    recommendations = []
    
    # Get component details for analysis
    ema_cloud = components.get("ema_cloud", {})
    fibonacci = components.get("fibonacci", {})
    price_action = components.get("price_action", {})
    
    # SPUNGUS POSITION SIZING AND TIMING
    # Determine time frame based on signal strength and market conditions
    if abs(score) > 50:  # Very strong signal
        days_to_expiry = 5   # Weekly options for very strong moves
        position_size = "5-7% of portfolio"
        conviction = "VERY HIGH CONVICTION"
    elif abs(score) > 35:  # Strong signal
        days_to_expiry = 7   # Weekly options for strong moves
        position_size = "3-5% of portfolio"
        conviction = "HIGH CONVICTION"
    elif abs(score) > 15:  # Moderate signal
        days_to_expiry = 14  # 2-week options
        position_size = "2-3% of portfolio"
        conviction = "MODERATE CONVICTION"
    else:  # Weak signal
        days_to_expiry = 30  # Monthly options for safer plays
        position_size = "1-2% of portfolio"
        conviction = "LOW CONVICTION"
    
    # SPUNGUS ENTRY TIMING ANALYSIS
    ema_details = ema_cloud.get("details", {})
    price_in_5_12_cloud = ema_details.get("price_in_5_12_cloud", False)
    trend_strength = ema_details.get("trend_strength", "Weak")
    cloud_5_12_above_34_50 = ema_details.get("cloud_5_12_above_34_50", False)
    
    # SPUNGUS SYSTEM ANALYSIS
    if "VERY STRONG BULLISH" in signal or "STRONG BULLISH" in signal:
        recommendations.extend([
            f"🎯 {conviction} PLAY: Buy {days_to_expiry}-day calls",
            f"📊 Target: Fibonacci 0.618-0.786 resistance levels",
            f"🛡️ Stop Loss: Below EMA 5/12 cloud support",
            f"⏰ Expiration: {days_to_expiry} days out",
            f"💰 Position Size: {position_size}"
        ])
        
        # Spungus-specific entry strategy
        if price_in_5_12_cloud:
            recommendations.append("🎯 Entry: Current price in 5/12 EMA cloud - EXCELLENT entry zone")
        elif ema_details.get("price_above_5_12", False):
            recommendations.append("🎯 Entry: Price above 5/12 EMA - Strong trend continuation")
        else:
            recommendations.append("🎯 Entry: Wait for pullback to 5/12 EMA cloud")
            
        # Trend strength analysis
        if trend_strength == "Strong":
            recommendations.append("🚀 Trend Strength: Strong separation between EMA clouds")
        elif trend_strength == "Moderate":
            recommendations.append("📈 Trend Strength: Moderate separation - good trend")
        else:
            recommendations.append("⚠️ Trend Strength: Weak separation - consider waiting")
            
        # Cloud alignment confirmation
        if cloud_5_12_above_34_50:
            recommendations.append("✅ EMA Alignment: 5/12 cloud above 34/50 cloud - BULLISH")
        else:
            recommendations.append("❌ EMA Alignment: Clouds not properly aligned - CAUTION")
            
    elif "BULLISH" in signal:
        recommendations.extend([
            f"📈 {conviction} PLAY: Buy {days_to_expiry}-day calls",
            f"📊 Target: Fibonacci 0.500-0.618 levels",
            f"🛡️ Stop Loss: Below EMA 34/50 support",
            f"⏰ Expiration: {days_to_expiry} days out",
            f"💰 Position Size: {position_size}"
        ])
        
        # Moderate bullish analysis
        if price_in_5_12_cloud:
            recommendations.append("🎯 Entry: Price in 5/12 EMA cloud - good entry zone")
        else:
            recommendations.append("🎯 Entry: Wait for pullback to 5/12 EMA support")
            
    elif "VERY STRONG BEARISH" in signal or "STRONG BEARISH" in signal:
        recommendations.extend([
            f"🎯 {conviction} PLAY: Buy {days_to_expiry}-day puts",
            f"📊 Target: Fibonacci 0.236-0.382 support levels",
            f"🛡️ Stop Loss: Above EMA 5/12 cloud resistance",
            f"⏰ Expiration: {days_to_expiry} days out",
            f"💰 Position Size: {position_size}"
        ])
        
        # Spungus-specific entry strategy for bearish
        if price_in_5_12_cloud:
            recommendations.append("🎯 Entry: Current price in 5/12 EMA cloud - EXCELLENT entry zone")
        elif ema_details.get("price_below_5_12", False):
            recommendations.append("🎯 Entry: Price below 5/12 EMA - Strong downtrend continuation")
        else:
            recommendations.append("🎯 Entry: Wait for bounce to 5/12 EMA cloud")
            
        # Trend strength analysis
        if trend_strength == "Strong":
            recommendations.append("📉 Trend Strength: Strong separation between EMA clouds")
        elif trend_strength == "Moderate":
            recommendations.append("📉 Trend Strength: Moderate separation - good downtrend")
        else:
            recommendations.append("⚠️ Trend Strength: Weak separation - consider waiting")
            
        # Cloud alignment confirmation
        if not cloud_5_12_above_34_50:
            recommendations.append("✅ EMA Alignment: 5/12 cloud below 34/50 cloud - BEARISH")
        else:
            recommendations.append("❌ EMA Alignment: Clouds not properly aligned - CAUTION")
            
    elif "BEARISH" in signal:
        recommendations.extend([
            f"📉 {conviction} PLAY: Buy {days_to_expiry}-day puts",
            f"📊 Target: Fibonacci 0.382-0.500 levels",
            f"🛡️ Stop Loss: Above EMA 34/50 resistance",
            f"⏰ Expiration: {days_to_expiry} days out",
            f"💰 Position Size: {position_size}"
        ])
        
        # Moderate bearish analysis
        if price_in_5_12_cloud:
            recommendations.append("🎯 Entry: Price in 5/12 EMA cloud - good entry zone")
        else:
            recommendations.append("🎯 Entry: Wait for bounce to 5/12 EMA resistance")
            
    else:  # NEUTRAL
        recommendations.extend([
            "⏸️ NEUTRAL SIGNAL: No clear directional bias",
            "📊 Wait for EMA clouds to align",
            "🎯 Consider straddle/strangle if expecting breakout",
            "💰 Position Size: 1% of portfolio (if any)",
            "⚠️ Risk: High - unclear direction"
        ])
    
    # SPUNGUS RISK MANAGEMENT
    recommendations.append("")
    recommendations.append("🛡️ SPUNGUS RISK MANAGEMENT:")
    
    # Fibonacci-based risk management
    fib_details = fibonacci.get("details", {})
    price_details = price_action.get("details", {})
    current_price = price_details.get("current_price", 0)
    nearest_support = fib_details.get("nearest_support", 0)
    nearest_resistance = fib_details.get("nearest_resistance", 0)
    
    if "BULLISH" in signal and current_price > 0:
        risk_distance = current_price - nearest_support
        risk_percentage = (risk_distance / current_price) * 100
        recommendations.append(f"📏 Risk Distance: ${risk_distance:.2f} ({risk_percentage:.1f}%)")
        recommendations.append(f"🎯 Risk/Reward: 1:2 minimum (target 2x risk distance)")
    elif "BEARISH" in signal and current_price > 0:
        risk_distance = nearest_resistance - current_price
        risk_percentage = (risk_distance / current_price) * 100
        recommendations.append(f"📏 Risk Distance: ${risk_distance:.2f} ({risk_percentage:.1f}%)")
        recommendations.append(f"🎯 Risk/Reward: 1:2 minimum (target 2x risk distance)")
    
    # Spungus-specific risk rules
    recommendations.append("📋 Spungus Rules:")
    recommendations.append("• Never risk more than 2% per trade")
    recommendations.append("• Use EMA clouds for entry/exit timing")
    recommendations.append("• Fibonacci levels for profit targets")
    recommendations.append("• Exit if EMA clouds cross (trend change)")
    
    return recommendations

@app.get("/api/signal/{symbol}")
async def get_signal(symbol: str):
    try:
        data, quote_data, meta_data = get_stock_data_with_retry(symbol)
        
        # Get current price from quote data (no fallbacks)
        current_price = float(quote_data['05. price'])
        market_time = meta_data.get('3. Last Refreshed', 0)
        
        # Generate sentiment score
        sentiment_data = get_sentiment_score(symbol)
        # Convert 0-100 scale to -50 to +50 scale for signal generation
        sentiment_score = (sentiment_data['score'] - 50)  # Convert 0-100 to -50 to +50
        
        # Convert to list for calculations
        price_data = [float(x) for x in data['4. close'].tolist()]
        
        # Generate trading signal using Spungus methodology
        signal, signal_score, components = generate_trading_signal(price_data, sentiment_score)
        
        # Generate option recommendations
        recommendations = generate_option_recommendation(signal, signal_score, components)
        
        return {
            "symbol": symbol,
            "signal": signal,
            "score": round(signal_score, 2),
            "current_price": round(current_price, 2),
            "market_time": market_time,
            "components": components,
            "recommendations": recommendations,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get signal for {symbol}: {str(e)}"}

# Serve React app for all other routes
@app.get("/{full_path:path}")
def serve_react_app(request: Request, full_path: str):
    """Serve React app for all non-API routes"""
    # Skip API routes
    if full_path.startswith(("price", "sentiment", "signal")):
        return {"detail": "Not Found"}
    
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    
    # Handle root path - serve index.html
    if not full_path or full_path == "":
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            return {"message": "Frontend not found. Please build the React app first."}
    
    # Try to serve static files
    file_path = os.path.join(frontend_dir, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fall back to index.html for React routing
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {"message": "Frontend not found. Please build the React app first."}

def calculate_fibonacci_score(current_price, fib_236, fib_382, fib_500, fib_618, fib_786):
    """Calculate Fibonacci score based on price position relative to levels"""
    # Determine which Fibonacci zone the price is in
    if current_price > fib_786:
        # Price above 0.786 - very bullish
        return 15
    elif current_price > fib_618:
        # Price above 0.618 - bullish
        return 10
    elif current_price > fib_500:
        # Price above 0.500 - moderately bullish
        return 5
    elif current_price > fib_382:
        # Price above 0.382 - neutral to slightly bullish
        return 0
    elif current_price > fib_236:
        # Price above 0.236 - neutral to slightly bearish
        return -5
    else:
        # Price below 0.236 - bearish
        return -10

def calculate_price_action_score(current_price, ema_5, ema_12, ema_34, ema_50):
    """Calculate price action score based on position relative to EMAs"""
    score = 0
    
    # Price relative to 5 EMA
    if current_price > ema_5:
        score += 5
    else:
        score -= 5
    
    # Price relative to 12 EMA
    if current_price > ema_12:
        score += 3
    else:
        score -= 3
    
    # Price relative to 34 EMA
    if current_price > ema_34:
        score += 2
    else:
        score -= 2
    
    # Price relative to 50 EMA
    if current_price > ema_50:
        score += 1
    else:
        score -= 1
    
    return score 