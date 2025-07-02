from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import hashlib
import requests
import time
from typing import Optional
from polygon import RESTClient
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

# Polygon.io Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
polygon_client = RESTClient(POLYGON_API_KEY)

# Setup logging for debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spungus-backend")

# Get the sentiment service URL from environment variables
SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL")

def get_sentiment_from_service(symbol: str) -> dict:
    """Get a rich sentiment analysis object from the dedicated sentiment service."""
    if not SENTIMENT_SERVICE_URL:
        raise Exception("SENTIMENT_SERVICE_URL is not configured.")

    url = f"{SENTIMENT_SERVICE_URL}/analyze"
    try:
        # The sentiment analysis can take time, so use a long timeout.
        response = requests.post(url, json={"symbol": symbol}, timeout=180)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Timeout when calling sentiment service for {symbol}")
        raise Exception(f"Sentiment analysis for {symbol} timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to sentiment analysis service: {e}")
        raise Exception(f"Sentiment analysis service is unavailable for {symbol}.")

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

def get_sentiment_score(symbol: str):
    """
    Fetches the sentiment analysis from the dedicated microservice.
    This function acts as a wrapper for the new service call.
    """
    try:
        # This now calls our new, powerful sentiment analysis service
        sentiment_data = get_sentiment_from_service(symbol)
        return sentiment_data
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol} from service: {e}")
        # Propagate the exception to be handled by the API endpoint
        raise

def get_stock_data_with_retry(symbol: str, max_retries: int = 3, delay: float = 2.0):
    """
    Get stock data from Polygon.io, with retry logic.
    Fetches daily bars for historical data and derives the current and previous day's prices.
    """
    for attempt in range(max_retries):
        try:
            # Get the last 200 days of daily bars for EMA calculations
            today = datetime.now().date()
            start_date = today - timedelta(days=200) # Fetch more data for accurate EMAs
            
            aggs = polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=today.strftime("%Y-%m-%d"),
                limit=5000 # Max limit
            )

            if not aggs or len(aggs) < 2:
                raise Exception(f"Insufficient historical data for {symbol} from Polygon.io (need at least 2 days).")

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # The current price is the close of the most recent trading day (T-1)
            current_price = df.iloc[-1]['close']
            
            # The previous close is from the day before that (T-2)
            prev_close_price = df.iloc[-2]['close']

            quote_data = {
                '05. price': current_price,
                '08. previous close': prev_close_price
            }
            meta_data = {
                '3. Last Refreshed': df.index[-1].isoformat()
            }
            
            return df, quote_data, meta_data
            
        except Exception as e:
            error_str = str(e).lower()
            # Polygon uses 429 for rate limiting
            is_rate_limit = "status_code=429" in error_str or "too many requests" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                print(f"Polygon.io rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            else:
                raise Exception(f"Polygon.io API error for {symbol}: {str(e)}")
    
    raise Exception(f"Failed to get data for {symbol} from Polygon.io after {max_retries} attempts")

@app.get("/api/price/{symbol}")
def get_price(symbol: str):
    try:
        # Get stock data
        data, quote_data, meta_data = get_stock_data_with_retry(symbol)
        
        # Get real-time quote data
        current_price = float(quote_data['05. price'])
        prev_close = float(quote_data['08. previous close'])
        
        # Calculate change
        price_change = current_price - prev_close
        price_change_percent = ((price_change / prev_close) * 100) if prev_close > 0 else 0
        
        # Calculate EMAs
        data['EMA5'] = calculate_ema(data['close'], 5)
        data['EMA12'] = calculate_ema(data['close'], 12)
        data['EMA34'] = calculate_ema(data['close'], 34)
        data['EMA50'] = calculate_ema(data['close'], 50)
        
        # Get latest values
        latest = data.iloc[-1]
        
        # Determine trend based on EMA clouds
        ema_5_12_above_34_50 = latest['EMA5'] > latest['EMA34'] and latest['EMA12'] > latest['EMA50']
        trend = "Bullish" if ema_5_12_above_34_50 else "Bearish"
        
        # Calculate Fibonacci levels from recent high/low
        recent_high = float(data['high'].max())
        recent_low = float(data['low'].min())
        fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
        
        # Get the actual data timestamp from Polygon.io
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

@app.get("/api/sentiment/{symbol}")
def get_sentiment(symbol: str):
    try:
        sentiment_data = get_sentiment_score(symbol)
        return {
            "symbol": symbol,
            "sentiment": sentiment_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # The error from get_sentiment_score is already descriptive
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
                "note": "Sentiment from dedicated AI service.",
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
        
        # Generate sentiment score from the dedicated service
        sentiment_data = get_sentiment_score(symbol)

        # The signal generator expects a score from -50 to +50.
        # The service provides 'overall_sentiment' on a 0-100 scale.
        overall_sentiment = sentiment_data.get('overall_sentiment', 50.0)
        sentiment_for_signal_generator = overall_sentiment - 50.0
        
        # Convert to list for calculations
        price_data = [float(x) for x in data['close'].tolist()]
        
        # Generate trading signal using Spungus methodology
        signal, signal_score, components = generate_trading_signal(price_data, sentiment_for_signal_generator)
        
        # Replace the basic sentiment component with the full, rich data from the service
        components['sentiment'] = sentiment_data

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

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Serve React app for all other routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

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