from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import aiohttp
import os
from datetime import datetime
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
import re
from gnews import GNews
from functools import partial

load_dotenv()

app = FastAPI(title="Sentiment Analysis Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local LLM Configuration
LOCAL_LLM_URL = os.getenv('LOCAL_LLM_URL', 'http://localhost:1234/v1/chat/completions')
LOCAL_LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'default')

class SentimentRequest(BaseModel):
    symbol: str
    max_articles: int = 10

class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: float
    sentiment_label: str
    confidence: float
    summary_reasoning: Optional[str] = None
    key_bullish_factors: Optional[List[str]] = None
    key_bearish_factors: Optional[List[str]] = None
    articles_analyzed: int
    articles: List[Dict]
    timestamp: str

def search_gnews_blocking(query: str, max_results: int):
    """Blocking function to search GNews, designed to be run in an executor."""
    print(f"Searching Google News for: {query}")
    try:
        gnews_client = GNews(language='en', country='US', max_results=max_results)
        return gnews_client.get_news(query)
    except Exception as e:
        print(f"gnews query failed for '{query}': {e}")
        return None

async def scrape_google_news(symbol: str, company_name: str, max_articles: int) -> List[Dict]:
    """Scrape news articles from Google News using financial-specific search terms."""
    print(f"Scraping Google News for {symbol} ({company_name}) with financial context")
    
    search_queries = [
        f'"{company_name}" stock',
        f'"{symbol}" stock',
        f'"{company_name}" finance',
        f'"{symbol}" earnings'
    ]
    
    loop = asyncio.get_running_loop()
    tasks = []
    # Distribute max_articles across our queries, with a minimum of 5
    max_per_query = max(5, max_articles // len(search_queries))
    
    for query in search_queries:
        # Use functools.partial to pass arguments to the executor function
        func = partial(search_gnews_blocking, query, max_per_query)
        tasks.append(loop.run_in_executor(None, func))
        
    search_results = await asyncio.gather(*tasks)
    
    combined_news = []
    for result in search_results:
        if result:
            combined_news.extend(result)

    articles = []
    seen_urls = set()

    for item in combined_news:
        if item['url'] not in seen_urls:
            articles.append({
                'title': item.get('title'),
                'url': item.get('url'),
                'summary': item.get('description'),
                'published': item.get('published date'),
                'source': item.get('publisher', {}).get('title')
            })
            seen_urls.add(item['url'])
    
    print(f"Found {len(articles)} unique articles from Google News.")
    return articles[:max_articles]

async def scrape_news_articles(symbol: str, max_articles: int = 10) -> List[Dict]:
    """Scrape news articles from Google News."""
    
    company_names = {
        'AAPL': 'Apple', 'TSLA': 'Tesla', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
        'AMZN': 'Amazon', 'NVDA': 'NVIDIA', 'META': 'Meta', 'NFLX': 'Netflix',
        'AMD': 'AMD', 'SPY': 'SPDR S&P 500', 'QQQ': 'Invesco QQQ', 'IWM': 'iShares Russell 2000',
        'SNAP': 'Snap Inc.'
    }
    company_name = company_names.get(symbol, symbol)
    
    articles = await scrape_google_news(symbol, company_name, max_articles)
    
    # If no articles found, throw an error
    if not articles:
        raise Exception(f"Failed to scrape any articles for {symbol} from Google News. Service may be blocked or unavailable.")
    
    # Remove duplicates and limit to max_articles
    unique_articles = []
    seen_urls = set()
    
    for article in articles:
        if article['url'] not in seen_urls and len(unique_articles) < max_articles:
            unique_articles.append(article)
            seen_urls.add(article['url'])
    
    if not unique_articles:
        raise Exception(f"No unique articles found for {symbol} after deduplication.")
    
    return unique_articles
    
def clean_text(text: str) -> str:
    """Clean HTML and special characters from text"""
    if text is None:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    return text.strip()

async def call_local_llm(prompt: str) -> str:
    """Call local LLM API"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a financial analyst specializing in sentiment analysis. Provide accurate, unbiased sentiment scores based on financial news content."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            async with session.post(LOCAL_LLM_URL, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    print(f"Local LLM API error: {response.status}")
                    return None
                    
    except Exception as e:
        print(f"Error calling local LLM: {e}")
        return None

async def analyze_article_sentiment(article: Dict) -> Dict:
    """Analyze sentiment of a single article using local LLM"""
    
    try:
        # Prepare text for analysis
        text = f"Title: {article['title']}\n\nSummary: {article['summary']}"
        
        # Create prompt for numerical sentiment analysis
        prompt = f"""
        Analyze the sentiment of this financial news article about {article.get('symbol', 'a company')}.
        
        Article:
        {text}
        
        Provide a sentiment analysis with a numerical score from 0 to 100, where:
        - 0-20: Very Bearish (extremely negative)
        - 21-40: Bearish (negative)
        - 41-60: Neutral (balanced)
        - 61-80: Bullish (positive)
        - 81-100: Very Bullish (extremely positive)
        
        Respond in this exact JSON format. Your entire response MUST be only the JSON object, without any markdown, comments, or other text.
        {{
            "sentiment_score": <number between 0 and 100>,
            "confidence": <number between 0 and 100>,
            "reasoning": "<brief explanation of the sentiment score>",
            "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
        }}
        
        Focus on financial impact and market sentiment. Consider:
        - Earnings, revenue, growth prospects
        - Market position and competition
        - Regulatory or legal issues
        - Management changes or strategy shifts
        - Market reaction and analyst opinions
        - Overall tone and language used
        """
        
        # Call local LLM
        response_text = await call_local_llm(prompt)
        
        if response_text:
            # Extract JSON from response
            try:
                # Find JSON in the response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                sentiment_data = json.loads(json_str)
                
                # Validate the sentiment score is within 0-100 range
                score = sentiment_data.get('sentiment_score', 50)
                score = max(0, min(100, score))  # Clamp to 0-100
                
                return {
                    'sentiment_score': score,
                    'confidence': sentiment_data.get('confidence', 50),
                    'reasoning': sentiment_data.get('reasoning', ''),
                    'key_factors': sentiment_data.get('key_factors', [])
                }
                
            except json.JSONDecodeError:
                raise Exception(f"LLM returned invalid JSON: {response_text}")
        else:
            raise Exception("LLM is not available or not responding")
            
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        raise Exception(f"Sentiment analysis failed: {str(e)}")

async def analyze_overall_sentiment(symbol: str, articles: List[Dict]) -> Dict:
    """Analyze the overall sentiment from a list of analyzed articles using a second LLM call."""
    
    prompt_data = []
    for article in articles:
        prompt_data.append({
            "title": article['title'],
            "source": article['source'],
            "sentiment": article['sentiment']
        })

    prompt = f"""
    You are a lead financial analyst providing a final investment summary for {symbol}.
    You have received the following sentiment analyses for a collection of recent news articles.
    Your task is to synthesize this information into a single, conclusive overview.

    Here is the data from your junior analysts:
    ---
    {json.dumps(prompt_data, indent=2)}
    ---

    Review all the articles, paying attention to the sentiment scores, confidence levels, and reasoning provided for each.
    Provide a final, consolidated sentiment analysis. Your entire response MUST be only the JSON object, without any markdown, comments, or other text.

    {{
        "overall_sentiment_score": <a single number from 0 to 100>,
        "sentiment_label": "<'Very Bullish', 'Bullish', 'Neutral', 'Bearish', or 'Very Bearish'>",
        "confidence": <a single number from 0 to 100, representing your confidence in this *overall* analysis>,
        "summary_reasoning": "<A concise paragraph summarizing the key drivers for your final sentiment. Mention the dominant themes, conflicting reports, and why you are leaning in a particular direction.>",
        "key_bullish_factors": ["<factor1>", "<factor2>"],
        "key_bearish_factors": ["<factor1>", "<factor2>"]
    }}

    Base your final judgment on the weight of the evidence. For example, a strong earnings report might outweigh general market jitters.
    """
    
    response_text = await call_local_llm(prompt)

    if response_text:
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            final_sentiment_data = json.loads(json_str)
            
            if all(k in final_sentiment_data for k in ["overall_sentiment_score", "sentiment_label", "confidence", "summary_reasoning"]):
                return final_sentiment_data
            else:
                raise Exception("LLM response for final analysis is missing required keys.")

        except json.JSONDecodeError:
            raise Exception(f"Final LLM analysis returned invalid JSON: {response_text}")
    else:
        raise Exception("Final LLM analysis call failed or returned no response.")

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis Service", 
        "status": "running",
        "local_llm_url": LOCAL_LLM_URL,
        "local_llm_model": LOCAL_LLM_MODEL
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment for a stock symbol"""
    
    try:
        # Step 1: Scrape news articles
        print(f"Scraping news for {request.symbol}...")
        articles = await scrape_news_articles(request.symbol, request.max_articles)
        
        if not articles:
            return SentimentResponse(
                symbol=request.symbol,
                overall_sentiment=50.0,
                sentiment_label="Neutral",
                confidence=0.0,
                articles_analyzed=0,
                articles=[],
                timestamp=datetime.now().isoformat()
            )
        
        # Step 2: Analyze sentiment for each article concurrently
        print(f"Analyzing sentiment for {len(articles)} articles...")
        
        analysis_tasks = []
        for article in articles:
            article['symbol'] = request.symbol
            analysis_tasks.append(analyze_article_sentiment(article))
        
        individual_sentiments = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        analyzed_articles = []
        for i, result in enumerate(individual_sentiments):
            if isinstance(result, dict):
                articles[i]['sentiment'] = result
                analyzed_articles.append(articles[i])
            else:
                print(f"Article analysis failed for '{articles[i]['title']}': {result}")

        if not analyzed_articles:
            raise Exception("Sentiment analysis failed for all articles.")

        # Step 3: Perform final overall sentiment analysis with a second LLM call
        print("Performing final analysis on all articles...")
        final_analysis = await analyze_overall_sentiment(request.symbol, analyzed_articles)
        
        # Step 4: Construct the final response
        return SentimentResponse(
            symbol=request.symbol,
            overall_sentiment=round(final_analysis['overall_sentiment_score'], 1),
            sentiment_label=final_analysis['sentiment_label'],
            confidence=round(final_analysis['confidence'], 1),
            summary_reasoning=final_analysis.get('summary_reasoning', ''),
            key_bullish_factors=final_analysis.get('key_bullish_factors', []),
            key_bearish_factors=final_analysis.get('key_bearish_factors', []),
            articles_analyzed=len(analyzed_articles),
            articles=analyzed_articles,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "local_llm_url": LOCAL_LLM_URL,
        "local_llm_model": LOCAL_LLM_MODEL
    }

@app.post("/test-llm")
async def test_llm():
    """Test endpoint to verify LLM integration works with real data"""
    try:
        # Test with real RSS scraping for a popular stock
        test_symbol = "AAPL"
        print(f"Testing LLM with real RSS data for {test_symbol}...")
        
        # Scrape real articles
        articles = await scrape_news_articles(test_symbol, max_articles=1)
        
        if not articles:
            return {
                "error": f"No real articles found for {test_symbol}. RSS feeds may be blocked.",
                "llm_working": False,
                "rss_working": False,
                "timestamp": datetime.now().isoformat()
            }
        
        # Test LLM with first real article
        test_article = articles[0]
        test_article['symbol'] = test_symbol
        
        # Analyze sentiment using LLM
        sentiment_result = await analyze_article_sentiment(test_article)
        
        return {
            "test_article": test_article,
            "llm_sentiment": sentiment_result,
            "llm_working": True,
            "rss_working": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "llm_working": False,
            "rss_working": False,
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 