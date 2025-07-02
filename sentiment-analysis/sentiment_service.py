from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
from datetime import datetime
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
import re
from contextlib import asynccontextmanager
from google_genai_client import get_llm_config, get_google_genai_model, call_llm_batch
from fastapi.responses import JSONResponse

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the Google GenAI model on startup
    print("Application startup: Initializing Google GenAI model...")
    model = get_google_genai_model()
    if model:
        print("Google GenAI model initialized successfully.")
    else:
        print("ERROR: Google GenAI model failed to initialize. The application might not function correctly.")
    yield
    # Clean up resources if needed on shutdown (not necessary for this client)
    print("Application shutdown.")

app = FastAPI(title="Sentiment Analysis Service", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model definitions for the API
class SentimentRequest(BaseModel):
    symbol: str

class BatchSentimentRequest(BaseModel):
    symbols: List[str]

class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: float
    sentiment_label: str
    confidence: float
    summary_reasoning: Optional[str] = None
    key_bullish_factors: Optional[List[str]] = None
    key_bearish_factors: Optional[List[str]] = None
    investment_recommendation: Optional[str] = None
    business_quality_assessment: Optional[str] = None
    valuation_perspective: Optional[str] = None
    contrarian_signal: Optional[str] = None
    intelligent_investor_summary: Optional[str] = None
    articles_analyzed: int
    articles: List[Dict] # This will contain sources found by the model
    timestamp: str

def get_master_prompt(symbol: str) -> str:
    """
    Creates the master prompt for the Gemini model, instructing it to perform
    sentiment analysis based on its training data and knowledge.
    """
    return f"""
You are a world-class financial analyst specializing in sentiment analysis. Your task is to analyze the stock symbol '{symbol}' based on your knowledge of recent market trends, financial news, and business fundamentals.

INSTRUCTIONS:
1. Analyze the current market sentiment for '{symbol}' based on your knowledge of recent developments, earnings, business performance, and market conditions.
2. Consider both fundamental analysis (business quality, competitive position, financial health) and market sentiment factors.
3. Provide a comprehensive analysis that focuses on business fundamentals and investment merit, not just stock price movement.
4. **You MUST return your analysis in the exact key-value format specified below.** Do not add any extra text or explanations outside of this format.

RETURN FORMAT:
OVERALL_SENTIMENT_SCORE: [A number from 0 (extremely bearish) to 100 (extremely bullish)]
SENTIMENT_LABEL: [Bullish/Bearish/Neutral]
CONFIDENCE: [A number from 0 to 100, representing your confidence in the analysis]
SUMMARY_REASONING: [A concise, 2-3 sentence explanation for your sentiment score from an investor's perspective]
KEY_BULLISH_FACTORS: [A comma-separated list of the top 3-5 bullish points based on your knowledge]
KEY_BEARISH_FACTORS: [A comma-separated list of the top 3-5 bearish points based on your knowledge]
INVESTMENT_RECOMMENDATION: [Buy/Hold/Sell/Avoid]
BUSINESS_QUALITY_ASSESSMENT: [High/Medium/Low, based on competitive moat, management, and profitability]
VALUATION_PERSPECTIVE: [Likely Overvalued/Fairly Valued/Likely Undervalued, based on your assessment]
CONTRARIAN_SIGNAL: [Strong Buy/Buy/Hold/Avoid, based on whether current sentiment seems to misprice the long-term quality]
INTELLIGENT_INVESTOR_SUMMARY: [A 3-sentence summary as if explaining to a fellow value investor]
ARTICLES_ANALYZED: [Set this to 5 to indicate analysis based on your training knowledge]
ARTICLES: [Return an empty JSON array: []]
"""

def parse_llm_response(response_text: str) -> Optional[Dict]:
    """
    Parses the structured key-value response from the LLM.
    This is a more robust version that handles multi-line values.
    """
    if not response_text:
        return None
    
    data = {}
    # Use regex to find key-value pairs, allowing for multi-line values.
    # It looks for a KEY:, then captures everything until the next KEY:.
    # The `(?=...|$)` is a lookahead that asserts the position is followed by another key or end of string.
    pattern = re.compile(r"(\w+):\s*(.*?)(?=\n\w+:\s*|$)", re.DOTALL)
    
    for match in pattern.finditer(response_text):
        key = match.group(1).strip().lower().replace(' ', '_')
        value = match.group(2).strip()
        data[key] = value

    # Specific parsing for numeric and list/JSON fields
    try:
        data['overall_sentiment'] = float(data.get('overall_sentiment_score', 50.0))
        data['confidence'] = float(data.get('confidence', 50.0))
        data['articles_analyzed'] = int(data.get('articles_analyzed', 0))
        
        if 'key_bullish_factors' in data and isinstance(data['key_bullish_factors'], str):
            data['key_bullish_factors'] = [f.strip() for f in data['key_bullish_factors'].split(',') if f.strip()]
        
        if 'key_bearish_factors' in data and isinstance(data['key_bearish_factors'], str):
            data['key_bearish_factors'] = [f.strip() for f in data['key_bearish_factors'].split(',') if f.strip()]
            
        if 'articles' in data and isinstance(data['articles'], str):
            try:
                # The LLM might return a JSON string, try to parse it
                data['articles'] = json.loads(data['articles'])
            except json.JSONDecodeError:
                print(f"[PARSE] Could not parse 'articles' field as JSON. Value was: {data['articles']}")
                data['articles'] = [] # Default to empty list on failure
        else:
            data['articles'] = [] # Ensure articles is a list
                
    except (ValueError, TypeError) as e:
        print(f"[PARSE] Error during data type conversion: {e}")
        # Return partial data or None
        return data # Return what was parsed

    return data

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis Service",
        "status": "running",
        **get_llm_config()
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyzes sentiment for a single stock symbol using the cost-effective Batch API.
    """
    print(f"[MAIN] Starting agentic sentiment analysis for {request.symbol} via batch API.")
    
    try:
        # Step 1: Create a batch request with a single prompt
        prompts = [get_master_prompt(request.symbol)]
        
        # Step 2: Call the LLM with the batch endpoint
        print(f"[MAIN] Calling LLM batch endpoint for single symbol: {request.symbol}...")
        batch_responses = await call_llm_batch(prompts, enable_search=True)
        
        if not batch_responses or not batch_responses[0]:
            raise HTTPException(status_code=500, detail="LLM call failed or returned an empty response.")
        
        response_text = batch_responses[0]
        
        # Step 3: Parse the structured response
        print(f"[MAIN] Parsing LLM response for {request.symbol}...")
        parsed_data = parse_llm_response(response_text)
        
        if not parsed_data:
            print(f"[MAIN] Failed to parse the response from the LLM. Raw response:\n{response_text}")
            raise HTTPException(status_code=500, detail="Failed to parse the structured response from the LLM.")
            
        # Step 4: Construct the final response object
        print(f"[MAIN] Constructing final response for {request.symbol}")
        response = SentimentResponse(
            symbol=request.symbol,
            overall_sentiment=round(parsed_data.get('overall_sentiment', 50.0), 1),
            sentiment_label=parsed_data.get('sentiment_label', 'Neutral'),
            confidence=round(parsed_data.get('confidence', 0), 1),
            summary_reasoning=parsed_data.get('summary_reasoning', ''),
            key_bullish_factors=parsed_data.get('key_bullish_factors', []),
            key_bearish_factors=parsed_data.get('key_bearish_factors', []),
            investment_recommendation=parsed_data.get('investment_recommendation'),
            business_quality_assessment=parsed_data.get('business_quality_assessment'),
            valuation_perspective=parsed_data.get('valuation_perspective'),
            contrarian_signal=parsed_data.get('contrarian_signal'),
            intelligent_investor_summary=parsed_data.get('intelligent_investor_summary'),
            articles_analyzed=parsed_data.get('articles_analyzed', 0),
            articles=parsed_data.get('articles', []),
            timestamp=datetime.now().isoformat()
        )
        
        print(f"[MAIN] Analysis complete for {request.symbol}. Overall sentiment: {response.overall_sentiment}")
        return response
        
    except Exception as e:
        print(f"[MAIN] ERROR in agentic sentiment analysis for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/analyze-batch", response_model=List[SentimentResponse])
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """
    Analyzes sentiment for a batch of stock symbols using the cost-effective Batch API.
    """
    print(f"[MAIN] Starting agentic sentiment analysis for {request.symbols} via batch API.")
    prompts = [get_master_prompt(symbol) for symbol in request.symbols]
    
    results = await call_llm_batch(prompts, enable_search=True)
    
    if not results:
        return JSONResponse(status_code=500, content={"error": "LLM call failed or returned an empty response."})

    final_results = []
    for i, response_text in enumerate(results):
        symbol = request.symbols[i]
        if not response_text:
            print(f"[MAIN] Batch analysis for symbol '{symbol}' failed or returned empty response.")
            # Optionally create a default error response
            # For now, we'll just skip it
            continue

        parsed_data = parse_llm_response(response_text)
        if not parsed_data:
            print(f"[MAIN] Failed to parse response for symbol '{symbol}'.")
            continue
        
        response = SentimentResponse(
            symbol=symbol,
            overall_sentiment=round(parsed_data.get('overall_sentiment', 50.0), 1),
            sentiment_label=parsed_data.get('sentiment_label', 'Neutral'),
            confidence=round(parsed_data.get('confidence', 0), 1),
            summary_reasoning=parsed_data.get('summary_reasoning', ''),
            key_bullish_factors=parsed_data.get('key_bullish_factors', []),
            key_bearish_factors=parsed_data.get('key_bearish_factors', []),
            investment_recommendation=parsed_data.get('investment_recommendation'),
            business_quality_assessment=parsed_data.get('business_quality_assessment'),
            valuation_perspective=parsed_data.get('valuation_perspective'),
            contrarian_signal=parsed_data.get('contrarian_signal'),
            intelligent_investor_summary=parsed_data.get('intelligent_investor_summary'),
            articles_analyzed=parsed_data.get('articles_analyzed', 0),
            articles=parsed_data.get('articles', []),
            timestamp=datetime.now().isoformat()
        )
        final_results.append(response)

    print(f"[MAIN] Batch analysis complete. Returning {len(final_results)} results.")
    return final_results

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        **get_llm_config()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 