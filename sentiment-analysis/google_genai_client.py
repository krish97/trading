import os
import asyncio
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()

# Google GenAI Configuration
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "sentimentanalysis-464402")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")

# Rate limiting
CONCURRENT_REQUEST_LIMIT = 5
semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

# Initialize Google GenAI Model object
model: Optional[genai.GenerativeModel] = None

def get_google_config() -> Dict[str, Any]:
    """Get Google GenAI configuration"""
    return {
        "model": GOOGLE_MODEL,
        "project_id": GOOGLE_PROJECT_ID,
        "location": GOOGLE_LOCATION,
        "provider": "Google GenAI SDK",
        "model_initialized": model is not None
    }

def get_google_genai_model() -> Optional[genai.GenerativeModel]:
    """Initializes and returns the Google GenAI GenerativeModel."""
    global model
    if model:
        return model

    if not GOOGLE_PROJECT_ID:
        print("WARNING: GOOGLE_PROJECT_ID is not set. GenAI client cannot be initialized.")
        return None
    
    try:
        print(f"Initializing Google GenAI model '{GOOGLE_MODEL}' for project='{GOOGLE_PROJECT_ID}'")
        
        # Configure the API key or use default credentials
        genai.configure()
        
        # Create the model with simple generation config
        generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
        )
        
        model = genai.GenerativeModel(
            model_name=GOOGLE_MODEL,
            generation_config=generation_config
        )
        
        print(f"Google GenAI model initialized successfully!")
        return model
    except Exception as e:
        print(f"Failed to initialize Google GenAI model: {e}")
        model = None
        return None

async def call_google_genai(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Call Google GenAI with the given prompt, with rate limiting and retries.
    """
    if not model:
        get_google_genai_model() # Attempt to initialize
        if not model:
            print("ERROR: GenAI Model not initialized")
            return None
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                print(f"[GOOGLE GENAI] Calling model {GOOGLE_MODEL} (Attempt {attempt + 1}/{max_retries})")
                
                # Use the simple generate_content method
                response = await model.generate_content_async(prompt)
                
                if response and response.text:
                    print(f"[GOOGLE GENAI] Response received, length: {len(response.text)}")
                    return response.text.strip()
                else:
                    print(f"[GOOGLE GENAI] No response text received")
                    return None

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[GOOGLE GENAI] Rate limit hit. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[GOOGLE GENAI] Rate limit error after {max_retries} attempts. Giving up.")
                        return None
                
                print(f"[GOOGLE GENAI] Error calling model: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None
        return None

async def call_llm_batch(prompts: List[str], enable_search: bool = True) -> List[Optional[str]]:
    """
    Calls the Google GenAI model with a batch of prompts using async calls.
    Note: enable_search parameter is kept for compatibility but search tool is not used in this simplified version.
    """
    if not model:
        get_google_genai_model() # Attempt to initialize
        if not model:
            print("ERROR: GenAI Model not initialized")
            return [None] * len(prompts)
    
    print(f"[GOOGLE GENAI BATCH] Calling model {GOOGLE_MODEL} with {len(prompts)} prompts.")
    
    try:
        async def generate_single(prompt: str) -> Optional[str]:
            try:
                response = await model.generate_content_async(prompt)
                return response.text if response and response.text else None
            except Exception as e:
                print(f"[GOOGLE GENAI BATCH] Error in single batch request: {e}")
                return None

        # Execute all prompts concurrently
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"[GOOGLE GENAI BATCH] Exception in result: {result}")
                final_results.append(None)
            else:
                final_results.append(result)

        print(f"[GOOGLE GENAI BATCH] Batch call completed, received {len([r for r in final_results if r])} successful responses out of {len(prompts)}.")
        return final_results

    except Exception as e:
        print(f"[GOOGLE GENAI BATCH] Error calling model in batch: {e}")
        return [None] * len(prompts)

get_llm_config = get_google_config 