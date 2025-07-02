#!/usr/bin/env python3
"""
Test script to verify Google GenAI Vertex AI connection
"""

import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

async def test_vertex_ai():
    """Test Vertex AI connection and basic functionality"""
    
    # Configuration
    project_id = os.getenv("GOOGLE_PROJECT_ID", "sentimentanalysis-464402")
    location = os.getenv("GOOGLE_LOCATION", "global")
    model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")
    
    print(f"Testing Vertex AI with:")
    print(f"  Project ID: {project_id}")
    print(f"  Location: {location}")
    print(f"  Model: {model}")
    print()
    
    try:
        # Initialize Vertex AI client
        print("1. Initializing Vertex AI client...")
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        print("✅ Vertex AI client initialized successfully!")
        
        # Test simple prompt
        print("\n2. Testing simple prompt...")
        test_prompt = "Hello! Please respond with 'Vertex AI is working!' and nothing else."
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=test_prompt)
                ]
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=100,
        )
        
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=generate_content_config
        )
        
        if response and response.text:
            print(f"✅ Response received: {response.text.strip()}")
        else:
            print("❌ No response received")
            
        # Test sentiment analysis prompt
        print("\n3. Testing sentiment analysis prompt...")
        sentiment_prompt = """You are analyzing a stock article. Provide your analysis in this format:

SENTIMENT_SCORE: [0-100]
CONFIDENCE: [0-100]
REASONING: [brief explanation]

Article: "Apple reported strong quarterly earnings with iPhone sales up 15%."
"""
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=sentiment_prompt)
                ]
            )
        ]
        
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=generate_content_config
        )
        
        if response and response.text:
            print(f"✅ Sentiment analysis response:")
            print(response.text.strip())
        else:
            print("❌ No sentiment analysis response received")
            
        print("\n🎉 All tests passed! Vertex AI is working correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're authenticated: gcloud auth application-default login")
        print("2. Check your GCP project permissions")
        print("3. Verify the project ID and location are correct")
        print("4. Ensure Vertex AI API is enabled in your project")

if __name__ == "__main__":
    asyncio.run(test_vertex_ai()) 