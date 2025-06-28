#!/usr/bin/env python3
"""
Test RSS feed connectivity from Docker container
"""
import asyncio
import aiohttp
import feedparser
from datetime import datetime

# RSS feeds to test
RSS_FEEDS = [
    "https://www.reuters.com/arc/outboundfeeds/rss/",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://feeds.content.dowjones.io/public/rss/mw_markets",
    "https://seekingalpha.com/feed.xml"
]

async def test_rss_feed(session, url, name):
    """Test a single RSS feed"""
    try:
        print(f"Testing {name}: {url}")
        
        async with session.get(url, timeout=10) as response:
            print(f"  Status: {response.status}")
            
            if response.status == 200:
                content = await response.text()
                print(f"  Content length: {len(content)} characters")
                
                if len(content.strip()) < 100:
                    print(f"  ❌ Empty or invalid content")
                    return False
                
                # Try to parse as RSS
                feed = feedparser.parse(content)
                print(f"  Feed entries: {len(feed.entries)}")
                
                if feed.entries:
                    print(f"  ✅ RSS feed working")
                    return True
                else:
                    print(f"  ❌ No entries found")
                    return False
            else:
                print(f"  ❌ HTTP {response.status}")
                return False
                
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

async def main():
    """Test all RSS feeds"""
    print("=== RSS Feed Connectivity Test ===")
    print(f"Time: {datetime.now()}")
    print()
    
    async with aiohttp.ClientSession() as session:
        results = []
        
        for i, url in enumerate(RSS_FEEDS):
            name = f"Feed {i+1}"
            result = await test_rss_feed(session, url, name)
            results.append((name, result))
            print()
        
        # Summary
        print("=== Summary ===")
        working = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"Working feeds: {working}/{total}")
        
        if working == 0:
            print("❌ All RSS feeds are blocked/unavailable")
            print("This is why sentiment analysis fails")
        elif working < total:
            print(f"⚠️  Some feeds working ({working}/{total})")
        else:
            print("✅ All RSS feeds working")

if __name__ == "__main__":
    asyncio.run(main()) 