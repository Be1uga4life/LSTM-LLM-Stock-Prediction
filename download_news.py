"""
News data downloader for stock prediction
Supports multiple news sources with fallback options
"""
import requests
import json
import os
from datetime import datetime, timedelta
import time

class NewsDownloader:
    """Download news data from various sources"""
    
    def __init__(self):
        self.sources = {
            'alphavantage': {
                'api_key': 'F7R9ZEGOKU8JUIO6',  # Your existing key
                'base_url': 'https://www.alphavantage.co/query'
            }
        }
    
    def download_alphavantage_news(self, symbol, days_back=60):
        """Download news from Alpha Vantage"""
        print(f"📰 Downloading news for {symbol} from Alpha Vantage...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API (YYYYMMDDTHHMM)
        time_from = start_date.strftime("%Y%m%dT0000")
        time_to = end_date.strftime("%Y%m%dT2359")
        
        # Build URL
        url = f"{self.sources['alphavantage']['base_url']}?function=NEWS_SENTIMENT&tickers={symbol}&time_from={time_from}&time_to={time_to}&apikey={self.sources['alphavantage']['api_key']}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            elif 'Note' in data:
                raise Exception(f"API Rate Limited: {data['Note']}")
            
            # Format news data
            news_formatted = []
            feed_items = data.get('feed', [])
            
            print(f"📊 Found {len(feed_items)} news items")
            
            for item in feed_items:
                try:
                    # Parse published date
                    time_published = item.get('time_published', '')
                    if time_published:
                        published_date = datetime.strptime(time_published, '%Y%m%dT%H%M%S').strftime('%Y-%m-%d')
                    else:
                        published_date = datetime.now().strftime('%Y-%m-%d')
                    
                    news_formatted.append({
                        "title": item.get('title', ''),
                        "summary": item.get('summary', ''),
                        "published": published_date,
                        "publisher": item.get('source', ''),
                        "link": item.get('url', ''),
                        "type": "Alpha Vantage",
                        "sentiment_score": item.get('overall_sentiment_score', 0),
                        "sentiment_label": item.get('overall_sentiment_label', 'neutral')
                    })
                except Exception as e:
                    print(f"⚠️ Error parsing news item: {e}")
                    continue
            
            return news_formatted
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
        except Exception as e:
            print(f"❌ API error: {e}")
            return None
    
    def create_sample_news(self, symbol, days_back=60):
        """Create sample news data for testing"""
        print(f"📝 Creating sample news data for {symbol}...")
        
        sample_news = []
        base_date = datetime.now()
        
        # Create diverse sample news
        news_templates = [
            {
                "title": f"{symbol} reports strong quarterly earnings",
                "summary": f"{symbol} exceeded analyst expectations with strong revenue growth and positive outlook.",
                "sentiment": "positive"
            },
            {
                "title": f"{symbol} announces new product launch",
                "summary": f"{symbol} unveiled innovative new products that could drive future growth.",
                "sentiment": "positive"
            },
            {
                "title": f"Market volatility affects {symbol} stock price",
                "summary": f"{symbol} shares fluctuated amid broader market uncertainty and economic concerns.",
                "sentiment": "neutral"
            },
            {
                "title": f"Analysts upgrade {symbol} price target",
                "summary": f"Several analysts raised their price targets for {symbol} citing strong fundamentals.",
                "sentiment": "positive"
            },
            {
                "title": f"{symbol} faces regulatory scrutiny",
                "summary": f"Regulatory agencies are reviewing {symbol}'s business practices, creating uncertainty.",
                "sentiment": "negative"
            },
            {
                "title": f"{symbol} CEO discusses company strategy",
                "summary": f"In a recent interview, {symbol}'s CEO outlined the company's strategic direction.",
                "sentiment": "neutral"
            }
        ]
        
        # Generate news over the time period
        for i in range(min(len(news_templates), 10)):  # Limit to avoid too much sample data
            news_date = base_date - timedelta(days=i*3)  # Space out over time
            template = news_templates[i % len(news_templates)]
            
            sample_news.append({
                "title": template["title"],
                "summary": template["summary"],
                "published": news_date.strftime('%Y-%m-%d'),
                "publisher": "Sample News Corp",
                "link": f"https://example.com/news/{i}",
                "type": "Sample Data",
                "sentiment_score": 0.1 if template["sentiment"] == "positive" else -0.1 if template["sentiment"] == "negative" else 0.0,
                "sentiment_label": template["sentiment"]
            })
        
        print(f"✅ Created {len(sample_news)} sample news items")
        return sample_news
    
    def download_news(self, symbol, days_back=60, use_sample_fallback=True):
        """Download news with fallback options"""
        print(f"🔍 Downloading news for {symbol} (last {days_back} days)...")
        
        # Try Alpha Vantage first
        news_data = self.download_alphavantage_news(symbol, days_back)
        
        if news_data and len(news_data) > 0:
            print(f"✅ Successfully downloaded {len(news_data)} news items from Alpha Vantage")
            return news_data
        
        # Fallback to sample data if enabled
        if use_sample_fallback:
            print("⚠️ API failed, using sample news data for testing")
            return self.create_sample_news(symbol, days_back)
        
        print("❌ No news data available")
        return []
    
    def save_news(self, news_data, symbol, output_dir='news'):
        """Save news data to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_news_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 News data saved to: {filepath}")
        return filepath

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download news data for stock prediction')
    parser.add_argument('symbol', help='Stock symbol (e.g., TSLA, MSFT)')
    parser.add_argument('--days', type=int, default=60, help='Days of news to download')
    parser.add_argument('--output-dir', default='news', help='Output directory')
    parser.add_argument('--no-fallback', action='store_true', help='Disable sample data fallback')
    
    args = parser.parse_args()
    
    # Download news
    downloader = NewsDownloader()
    news_data = downloader.download_news(
        args.symbol.upper(), 
        args.days, 
        use_sample_fallback=not args.no_fallback
    )
    
    if news_data:
        filepath = downloader.save_news(news_data, args.symbol.upper(), args.output_dir)
        
        # Show preview
        print(f"\n📰 News Preview (first 3 items):")
        for i, item in enumerate(news_data[:3]):
            print(f"\n{i+1}. {item['title']}")
            print(f"   Date: {item['published']}")
            print(f"   Publisher: {item['publisher']}")
            print(f"   Sentiment: {item.get('sentiment_label', 'N/A')}")
        
        return 0
    else:
        print("❌ Failed to download news data")
        return 1

if __name__ == "__main__":
    exit(main())