"""
LLM enhancement module for adjusting LSTM predictions based on news sentiment
Handles LLM API communication and prediction adjustments
"""
import requests
import re
from datetime import datetime


class LLMEnhancer:
    """LLM-based prediction enhancement using news sentiment"""
    
    def __init__(self, model_name="llama3", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
    
    def prepare_prompt(self, symbol, lstm_prediction, news_data, prediction_date):
        """Prepare prompt for LLM adjustment"""
        # Use top 10 news items for context
        relevant_news = news_data[:10]
        
        news_section = ""
        for i, news in enumerate(relevant_news):
            news_section += f"{i+1}. {news.get('title', 'No Title')}\n"
            if news.get('summary'):
                news_section += f"   Summary: {news.get('summary')[:100]}...\n"
            
            published_str = news.get('published', 'Unknown')
            if len(published_str) == 8:
                try:
                    formatted_date = datetime.strptime(published_str, '%Y%m%d').strftime('%Y-%m-%d')
                    news_section += f"   Date: {formatted_date}\n"
                except:
                    news_section += f"   Date: {published_str}\n"
            else:
                news_section += f"   Date: {published_str}\n"
            
            if news.get('publisher'):
                news_section += f"   Publisher: {news.get('publisher')}\n"
            news_section += "\n"
        
        if not news_section:
            news_section = "No news data available."
        
        prompt = f"""You are a financial analyst. Analyze news about {symbol} and provide an adjustment to the algorithmic prediction.

STOCK: {symbol}
DATE: {prediction_date.strftime('%Y-%m-%d')}
LSTM PREDICTION: ${lstm_prediction:.2f}

NEWS CONTEXT:
{news_section}

Based on the overall sentiment and information in the news, consider both recent and historical news context.

FORMAT YOUR RESPONSE AS:
Analysis: [Brief analysis]
Adjustment Factor: [+/-X.X]%
Adjusted Price: $[calculated price]
"""
        return prompt
    
    def get_adjustment(self, symbol, lstm_prediction, news_data, prediction_date):
        """Get LLM adjustment for the prediction"""
        try:
            prompt = self.prepare_prompt(symbol, lstm_prediction, news_data, prediction_date)
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                adjustment_info = self._extract_adjustment(llm_response)
                adjustment_factor = adjustment_info['adjustment']
                adjusted_prediction = lstm_prediction * (1 + adjustment_factor)
                return adjusted_prediction, adjustment_info['reasoning']
            else:
                print(f"LLM API error: {response.status_code}")
                return lstm_prediction, "LLM API error - no adjustment applied"
                
        except requests.exceptions.RequestException as e:
            print(f"LLM request failed: {e}")
            return lstm_prediction, f"LLM request failed: {e}"
        except Exception as e:
            print(f"LLM adjustment error: {e}")
            return lstm_prediction, f"LLM adjustment error: {e}"
    
    def _extract_adjustment(self, response_text):
        """Extract adjustment information from LLM response"""
        # Look for adjustment factor pattern
        adjustment_pattern = r"Adjustment Factor: ([+-]?\d+\.?\d*)%"
        adjustment_match = re.search(adjustment_pattern, response_text)
        
        if adjustment_match:
            adjustment = float(adjustment_match.group(1)) / 100.0
            # Limit adjustment to ±5%
            adjustment = max(min(adjustment, 0.05), -0.05)
            return {"adjustment": adjustment, "reasoning": response_text}
        
        # Fallback: look for any percentage
        percent_pattern = r"([+-]?\d+\.?\d*)%"
        percent_matches = re.findall(percent_pattern, response_text)
        if percent_matches:
            adjustment = float(percent_matches[0]) / 100.0
            adjustment = max(min(adjustment, 0.05), -0.05)
            return {"adjustment": adjustment, "reasoning": response_text}
        
        return {"adjustment": 0.0, "reasoning": "No valid adjustment found"}
    
    def enhance_predictions(self, symbol, lstm_predictions, news_data, start_date):
        """Enhance multiple LSTM predictions with LLM adjustments"""
        enhanced_predictions = []
        adjustments_log = []
        
        for i, lstm_pred in enumerate(lstm_predictions):
            prediction_date = start_date + pd.Timedelta(days=i)
            
            enhanced_pred, reasoning = self.get_adjustment(
                symbol, lstm_pred, news_data, prediction_date
            )
            
            enhanced_predictions.append(enhanced_pred)
            adjustments_log.append({
                'date': prediction_date,
                'lstm_prediction': lstm_pred,
                'enhanced_prediction': enhanced_pred,
                'adjustment_factor': (enhanced_pred - lstm_pred) / lstm_pred,
                'reasoning': reasoning
            })
        
        return enhanced_predictions, adjustments_log