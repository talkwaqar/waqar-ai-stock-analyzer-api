from langchain_openai import ChatOpenAI
from typing import Dict, List, Optional
from fastapi import HTTPException
import finnhub
import json
import google.generativeai as genai
import typing_extensions as typing
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.globals import set_debug
from app.prompts.stock_prompts import SYMBOL_EXTRACTION_PROMPT, FINANCIAL_ANALYSIS_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_debug(True)  # Enable Langchain debug logging

class StockAnalyzer:
    SUPPORTED_LLMS = {
        "openai": "OpenAI GPT-4o",
        "gemini": "Gemini 2.0 Flash"
    }

    def __init__(self, openai_api_key: str, finnhub_api_key: str, gemini_api_key: str, default_llm: str = "gemini"):
        """Initialize the StockAnalyzer with API keys and default LLM choice"""
        if default_llm not in self.SUPPORTED_LLMS:
            raise ValueError(
                f"Unsupported LLM: {default_llm}. "
                f"Supported LLMs are: {', '.join(self.SUPPORTED_LLMS.keys())}"
            )
        
        self.default_llm = default_llm
        logger.info(f"Initializing StockAnalyzer with default LLM: {self.SUPPORTED_LLMS[default_llm]}")
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_llm = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Gemini LLM initialized")
        
        # Initialize OpenAI
        self.openai_llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            api_key=openai_api_key,
            max_tokens=2000,
            model_kwargs={"response_format": {"type": "json_object"}},
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        logger.info("OpenAI LLM initialized")
        
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        logger.info("Finnhub client initialized")

    async def extract_symbols(self, text: str) -> Dict[str, List[str]]:
        """Extract stock symbols from text using the configured LLM"""
        try:
            if self.default_llm == "gemini":
                response = self.gemini_llm.generate_content(
                    SYMBOL_EXTRACTION_PROMPT.format(text=text),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        candidate_count=1,
                        response_mime_type="application/json"
                    )
                )
                # Extract JSON from response
                if hasattr(response, 'text'):
                    try:
                        # Clean the response text
                        text = response.text.strip()
                        # Remove markdown code blocks if present
                        if text.startswith("```json"):
                            text = text[7:-3]  # Remove ```json and ```
                        elif text.startswith("```"):
                            text = text[3:-3]  # Remove ``` and ```
                        # Parse JSON
                        return json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Gemini response as JSON")
                        # Fallback to manual extraction
                        return {"symbols": self._extract_symbols_manually(text)}
            else:
                response = await self.openai_llm.apredict(
                    SYMBOL_EXTRACTION_PROMPT.format(text=text)
                )
                return json.loads(response)
                
        except Exception as e:
            logger.error(f"Symbol extraction failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Symbol extraction failed: {str(e)}"
            )

    async def analyze_financials(self, financial_data: Dict, period: str) -> Dict:
        """Analyze financial data using the configured LLM"""
        try:
            logger.info(f"Starting financial analysis for period: {period}")
            logger.debug(f"Input financial data: {json.dumps(financial_data, indent=2)}")
            
            analysis_prompt = FINANCIAL_ANALYSIS_PROMPT.format(
                balance_sheet=str(financial_data.get('bs', {})),
                cash_flow=str(financial_data.get('cf', {})),
                income_statement=str(financial_data.get('ic', {})),
                period=period
            )
            
            logger.debug(f"Analysis prompt: {analysis_prompt}")
            
            if self.default_llm == "gemini":
                response = self.gemini_llm.generate_content(
                    analysis_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        candidate_count=1,
                        response_mime_type="application/json"
                    )
                )
                
                logger.debug(f"Gemini raw response: {response}")
                
                if hasattr(response, 'text'):
                    try:
                        text_content = response.text
                        logger.info(f"Raw response text: {text_content}")
                        
                        cleaned_text = text_content.replace('\\n', '\n').replace('\\"', '"')
                        logger.debug(f"Cleaned text: {cleaned_text}")
                        
                        parsed_json = json.loads(cleaned_text)
                        logger.info("Successfully parsed JSON response")
                        return parsed_json
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed: {str(e)}")
                        return {
                            "positives": ["Could not analyze financial data"],
                            "negatives": ["Analysis failed: Invalid JSON response"]
                        }
                
                return {
                    "positives": ["Could not analyze financial data"],
                    "negatives": ["Analysis failed: No text in response"]
                }
                
            else:
                response = await self.openai_llm.apredict(analysis_prompt)
                return json.loads(response)
                
        except Exception as e:
            logger.error(f"Financial analysis failed: {str(e)}", exc_info=True)
            return {
                "positives": ["Error during analysis"],
                "negatives": [f"Analysis error: {str(e)}"]
            }

    async def analyze_stock(self, user_input: str, llm_type: str = None) -> Dict:
        """Main function to analyze multiple stocks based on user input"""
        try:
            # Validate LLM type immediately if provided
            if llm_type:
                if llm_type not in self.SUPPORTED_LLMS:
                    error_msg = f"Unsupported LLM type: {llm_type}. Supported LLMs are: {', '.join(self.SUPPORTED_LLMS.keys())}"
                    logger.error(error_msg)
                    raise HTTPException(
                        status_code=400,
                        detail=error_msg
                    )
                
                original_llm = self.default_llm
                self.default_llm = llm_type
            
            try:
                logger.info(f"Starting stock analysis using {self.SUPPORTED_LLMS[self.default_llm]}")
                
                # Extract stock symbols
                extraction_result = await self.extract_symbols(user_input)
                symbols = extraction_result.get("symbols", [])
                
                if not symbols:
                    logger.warning("No symbols extracted from input")
                    raise HTTPException(
                        status_code=400,
                        detail="No stock symbols found to analyze. Please provide valid stock symbols."
                    )
                
                logger.info(f"Analyzing symbols: {symbols}")
                
                # Analyze each stock
                results = {}
                for symbol in symbols:
                    try:
                        logger.info(f"Fetching data for symbol: {symbol}")
                        financial_data = await self.fetch_financial_data(symbol)
                        
                        if not financial_data.get('data'):
                            logger.warning(f"No financial data available for {symbol}")
                            results[symbol] = {
                                "positives": [f"No financial data available for {symbol}"],
                                "negatives": ["Try another time period or verify the symbol"]
                            }
                            continue
                            
                        latest_report = self._extract_latest_report(financial_data)
                        
                        if not latest_report or not latest_report.get('report'):
                            logger.warning(f"No report data available for {symbol}")
                            results[symbol] = {
                                "positives": [f"Unable to analyze {symbol} at this time"],
                                "negatives": ["Financial data format not as expected"]
                            }
                            continue
                        
                        report = latest_report['report']
                        period = f"{latest_report.get('startDate', '')} to {latest_report.get('endDate', '')}"
                        
                        logger.info(f"Analyzing {symbol} for period: {period}")
                        analysis = await self.analyze_financials(report, period)
                        results[symbol] = analysis
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
                        results[symbol] = {
                            "positives": [f"Attempted to analyze {symbol}"],
                            "negatives": [f"Analysis error: {str(e)}"]
                        }
                
                return {
                    "analyses": results
                }
                
            finally:
                # Restore original LLM setting if it was changed
                if llm_type:
                    self.default_llm = original_llm
            
        except HTTPException as he:
            # Re-raise HTTP exceptions
            raise he
        except Exception as e:
            logger.error(f"Stock analysis failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )

    async def fetch_financial_data(self, symbol: str) -> Dict:
        """Fetch financial data from Finnhub"""
        try:
            logger.info(f"Fetching financial data for {symbol}")
            
            # Get all financials in one call
            financials = self.finnhub_client.financials_reported(
                symbol=symbol,
                freq='quarterly'  # Using quarterly for more recent data
            )
            
            logger.debug(f"Raw financial response: {json.dumps(financials, indent=2)}")
            
            if not financials or 'data' not in financials or not financials['data']:
                logger.warning(f"No financial data found for {symbol}")
                return {'data': []}
            
            # Sort by date to get the most recent report
            sorted_data = sorted(
                financials['data'],
                key=lambda x: x.get('endDate', ''),
                reverse=True
            )
            
            latest_report = sorted_data[0]
            logger.info(f"Latest report date: {latest_report.get('endDate')}")
            
            return {
                'data': [latest_report]
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch financial data for {symbol}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch financial data: {str(e)}"
            )

    def _extract_latest_report(self, financial_data: Dict) -> Dict:
        """Extract the latest financial report"""
        try:
            logger.info("Starting to extract latest report")
            logger.debug(f"Input financial data: {json.dumps(financial_data, indent=2)}")
            
            if not financial_data.get('data') or not financial_data['data']:
                logger.warning("No data found in financial_data")
                return {'report': None}
            
            # Get the first (most recent) report
            latest_report = financial_data['data'][0]
            
            if 'report' not in latest_report:
                logger.warning("No report field in latest data")
                return {'report': None}
                
            report_data = latest_report['report']
            
            # Extract dates
            start_date = latest_report.get('startDate', '')
            end_date = latest_report.get('endDate', '')
            
            logger.info(f"Found report for period: {start_date} to {end_date}")
            logger.debug(f"Report data: {json.dumps(report_data, indent=2)}")
            
            return {
                'report': report_data,
                'startDate': start_date,
                'endDate': end_date
            }
            
        except Exception as e:
            logger.error(f"Failed to extract latest report: {str(e)}", exc_info=True)
            return {'report': None}

    def _extract_symbols_manually(self, text: str) -> List[str]:
        """Manually extract stock symbols as fallback"""
        logger.warning("Falling back to manual symbol extraction")
        common_symbols = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "facebook": "META",
            "netflix": "NFLX",
            "nvidia": "NVDA"
        }
        
        text = text.lower()
        symbols = []
        
        for company, symbol in common_symbols.items():
            if company in text:
                symbols.append(symbol)
                
        logger.info(f"Manually extracted symbols: {symbols}")
        return symbols 