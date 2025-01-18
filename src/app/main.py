from fastapi import FastAPI, HTTPException, Depends
from app.dependencies import get_stock_analyzer
from app.schemas.stock import StockRequest, StockResponse
from app.services.langchain_service import StockAnalyzer
from fastapi.middleware.cors import CORSMiddleware
import logging
from langchain.globals import set_debug

# Configure logging
logging.basicConfig(level=logging.INFO)
set_debug(True)  # Enable Langchain debug logging at application startup

app = FastAPI(
    title="Stock Earnings Analyzer",
    description="""
    Analyzes multiple stocks' financial data using AI.
    
    Supported LLMs (Language Models):
    - Gemini: Gemini 2.0 Flash
    - OpenAI: GPT-4o model
    Default: Gemini
    
    Examples:
    - Single stock: "Analyze Apple stock"
    - Multiple stocks: "Compare Apple, Microsoft, and Google"
    - Natural language: "How are Tesla and Ford performing?"
    - With context: "Look at Amazon and Walmart's financial performance"
    
    The API will:
    1. Extract stock symbols from your input
    2. Fetch financial data for each stock
    3. Provide AI-powered analysis with specific metrics
    
    Analysis includes:
    - Revenue and profitability metrics
    - Debt and financial health indicators
    - Cash flow position
    - Growth metrics
    
    You can specify which LLM to use by adding the 'llm' parameter:
    - Use Gemini: {"user_input": "Analyze Apple", "llm": "gemini"}
    - Use OpenAI: {"user_input": "Analyze Apple", "llm": "openai"}
    """,
    version="1.0.0",
    openapi_tags=[{
        "name": "stocks",
        "description": "Stock analysis operations using AI-powered LLMs"
    }]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-stock", 
    response_model=StockResponse,
    tags=["stocks"],
    summary="Analyze multiple stocks",
    description="""
    Analyzes one or more stocks based on natural language input.
    
    Examples of valid inputs:
    - "Analyze Apple stock performance"
    - "Compare Tesla and Ford financials"
    - "How are Google, Amazon, and Microsoft doing?"
    
    Returns detailed financial analysis for each stock.
    
    You can specify which LLM to use by adding the 'llm' parameter:
    - Use Gemini: {"user_input": "Analyze Apple", "llm": "gemini"}
    - Use OpenAI: {"user_input": "Analyze Apple", "llm": "openai"}
    """
)
async def analyze_stock(
    request: StockRequest,
    analyzer: StockAnalyzer = Depends(get_stock_analyzer)
):
    """Analyze multiple stocks using AI"""
    try:
        result = await analyzer.analyze_stock(request.user_input, request.llm)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health",
    tags=["health"],
    summary="Health check",
    description="Check if the API is running"
)
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 