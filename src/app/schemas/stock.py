from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from app.core.config import get_settings
from typing_extensions import Annotated

settings = get_settings()

class StockRequest(BaseModel):
    user_input: str = Field(
        ...,
        description="Natural language input describing which stocks to analyze",
        examples=[
            "Analyze Apple stock",
            "Compare Tesla and Ford performance",
            "How are Google, Amazon, and Microsoft doing?",
            "Look at Netflix earnings"
        ]
    )
    llm: Optional[str] = Field(
        None,
        description="LLM type to use (openai or gemini). Default: gemini",
        examples=["gemini", "openai"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_input": "Analyze Apple stock",
                    "llm": "gemini"
                },
                {
                    "user_input": "Compare Tesla and Ford",
                    "llm": "openai"
                },
                {
                    "user_input": "How are Google and Microsoft performing?",
                    "llm": "gemini"
                }
            ]
        }
    }

class StockAnalysis(BaseModel):
    positives: List[str] = Field(
        ...,
        description="List of positive insights about the stock",
        examples=[
            [
                "The company generated a substantial net income of $79 billion, indicating strong profitability.",
                "Gross margin is very high at $136.8 billion, showing efficient cost management relative to revenue.",
                "Operating income is also robust at $93.6 billion, demonstrating effective operational performance.",
                "The company has a strong cash flow from operations at $91.4 billion, indicating a healthy ability to generate cash from its core business.",
                "The company has a large amount of marketable securities, both current and non-current, totaling $127.476 billion, which provides financial flexibility.",
                "The company has a large amount of cash and cash equivalents at $25.565 billion."
            ]
        ]
    )
    negatives: List[str] = Field(
        ...,
        description="List of negative insights about the stock",
        examples=[
            [
                "The company has a significant amount of total liabilities at $264.9 billion, which is a large portion of total assets.",
                "The company has a large amount of current liabilities at $131.6 billion, which is greater than the total current assets of $125.4 billion, indicating a potential liquidity risk.",
                "The company has a negative retained earnings of -$4.7 billion, which is a concern.",
                "The company spent a significant amount on share repurchases at $69.8 billion, which could be seen as a less productive use of cash than investing in growth.",
                "The company has a negative cash flow from financing activities at -$97 billion, primarily due to share repurchases and dividend payments.",
                "The company's cash and cash equivalents decreased by $4.1 billion during the period."
            ]
        ]
    )

class StockResponse(BaseModel):
    analyses: Dict[str, StockAnalysis] = Field(
        ...,
        description="Analysis results for each requested stock symbol",
        examples=[{
            "AAPL": {
                "positives": [
                    "The company generated a substantial net income of $79 billion, indicating strong profitability.",
                    "Gross margin is very high at $136.8 billion, showing efficient cost management relative to revenue.",
                    "Operating income is also robust at $93.6 billion, demonstrating effective operational performance.",
                    "The company has a strong cash flow from operations at $91.4 billion, indicating a healthy ability to generate cash from its core business.",
                    "The company has a large amount of marketable securities, both current and non-current, totaling $127.476 billion, which provides financial flexibility.",
                    "The company has a large amount of cash and cash equivalents at $25.565 billion."
                ],
                "negatives": [
                    "The company has a significant amount of total liabilities at $264.9 billion, which is a large portion of total assets.",
                    "The company has a large amount of current liabilities at $131.6 billion, which is greater than the total current assets of $125.4 billion, indicating a potential liquidity risk.",
                    "The company has a negative retained earnings of -$4.7 billion, which is a concern.",
                    "The company spent a significant amount on share repurchases at $69.8 billion, which could be seen as a less productive use of cash than investing in growth.",
                    "The company has a negative cash flow from financing activities at -$97 billion, primarily due to share repurchases and dividend payments.",
                    "The company's cash and cash equivalents decreased by $4.1 billion during the period."
                ]
            }
        }]
    ) 