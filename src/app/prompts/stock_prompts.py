SYMBOL_EXTRACTION_PROMPT = """
Extract stock symbols from the following text and return them in JSON format.
Convert company names to their stock symbols (e.g., Apple -> AAPL, Microsoft -> MSFT, Google -> GOOGL).

Text: {text}

Return format must be:
{{"symbols": ["SYMBOL1", "SYMBOL2", ...]}}

Example:
Input: "Compare Apple and Microsoft stocks"
Output: {{"symbols": ["AAPL", "MSFT"]}}
"""

FINANCIAL_ANALYSIS_PROMPT = """
Analyze these financial statements and return a JSON response with positives and negatives:
Balance Sheet: {balance_sheet}
Cash Flow: {cash_flow}
Income Statement: {income_statement}
Period: {period}

Focus on:
1. Revenue and Profitability (with specific numbers)
2. Debt and Financial Health (with ratios)
3. Cash Flow Position (with actual figures)
4. Growth Metrics (with percentages)

Response format must be:
{{"positives": ["point 1", "point 2"], "negatives": ["point 1", "point 2"]}}
""" 