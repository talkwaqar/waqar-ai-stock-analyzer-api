# Stock Earnings Analyzer API

An intelligent FastAPI service that leverages AI (Gemini and OpenAI) to analyze stock financial data. The API processes natural language queries about stocks and provides detailed financial analysis using real-time data from Finnhub.

## 🚀 Features

- **Natural Language Processing**: Convert queries like "Compare Apple and Microsoft" into stock analysis
- **Multi-Stock Analysis**: Analyze multiple stocks simultaneously
- **Dual AI Support**: 
  - Google Gemini 2.0 Flash
  - OpenAI GPT-4o
- **Comprehensive Financial Analysis**:
  - Revenue & Profitability Metrics
  - Debt & Financial Health Indicators
  - Cash Flow Analysis
  - Growth & Performance Metrics
- **Real-time Data**: Live financial data from Finnhub API
- **Detailed Documentation**: Interactive API docs with examples

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **AI Models**: 
  - Google Gemini 2.0 Flash (Default)
  - OpenAI GPT-4o
- **Data Provider**: Finnhub API
- **Python**: 3.11+
- **Dependencies**: 
  - langchain
  - pydantic
  - google-generativeai
  - openai
  - finnhub-python

## How to run the project

1. Clone the repository
2. Install the dependencies
3. Run the project
    ```bash
    uvicorn src.app.main:app --reload
    ```

4. Access the API at `http://localhost:8000/docs` 

## Security Notes

- Never commit `.env` file with actual API keys
- Use `.env.example` for template configuration
- Rotate API keys if they've been accidentally exposed
- Set appropriate CORS policies in production

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
