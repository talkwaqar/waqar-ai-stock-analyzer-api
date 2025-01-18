# Stock Earnings Analyzer API

A FastAPI-based service that provides AI-powered analysis of stock financial data using either Google's Gemini or OpenAI's GPT-4.

## Features

- Natural language processing to extract stock symbols from user queries
- Real-time financial data fetching from Finnhub
- AI-powered analysis of financial statements using multiple LLM options
- Detailed analysis of:
  - Revenue and Profitability metrics
  - Debt and Financial Health indicators
  - Cash Flow Position
  - Growth Metrics
- Support for multiple stock comparison
- Configurable LLM choice (Gemini or OpenAI)

## Tech Stack

- **Framework**: FastAPI
- **AI/ML**: 
  - Google Gemini Pro
  - OpenAI GPT-4
- **Financial Data**: Finnhub API
- **Python**: 3.11+


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
