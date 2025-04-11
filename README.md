# Multi-Agent VC Startup Evaluator - Setup and Usage Instructions

This document provides instructions on how to set up and use the Multi-Agent VC Startup Evaluator system.

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository or create a new project directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:

   ```bash
   pip install fastapi uvicorn openai streamlit PyPDF2 python-multipart
   ```

4. Set your OpenAI API key as an environment variable:

   ```bash
   # On Linux/macOS
   export OPENAI_API_KEY="your-api-key-here"

   # On Windows (Command Prompt)
   set OPENAI_API_KEY=your-api-key-here

   # On Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

## Running the Application

1. Save the provided code as `main.py`
2. Run the application:
   ```bash
   python main.py
   ```
3. The application will start two services:
   - FastAPI backend on port 8000 (http://localhost:8000)
   - Streamlit frontend (a new browser window should open automatically)

## Using the Application

1. In the Streamlit interface, enter the name of the startup you want to evaluate
2. Upload the startup's pitch deck PDF file
3. Click "Start Evaluation"
4. Wait for the analysis to complete (this may take a few minutes)
5. Review the results from all three agents:
   - Pain Point and Team Fit Analysis
   - Market and Competition Analysis
   - Final Investment Recommendation

## API Documentation

If you prefer to use the API directly:

- API documentation: http://localhost:8000/docs
- Endpoints:
  - POST `/evaluate-startup/`: Upload a pitch deck for evaluation
  - GET `/evaluation-status/{evaluation_id}`: Check the status of an evaluation

## System Architecture

This application implements a multi-agent system with three specialized agents:

1. **Pain Point and Team Fit Agent**: Evaluates problem clarity and team readiness
2. **Market and Competition Analyst Agent**: Analyzes market size and competitive positioning
3. **Final Recommendation Agent**: Synthesizes outputs to provide an investment recommendation
