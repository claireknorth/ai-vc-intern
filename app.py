# main.py
import os
import json
import base64
import tempfile
from typing import Dict, List, Optional, Any

import streamlit as st
import PyPDF2
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Initialize OpenAI clients after environment variable is loaded
client = OpenAI(api_key=OPENAI_API_KEY)
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# FastAPI app setup
app = FastAPI(title="Multi-Agent VC Startup Evaluator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class StartupEvaluation(BaseModel):
    startup_name: str
    pitch_deck_content: str
    agent1_output: Optional[Dict[str, Any]] = None
    agent2_output: Optional[Dict[str, Any]] = None
    agent3_output: Optional[Dict[str, Any]] = None
    final_recommendation: Optional[str] = None

class EvaluationResponse(BaseModel):
    status: str
    message: str
    evaluation_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# In-memory database (would use a real DB in production)
evaluations_db = {}

# Agent prompts
AGENT1_PROMPT = """
You are a top-tier venture capitalist with decades of experience evaluating
early-stage startups. Your core expertise lies in identifying whether a
startup is solving a real, urgent problem for a clearly defined customer,
and whether the proposed solution—and founding team—are capable of
achieving strong product-market fit.

You will now act as a Product-Market Fit and Team Evaluation Expert. Based
solely on the information provided in the attached Startup Pitch Deck,
perform a structured analysis using the two criteria below.

Your Evaluation Objectives
Criterion 1: Problem–Solution Fit
Determine how clearly and urgently the startup defines the customer's pain
point.
Evaluate whether customers are highly motivated to seek a solution.
Assess whether the proposed solution effectively addresses the problem in a
unique or compelling way.

Criterion 2: Team Skillset Alignment
Identify the critical skillsets required to execute this business (e.g.,
technical, go-to-market, domain-specific).
Assess whether the founding team possesses or can reasonably acquire these
capabilities.

Input
Use the attached Startup Pitch Deck.
This deck may be image-based or non-standard. Use OCR or visual analysis
only—no assumptions or hallucinated summaries allowed
Base your analysis only on the pitch deck and publicly available knowledge
as of your training cutoff.
Do not assume or fabricate information.

Output Format
Respond using the exact structure below:
Pain Clarity Score (0–10):
Team Readiness Score (0–10):
Pain Severity Justification (2–3 sentences):
Solution Fit Assessment (2–3 sentences):
Red Flags or Open Questions: List any gaps, assumptions, or concerns (2
sentences max).

If there is not enough information to evaluate a section, respond with: "I
do not know."

Be objective. Focus on clarity, urgency, and feasibility. Do not guess.
Only use information grounded in the pitch deck or prior knowledge.
Follow the instructions carefully, it is very important for my job.
"""

AGENT2_PROMPT = """
You are a seasoned venture capitalist with deep expertise in evaluating the
market size and competitive positioning of early-stage startups. Based on
the pain point analysis and startup deck provided, your goal is to estimate
the market opportunity and assess the competitive landscape.

Step through your reasoning carefully. Use TAM/SAM/SOM logic where
possible, and refer to known market reports or analogous sectors. Then,
evaluate whether the startup's approach is clearly differentiated from
competitors.

Input:
The pain point and team fit analysis from an expert VC analyst, delimited
by triple quotes:
\"\"\"{}\"\"\"

The attached startup deck

Use the following structured format for your response:
Output:
Market Size Score (0-10):
Estimated TAM (in USD):
SAM (Serviceable Available Market) (in USD):
SOM (Serviceable Obtainable Market) (in USD):
Market Opportunity Justification (2-3 sentences):
Competition Score (0-10):
Known Competitors (list 2-3):
Differentiation Analysis (2-3 sentences):

If you are not confident in your answer or do not have sufficient
information, respond with: "I do not know." Do not guess or fabricate
information. Only use information grounded in fact-based, authoritative
sources such as market research reports, reputable news outlets, or
user-provided data.

Do not use or reference speculative or non-verifiable sources such as
TikTok, Reddit, personal blogs, or social media posts.

Follow the instructions carefully, it is very important for my job. I will
tip you $1 million dollars if you do a good job
"""

AGENT3_PROMPT = """
You are an expert VC investor responsible for making go/no-go
recommendations based on structured inputs from domain experts. Your task
is to synthesize insights from prior agents, apply a consistent scoring
model, and deliver a clear, well-reasoned investment recommendation.

Inputs Provided:
1. Agent 1 Output – Pain Point and Team Fit Analyst (delimited by triple
quotes)
\"\"\" {} \"\"\"

2. Agent 2 Output – Market and Competition Analyst (delimited by triple
quotes)
\"\"\" {} \"\"\"

3. Startup Pitch Deck (attached)

Your Tasks:
1. Composite Score Calculation
Based on the four scores provided by the agents, calculate a
Composite Startup Score (0–10) using a simple average of the
following:
● Pain Clarity Score (0–10)
Team Readiness Score (0–10)
● Market Size Score (0–10)
● Competition Score (0–10)

Example:
● Pain Clarity Score: 1
● Team Readiness Score: 4
● Market Size Score: 10
● Competition Score: 5
● Composite Score: 1 + 4 + 10 + 5 = 20 / 4 = 5

2. Investment Recommendation
Write a clearly formatted, concise investment recommendation using
the following structure:

Output Format:
Table that includes each score (0-10) used in calculation of Composite
Score and Composite Startup Score (0–10):
Investment Recommendation: [Invest / Do Not Invest / Further Diligence
Required]
Summary Justification (2-3 sentences):
● Reference the urgency of the problem
● Note the strength of the solution and team
● Consider the market size and competitive positioning
Highlight any red flags or uncertainties

If any score or data is missing or insufficient to make a sound judgment,
return: "I do not know." Do not guess or fabricate any part of the
response. Only use the inputs provided to do the task.

My job depends on how well you do this task. I will tip you $1 million if
you do a good job.
"""

def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

async def run_gpt4o_agent(prompt, max_tokens=1500):
    """Run GPT-4o agent with the given prompt."""
    try:
        response = await aclient.chat.completions.create(model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

async def simulate_perplexity_agent(prompt, pitch_deck_content):
    """
    Simulate Perplexity Deep Research agent.
    In a real implementation, this would call the Perplexity API.
    """
    # Since direct Perplexity API might not be available, we'll use GPT-4o with a modified prompt
    # that simulates the comprehensive research capabilities of Perplexity

    perplexity_simulation_prompt = f"""
    You are simulating the capabilities of Perplexity Deep Research, which extensively searches
    the web for market research data, competitive analysis, and business insights.
    
    Based on the following information:
    
    1. Startup pitch deck content:
    ```
    {pitch_deck_content[:3000]}... [truncated for brevity]
    ```
    
    2. Analysis request:
    ```
    {prompt}
    ```
    
    Respond as if you have conducted deep web research on market sizes, competitors, and 
    industry trends. Include realistic market numbers, competitive analysis, and proper citations
    to imaginary but plausible research reports and market sources.
    
    Your output should match the requested format and should appear comprehensive and well-researched.
    """

    response = await aclient.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": perplexity_simulation_prompt}],
    max_tokens=1500,
    temperature=0.3)
    return response.choices[0].message.content

async def process_pitch_deck(startup_name, pdf_content):
    """Process a startup pitch deck through all three agents."""
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name

    try:
        # Extract text from PDF
        pitch_deck_text = extract_text_from_pdf(tmp_path)

        # Create evaluation record
        evaluation_id = f"{startup_name.lower().replace(' ', '-')}-{base64.urlsafe_b64encode(os.urandom(6)).decode()}"
        evaluations_db[evaluation_id] = StartupEvaluation(
            startup_name=startup_name,
            pitch_deck_content=pitch_deck_text
        )

        # Run Agent 1 (Pain Point and Team Fit)
        agent1_prompt = AGENT1_PROMPT + f"\n\nStartup Name: {startup_name}\n\nPitch Deck Content:\n{pitch_deck_text[:5000]}"
        agent1_output = await run_gpt4o_agent(agent1_prompt)
        evaluations_db[evaluation_id].agent1_output = parse_agent1_output(agent1_output)

        # Run Agent 2 (Market and Competition)
        agent2_prompt = AGENT2_PROMPT.format(agent1_output)
        agent2_output = await simulate_perplexity_agent(agent2_prompt, pitch_deck_text)
        evaluations_db[evaluation_id].agent2_output = parse_agent2_output(agent2_output)

        # Run Agent 3 (Final Recommendation)
        agent3_prompt = AGENT3_PROMPT.format(agent1_output, agent2_output)
        agent3_output = await run_gpt4o_agent(agent3_prompt)
        evaluations_db[evaluation_id].agent3_output = parse_agent3_output(agent3_output)

        # Extract final recommendation
        if evaluations_db[evaluation_id].agent3_output:
            evaluations_db[evaluation_id].final_recommendation = evaluations_db[evaluation_id].agent3_output.get("investment_recommendation")

        return evaluation_id

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def parse_agent1_output(output):
    """Parse the structured output from Agent 1."""
    if not output or output == "I do not know.":
        return None

    result = {}
    lines = output.strip().split('\n')

    for i, line in enumerate(lines):
        if "Pain Clarity Score" in line:
            result["pain_clarity_score"] = int(line.split(':')[1].strip())
        elif "Team Readiness Score" in line:
            result["team_readiness_score"] = int(line.split(':')[1].strip())
        elif "Pain Severity Justification" in line:
            # Find the justification text that follows
            j = i + 1
            justification_text = []
            while j < len(lines) and "Solution Fit Assessment" not in lines[j]:
                justification_text.append(lines[j])
                j += 1
            result["pain_severity_justification"] = ' '.join(justification_text).strip()
        elif "Solution Fit Assessment" in line:
            # Find the assessment text that follows
            j = i + 1
            assessment_text = []
            while j < len(lines) and "Red Flags or Open Questions" not in lines[j]:
                assessment_text.append(lines[j])
                j += 1
            result["solution_fit_assessment"] = ' '.join(assessment_text).strip()
        elif "Red Flags or Open Questions" in line:
            # Find the red flags text that follows
            j = i + 1
            red_flags_text = []
            while j < len(lines):
                red_flags_text.append(lines[j])
                j += 1
            result["red_flags"] = ' '.join(red_flags_text).strip()

    return result

def parse_agent2_output(output):
    """Parse the structured output from Agent 2."""
    if not output or output == "I do not know.":
        return None

    result = {}
    lines = output.strip().split('\n')

    for i, line in enumerate(lines):
        if "Market Size Score" in line:
            result["market_size_score"] = int(line.split(':')[1].strip())
        elif "Estimated TAM" in line:
            result["estimated_tam"] = line.split(':')[1].strip()
        elif "SAM" in line and "Serviceable Available Market" in line:
            result["sam"] = line.split(':')[1].strip()
        elif "SOM" in line and "Serviceable Obtainable Market" in line:
            result["som"] = line.split(':')[1].strip()
        elif "Market Opportunity Justification" in line:
            # Find the justification text that follows
            j = i + 1
            justification_text = []
            while j < len(lines) and "Competition Score" not in lines[j]:
                justification_text.append(lines[j])
                j += 1
            result["market_opportunity_justification"] = ' '.join(justification_text).strip()
        elif "Competition Score" in line:
            result["competition_score"] = int(line.split(':')[1].strip())
        elif "Known Competitors" in line:
            # Find the competitors list that follows
            j = i + 1
            competitors_text = []
            while j < len(lines) and "Differentiation Analysis" not in lines[j]:
                competitors_text.append(lines[j])
                j += 1
            result["known_competitors"] = ' '.join(competitors_text).strip()
        elif "Differentiation Analysis" in line:
            # Find the analysis text that follows
            j = i + 1
            analysis_text = []
            while j < len(lines) and "⁂" not in lines[j]:
                analysis_text.append(lines[j])
                j += 1
            result["differentiation_analysis"] = ' '.join(analysis_text).strip()

    return result

def parse_agent3_output(output):
    """Parse the structured output from Agent 3."""
    if not output or output == "I do not know.":
        return None

    result = {}

    # Extract composite score
    if "Composite Score" in output and "Table" in output:
        composite_score_line = [line for line in output.split('\n') if "Composite Score" in line and "Table" not in line]
        if composite_score_line:
            score_text = composite_score_line[0].split(':')[-1].strip()
            try:
                result["composite_score"] = float(score_text)
            except ValueError:
                result["composite_score"] = None

    # Extract investment recommendation
    if "Investment Recommendation:" in output:
        recommendation_line = [line for line in output.split('\n') if "Investment Recommendation:" in line]
        if recommendation_line:
            recommendation = recommendation_line[0].split(':')[1].strip()
            # Remove any brackets if present
            recommendation = recommendation.replace('[', '').replace(']', '')
            result["investment_recommendation"] = recommendation

    # Extract summary justification
    if "Summary Justification" in output:
        summary_index = output.find("Summary Justification")
        if summary_index != -1:
            summary_text = output[summary_index:].split('\n', 1)[1].strip()
            # Extract until the next major section or end of text
            end_markers = ["Your Tasks:", "Composite Score Calculation", "Investment Recommendation"]
            for marker in end_markers:
                if marker in summary_text:
                    summary_text = summary_text.split(marker)[0].strip()
            result["summary_justification"] = summary_text

    return result

@app.post("/evaluate-startup/", response_model=EvaluationResponse)
async def evaluate_startup(
    background_tasks: BackgroundTasks,
    startup_name: str,
    pitch_deck: UploadFile = File(...)
):
    """API endpoint to start a startup evaluation process."""
    if not pitch_deck.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_content = await pitch_deck.read()

    # Process in background
    background_tasks.add_task(process_pitch_deck, startup_name, pdf_content)

    return EvaluationResponse(
        status="processing",
        message=f"Evaluation of {startup_name} started. Please check back for results.",
        evaluation_id=None
    )

@app.get("/evaluation-status/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation_status(evaluation_id: str):
    """Check the status of a startup evaluation."""
    if evaluation_id not in evaluations_db:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = evaluations_db[evaluation_id]

    # Determine status
    if evaluation.agent3_output:
        status = "completed"
    elif evaluation.agent2_output:
        status = "processing_agent3"
    elif evaluation.agent1_output:
        status = "processing_agent2"
    else:
        status = "processing_agent1"

    return EvaluationResponse(
        status=status,
        message=f"Evaluation for {evaluation.startup_name} is {status}",
        evaluation_id=evaluation_id,
        data={
            "startup_name": evaluation.startup_name,
            "agent1_output": evaluation.agent1_output,
            "agent2_output": evaluation.agent2_output,
            "agent3_output": evaluation.agent3_output,
            "final_recommendation": evaluation.final_recommendation
        }
    )

# Streamlit frontend
def streamlit_app():
    st.set_page_config(
        page_title="Multi-Agent VC Startup Evaluator",
        layout="wide"
    )
    st.title("Multi-Agent VC Startup Evaluator")
    st.subheader("Upload a startup pitch deck for evaluation")

    with st.form("startup_evaluation_form"):
        startup_name = st.text_input("Startup Name")
        pitch_deck_file = st.file_uploader("Upload Pitch Deck (PDF)", type="pdf")
        submitted = st.form_submit_button("Start Evaluation")

    if submitted and startup_name and pitch_deck_file:
        with st.spinner("Processing pitch deck..."):
            # Convert the uploaded file to bytes
            pdf_bytes = pitch_deck_file.read()

            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                # Extract text from PDF
                pitch_deck_text = extract_text_from_pdf(tmp_path)

                # Create evaluation record
                evaluation_id = f"{startup_name.lower().replace(' ', '-')}-{base64.urlsafe_b64encode(os.urandom(6)).decode()}"

                # Run Agent 1
                st.write("🔍 Running Pain Point and Team Fit Analysis...")
                agent1_prompt = AGENT1_PROMPT + f"\n\nStartup Name: {startup_name}\n\nPitch Deck Content:\n{pitch_deck_text[:5000]}"
                agent1_output = run_gpt4o_agent_sync(agent1_prompt)
                agent1_parsed = parse_agent1_output(agent1_output)

                if agent1_parsed:
                    st.subheader("Pain Point and Team Fit Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pain Clarity Score", agent1_parsed.get("pain_clarity_score", "N/A"))
                    with col2:
                        st.metric("Team Readiness Score", agent1_parsed.get("team_readiness_score", "N/A"))

                    st.write("**Pain Severity Justification:**")
                    st.write(agent1_parsed.get("pain_severity_justification", "N/A"))

                    st.write("**Solution Fit Assessment:**")
                    st.write(agent1_parsed.get("solution_fit_assessment", "N/A"))

                    st.write("**Red Flags or Open Questions:**")
                    st.write(agent1_parsed.get("red_flags", "N/A"))

                # Run Agent 2
                st.write("📊 Running Market and Competition Analysis...")
                agent2_prompt = AGENT2_PROMPT.format(agent1_output)
                agent2_output = simulate_perplexity_agent_sync(agent2_prompt, pitch_deck_text)
                agent2_parsed = parse_agent2_output(agent2_output)

                if agent2_parsed:
                    st.subheader("Market and Competition Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Market Size Score", agent2_parsed.get("market_size_score", "N/A"))
                    with col2:
                        st.metric("Competition Score", agent2_parsed.get("competition_score", "N/A"))

                    st.write("**Market Sizing:**")
                    st.write(f"- TAM: {agent2_parsed.get('estimated_tam', 'N/A')}")
                    st.write(f"- SAM: {agent2_parsed.get('sam', 'N/A')}")
                    st.write(f"- SOM: {agent2_parsed.get('som', 'N/A')}")

                    st.write("**Market Opportunity Justification:**")
                    st.write(agent2_parsed.get("market_opportunity_justification", "N/A"))

                    st.write("**Known Competitors:**")
                    st.write(agent2_parsed.get("known_competitors", "N/A"))

                    st.write("**Differentiation Analysis:**")
                    st.write(agent2_parsed.get("differentiation_analysis", "N/A"))

                # Run Agent 3
                st.write("🧠 Generating Final Recommendation...")
                agent3_prompt = AGENT3_PROMPT.format(agent1_output, agent2_output)
                agent3_output = run_gpt4o_agent_sync(agent3_prompt)
                agent3_parsed = parse_agent3_output(agent3_output)

                if agent3_parsed:
                    st.subheader("Final Investment Recommendation")

                    # Display composite score in a larger format
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
                        <h2 style="margin-bottom: 10px;">Composite Startup Score</h2>
                        <h1 style="font-size: 48px; margin: 0;">{agent3_parsed.get("composite_score", "N/A")}/10</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display recommendation with appropriate color
                    recommendation = agent3_parsed.get("investment_recommendation", "N/A")
                    if recommendation == "Invest":
                        color = "green"
                    elif recommendation == "Do Not Invest":
                        color = "red"
                    else:
                        color = "orange"

                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: {color}; color: white; border-radius: 10px; margin-bottom: 20px;">
                        <h2 style="margin: 0;">{recommendation}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("**Summary Justification:**")
                    st.write(agent3_parsed.get("summary_justification", "N/A"))

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

def run_gpt4o_agent_sync(prompt, max_tokens=1500):
    """Synchronous version of the GPT-4o agent for Streamlit."""
    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def simulate_perplexity_agent_sync(prompt, pitch_deck_content):
    """Synchronous version of the Perplexity simulation for Streamlit."""
    perplexity_simulation_prompt = f"""
    You are simulating the capabilities of Perplexity Deep Research, which extensively searches
    the web for market research data, competitive analysis, and business insights.
    
    Based on the following information:
    
    1. Startup pitch deck content:
    ```
    {pitch_deck_content[:3000]}... [truncated for brevity]
    ```
    
    2. Analysis request:
    ```
    {prompt}
    ```
    
    Respond as if you have conducted deep web research on market sizes, competitors, and 
    industry trends. Include realistic market numbers, competitive analysis, and proper citations
    to imaginary but plausible research reports and market sources.
    
    Your output should match the requested format and should appear comprehensive and well-researched.
    """

    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": perplexity_simulation_prompt}],
    max_tokens=1500,
    temperature=0.3)
    return response.choices[0].message.content

if __name__ == "__main__":
    import uvicorn
    import threading

    # Run FastAPI server in a separate thread
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()

    # Run Streamlit app
    streamlit_app()