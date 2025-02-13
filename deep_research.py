#!/usr/bin/env python3
"""
Deeper Seeker v2.0 – Enhanced Deep Research Tool
Author: Your Name
Date: 2025-02-13
"""

import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional

import aiohttp
import pyfiglet
from colorama import Fore, Style, init
from openai import OpenAI

# Initialize colorama and logging
init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optionally load environment variables from a .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.info("python-dotenv not installed; proceeding without it.")

# Configuration
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXA_BASE_URL = "https://api.exa.ai"
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "9"))

if not EXA_API_KEY or not OPENAI_API_KEY:
    logging.error("Missing required API keys. Please set EXA_API_KEY and OPENAI_API_KEY in your environment.")
    exit(1)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# System prompt for OpenAI calls
SYS_PROMPT = """
You are a highly capable AI research expert operating like a junior analyst in venture capital, consulting, or investment banking. Your role involves comprehensive market research, competitor analysis, and qualitative studies. You excel in understanding complex queries, reasoning through challenges, and strategic planning.

Key Capabilities:
- Internet Research: Access and evaluate credible sources.
- Data Synthesis: Transform diverse information into coherent, actionable insights.
- Iterative Refinement: Revise your plan and perform additional research as needed.
- Structured Output: Generate a structured JSON output to call the `exa_search` function.

Workflow:
1. Plan Creation: Analyze the user’s query and develop a structured research plan.
2. Generate Search Query: Formulate precise search queries (targeting data from 2024).
3. Structured Output for exa_search: Output a JSON structure for each search action.
4. Execute and Iterate: Call `exa_search`, review and incorporate results.
5. Final Reporting: Synthesize all gathered information into an in-depth report with inline citations.

Maintain clarity, accuracy, and efficiency.
"""

async def exa_search(query: str) -> dict:
    """Asynchronously execute search using Exa API with citations."""
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    payload = {"query": query, "stream": False, "text": True, "mode": "fast"}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{EXA_BASE_URL}/search", json=payload, headers=headers) as response:
            if response.status != 200:
                logging.error(f"Exa search failed with status {response.status}")
                return {}
            return await response.json()

def process_search_results(results: dict) -> str:
    """Process and format Exa search results."""
    if not results.get("results"):
        return "No relevant results found"
    summary_lines = []
    for result in results["results"]:
        title = result.get("title", "Untitled")
        url = result.get("url", "No URL provided")
        text_excerpt = result.get("text", "")[:200].strip()  # limit to first 200 chars
        highlights = result.get("highlights", [])
        summary_lines.append(f"### {title}")
        summary_lines.append(f"**URL:** {url}")
        summary_lines.append(f"**Content:** {text_excerpt}...")
        if highlights:
            summary_lines.append("**Highlights:**")
            for item in highlights:
                summary_lines.append(f"- {item}")
        summary_lines.append("")
    return "\n".join(summary_lines)

def generate_final_report(context: List[Dict[str, Any]]) -> str:
    """Generate final synthesized report using OpenAI."""
    research_history = "\n\n".join(
        f"## Research Step {i+1}\n**Query:** {step['query']}\n**Results:**\n{step['results']}"
        for i, step in enumerate(context)
    )
    messages = [
        {"role": "system", "content": SYS_PROMPT + "\nMaintain the inline citations from the research data in the final answer. Provide output in Markdown."},
        {"role": "user", "content": f"Synthesize a final, detailed report using the following research data:\n\n{research_history}"}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4-mini",
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content

def generate_research_step(user_query: str, context: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Generate next research step using OpenAI with structured output."""
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_query}]
    # Append previous steps from context
    for step in context:
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "action": "exa_search",
                "query": step["query"],
                "reasoning": step["reasoning"],
                "plan": step["plan"]
            })
        })
        messages.append({
            "role": "system",
            "content": f"Search Results from '{step['query']}':\n{step['results']}"
        })
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        step_output = json.loads(content)
        return step_output
    except Exception as e:
        logging.error(f"Error generating research step: {e}")
        return None

class ResearchAgent:
    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.context: List[Dict[str, Any]] = []
        self.max_iterations = max_iterations

    async def execute_research_plan(self, user_query: str) -> str:
        """Execute iterative research process."""
        logging.info("Initializing research plan...")
        for i in range(self.max_iterations):
            logging.info(f"--- Iteration {i+1} ---")
            research_step = generate_research_step(user_query, self.context)
            if not research_step or not research_step.get("query"):
                logging.error("No search query generated; terminating iteration.")
                break
            logging.info(f"Search Query: {research_step['query']}")
            # Execute asynchronous search call
            search_results = await exa_search(research_step["query"])
            processed_results = process_search_results(search_results)
            self.context.append({
                "query": research_step["query"],
                "reasoning": research_step.get("reasoning", ""),
                "plan": research_step.get("plan", ""),
                "results": processed_results
            })
            logging.info(f"Search Results (truncated): {processed_results[:500]}...")
            if "no relevant results" in processed_results.lower():
                logging.error("Terminating due to lack of relevant results.")
                break
            # Optional: pause between iterations to prevent rate limits
            time.sleep(1)
        logging.info("Generating final report...")
        return generate_final_report(self.context)

def display_banner() -> None:
    banner = pyfiglet.figlet_format("Deeper Seeker v2")
    print(Fore.CYAN + banner + Style.RESET_ALL)

async def main() -> None:
    display_banner()
    agent = ResearchAgent()
    user_query = input("Enter your research query: ").strip()
    if not user_query:
        logging.error("Empty query provided. Exiting.")
        return
    final_report = await agent.execute_research_plan(user_query)
    print(f"\n{Fore.GREEN}=== FINAL REPORT ==={Style.RESET_ALL}\n")
    print(final_report)
    print(f"\n{Fore.CYAN}=== RESEARCH CONTEXT ==={Style.RESET_ALL}\n")
    print(json.dumps(agent.context, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
