#!/usr/bin/env python3
import os
import re
import requests
import openai
from bs4 import BeautifulSoup
from zenrows import ZenRowsClient
from urllib.parse import quote_plus
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ZENROWS_API_KEY = os.environ.get("ZENROWS_API_KEY")

# --- Constants ---
MAX_SUBQUESTIONS = 5            # Maximum sub-questions per level
MAX_RECURSION_DEPTH = 2         # Maximum recursion depth for deeper inquiry
MAX_SUMMARY_LENGTH = 300        # Maximum word count for each summary
ZENROWS_CONCURRENCY = 5         # Number of concurrent requests

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Helper Functions ---

def generate_subquestions(query: str, existing_results: str = None, depth: int = 0) -> list:
    """
    Generate up to MAX_SUBQUESTIONS detailed sub-questions for deeper research.
    If previous findings exist, new sub-questions will build on them.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = OPENAI_API_KEY
    prompt = (f"Generate a list of up to {MAX_SUBQUESTIONS} detailed sub-questions for in-depth research on the topic: "
              f"'{query}'. ")
    if existing_results:
        prompt += f"Consider these existing findings:\n{existing_results}\nGenerate NEW sub-questions that explore unanswered aspects or delve deeper."
    prompt += " Return the sub-questions as a numbered list."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200 * MAX_SUBQUESTIONS,
        )
        response_text = response.choices[0].message.content
        subquestions = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', response_text)
        if not subquestions:
            subquestions = [q.strip() for q in response_text.split('\n') if q.strip()]
        logging.info(f"Generated sub-questions (depth {depth}): {subquestions}")
        return subquestions[:MAX_SUBQUESTIONS]
    except openai.OpenAIError as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return []

def scrape_page(url: str, client: ZenRowsClient) -> str:
    """
    Scrape a single page using ZenRows with JavaScript rendering and antibot bypass.
    """
    try:
        params = {
            "url": url,
            "js_render": "true",
            "antibot": "true",
            "premium_proxy": "true",
        }
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return ""

def batch_scrape_pages(urls: list, client: ZenRowsClient) -> list:
    """
    Scrape multiple pages concurrently using a ThreadPoolExecutor.
    """
    responses_content = []
    with ThreadPoolExecutor(max_workers=ZENROWS_CONCURRENCY) as executor:
        future_to_url = {executor.submit(scrape_page, url, client): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                responses_content.append(content)
            except Exception as exc:
                logging.error(f"Error scraping {url}: {exc}")
    return responses_content

def summarize_content(text: str, query: str) -> str:
    """
    Summarize the given text (cleaned from HTML) focusing on its relation to the query.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = OPENAI_API_KEY
    prompt = (f"Summarize the following text in at most {MAX_SUMMARY_LENGTH} words, focusing on how it relates "
              f"to the question: '{query}'.\n\nText:\n{text}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise and accurate summarization assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_SUMMARY_LENGTH * 2,
        )
        summary = response.choices[0].message.content.strip()
        logging.info(f"Generated summary (length: {len(summary.split())} words)")
        return summary
    except openai.OpenAIError as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return "Error: Could not generate summary."

def extract_text_from_html(html_content: str) -> str:
    """
    Extract and clean text from HTML using BeautifulSoup.
    """
    soup = BeautifulSoup(html_content, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text

def synthesize_results(summaries: list, original_query: str) -> str:
    """
    Combine multiple summaries into one comprehensive answer with citations.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = OPENAI_API_KEY
    prompt = f"Synthesize the following summaries into a comprehensive answer to the research question: '{original_query}'. " \
             "Include citations referring to the summary numbers.\n\n"
    for i, summary in enumerate(summaries):
        prompt += f"{i+1}. {summary}\n"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant that synthesizes information from multiple sources."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )
        synthesis = response.choices[0].message.content.strip()
        logging.info("Final synthesis generated.")
        return synthesis
    except openai.OpenAIError as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return "Error: Could not synthesize results."

def extract_search_result_links(html_content: str) -> list:
    """
    Extract result links from a Google search results page.
    """
    soup = BeautifulSoup(html_content, "lxml")
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/url?q='):
            match = re.search(r'/url\?q=([^&]+)&', href)
            if match:
                url = match.group(1)
                if not any(skip in url for skip in ['google.com', 'youtube.com', 'wikipedia.org']):
                    links.append(url)
    return links

def deep_research(query: str, depth: int = 0, prev_summaries: list = None) -> (list, list):
    """
    Perform deep research recursively. At each level, generate sub-questions,
    perform web searches, scrape and summarize pages, and synthesize findings.
    """
    logging.info(f"Starting research on: '{query}' (depth {depth})")
    if depth > MAX_RECURSION_DEPTH:
        logging.info("Max recursion depth reached.")
        return (prev_summaries if prev_summaries else [], [])
    
    # Generate sub-questions based on previous summaries (if any)
    existing_results = " ".join(prev_summaries) if prev_summaries else None
    subquestions = generate_subquestions(query, existing_results, depth)
    if not subquestions:
        logging.info("No sub-questions generated.")
        return (prev_summaries if prev_summaries else [], [])
    
    all_summaries = list(prev_summaries) if prev_summaries else []
    all_urls = []
    client = ZenRowsClient(ZENROWS_API_KEY) if ZENROWS_API_KEY else None
    if client is None:
        logging.warning("ZenRows API key not found. Web scraping may fail.")
    
    # Process each sub-question
    for idx, subq in enumerate(subquestions):
        logging.info(f"Searching for sub-question {idx+1}: {subq}")
        search_url = f"https://www.google.com/search?q={quote_plus(subq)}"
        search_results_html = scrape_page(search_url, client) if client else ""
        if not search_results_html:
            continue
        result_links = extract_search_result_links(search_results_html)
        page_urls = result_links[:3]  # Limit to top 3 results per sub-question
        all_urls.extend(page_urls)
        logging.info(f"Found {len(page_urls)} relevant pages for sub-question {idx+1}.")
        scraped_pages = batch_scrape_pages(page_urls, client) if client else []
        for page_content in scraped_pages:
            if page_content:
                clean_text = extract_text_from_html(page_content)
                summary = summarize_content(clean_text, subq)
                all_summaries.append(summary)
    
    # At the top level, synthesize results and print the final report
    if depth == 0:
        final_answer = synthesize_results(all_summaries, query)
        logging.info("\n--- Final Research Report ---\n" + final_answer)
        logging.info("\n--- Sources ---")
        for i, url in enumerate(all_urls):
            logging.info(f"{i+1}. {url}")
    # Recursively deepen the research if within allowed depth
    if depth < MAX_RECURSION_DEPTH:
        deeper_summaries, deeper_urls = deep_research(query, depth + 1, all_summaries)
        all_summaries.extend(deeper_summaries)
        all_urls.extend(deeper_urls)
    
    return (all_summaries, all_urls)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        research_topic = ""
        while not research_topic.strip():
            research_topic = input("Enter your research topic: ").strip()
            if not research_topic:
                print("Please enter a valid research topic.")
        if not ZENROWS_API_KEY:
            logging.warning("ZenRows API key not found. Set the ZENROWS_API_KEY environment variable.")
        if not OPENAI_API_KEY:
            logging.warning("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        summaries, urls = deep_research(research_topic)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
