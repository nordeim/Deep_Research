import os
import re
import requests
import openai  # For LLM question generation and summarization
from bs4 import BeautifulSoup
from zenrows import ZenRowsClient
from urllib.parse import quote_plus

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Get from environment variable
ZENROWS_API_KEY = os.environ.get("ZENROWS_API_KEY") # Get from environment variable

# --- Constants ---
MAX_SUBQUESTIONS = 5  # Limit the number of sub-questions per level
MAX_RECURSION_DEPTH = 2  # Limit the depth of recursive questioning
MAX_SUMMARY_LENGTH = 300 # words
ZENROWS_CONCURRENCY = 5 # Number of parallel requests to ZenRows. Adjust based on your plan.

# --- Helper Functions ---
def generate_subquestions(query, existing_results=None, depth=0):
    """Generates sub-questions using OpenAI's API."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

    openai.api_key = OPENAI_API_KEY

    prompt = f"Generate a list of up to {MAX_SUBQUESTIONS} detailed sub-questions for in-depth research on the topic: '{query}'. "
    if existing_results:
        prompt += f"Consider these existing findings:\n{existing_results}\nGenerate NEW sub-questions that explore unanswered aspects or delve deeper."
    prompt += "Return the sub-questions as a numbered list."

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or a more capable model if you have access
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200 * MAX_SUBQUESTIONS,  # Adjust as needed
        )
        # Extract sub-questions from the response, handling different formats
        response_text = response.choices[0].message.content
        subquestions = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', response_text) # Match numbered list items
        if not subquestions:
            subquestions = response_text.split('\n') # fallback to newline separation
        # Clean up extracted subquestions
        subquestions = [q.strip() for q in subquestions if q.strip()]

        print(f"  Generated sub-questions (depth {depth}): {subquestions}")
        return subquestions[:MAX_SUBQUESTIONS] # Ensure we don't exceed the limit.

    except openai.OpenAIError as e:
        print(f"Error calling OpenAI API: {e}")
        return []


def scrape_page(url, client):
    """Scrapes a single page using ZenRows."""
    try:
        params = {
            "url": url,
            "js_render": "true",  # Enable JavaScript rendering
            "antibot": "true",  # Enable antibot bypass
            "premium_proxy": "true",  # Use premium proxies for better reliability
            # "proxy_country": "us",  # Optional: Specify proxy country
        }
        response = client.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""

def batch_scrape_pages(urls, client):
    """Scrapes multiple pages concurrently using ZenRows."""
    responses_content = []
    # Use ZenRowsClient context manager for multiple requests
    for url in urls:
        try:
            params = {
                "url": url,
                "js_render": "true",  # Enable JavaScript rendering
                "antibot": "true",  # Enable antibot bypass
                "premium_proxy": "true",  # Use premium proxies
            }
            response = client.get(url, params=params)
            response.raise_for_status()
            responses_content.append(response.text)
        except Exception as e:
            print(f"An error occurred: {e}")
    return responses_content

def summarize_content(text, query):
    """Summarizes a text using OpenAI's API, focusing on the given query."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

    openai.api_key = OPENAI_API_KEY

    prompt = f"Summarize the following text in at most {MAX_SUMMARY_LENGTH} words, focusing on how it relates to the question: '{query}'.\n\nText:\n{text}"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or a more capable model
            messages=[
                {"role": "system", "content": "You are a concise and accurate summarization assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_SUMMARY_LENGTH * 2,  # Allow for longer summaries
        )
        summary = response.choices[0].message.content.strip()
        print(f"  Summary generated (length: {len(summary.split())} words)")
        return summary
    except openai.OpenAIError as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error: Could not generate summary."

def extract_text_from_html(html_content):
    """Extracts clean text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style tags
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text and remove extra whitespace
    text = soup.get_text(separator=" ", strip=True)
    return text

def synthesize_results(summaries, original_query):
    """Synthesizes multiple summaries into a final answer using OpenAI."""
    if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

    openai.api_key = OPENAI_API_KEY

    prompt = f"Synthesize the following summaries into a comprehensive answer to the original research question: '{original_query}'.  Provide citations to the source numbers.\n\nSummaries (numbered):\n"
    for i, summary in enumerate(summaries):
        prompt += f"{i+1}. {summary}\n"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # or more capable model
            messages=[
                {"role": "system", "content": "You are a research assistant that synthesizes information from multiple sources."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500  # Adjust as needed
        )
        synthesis = response.choices[0].message.content.strip()
        print(f"  Final synthesis generated.")
        return synthesis
    except openai.OpenAIError as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error: Could not synthesize results."


def deep_research(query, depth=0):
    """Performs the deep research, recursively if needed."""

    print(f"Starting research on: '{query}' (depth {depth})")

    if depth > MAX_RECURSION_DEPTH:
        print("Max recursion depth reached.")
        return []

    # 1. Generate Sub-questions
    if depth == 0: # first level
      subquestions = generate_subquestions(query)
    else: # recursive level
      # Create a summary of findings from the *previous* level to inform subquestion generation
      combined_summary = " ".join(results)
      subquestions = generate_subquestions(query, combined_summary, depth)

    if not subquestions:
        print("No sub-questions generated.")
        return []

    all_summaries = []
    all_urls = []

    # Initialize ZenRows client outside the loop
    client = ZenRowsClient(ZENROWS_API_KEY)

    # 2. Search and Scrape for each sub-question
    for i, subq in enumerate(subquestions):
        print(f"  Searching for sub-question {i+1}: {subq}")
        # Use a search engine (here, using a simple Google search via ZenRows)
        search_url = f"https://www.google.com/search?q={quote_plus(subq)}"
        search_results_html = scrape_page(search_url, client)
        if not search_results_html:
            continue

        # Extract links from the search results (basic example, needs refinement)
        soup = BeautifulSoup(search_results_html, "lxml")
        result_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('/url?q=')]
        # Clean up and filter result links
        cleaned_links = []
        for link in result_links:
            # Extract the actual URL
            match = re.search(r'/url\?q=(.+)&', link)
            if match:
                cleaned_url = match.group(1)
                # Basic filtering: Skip some common non-result pages
                if not any(skip in cleaned_url for skip in ['google.com', 'youtube.com', 'wikipedia.org']):  # Customize as needed
                    cleaned_links.append(cleaned_url)
        page_urls = cleaned_links[:3]  # Limit to top 3 results per sub-question, for brevity
        all_urls.extend(page_urls)
        print(f"    Found {len(page_urls)} relevant pages.")

        # 3. Scrape and Summarize each page
        scraped_pages = batch_scrape_pages(page_urls, client)
        for page_content in scraped_pages:
            if page_content:
                clean_text = extract_text_from_html(page_content)
                summary = summarize_content(clean_text, subq)
                all_summaries.append(summary)

    # Close ZenRows client
    # client.close()  # ZenRowsClient doesn't require explicit closing

    # 4. Synthesize Summaries (only at the top level)
    if depth == 0:
        final_answer = synthesize_results(all_summaries, query)
        print("\n--- Final Research Report ---")
        print(final_answer)
        print("\n--- Sources ---")
        for i, url in enumerate(all_urls):
            print(f"{i+1}. {url}")

    # 5. Recursive Call (if needed and within depth limit)
    if depth < MAX_RECURSION_DEPTH:
      print(f"Recursive call at depth {depth + 1}")
      deep_research(query, depth + 1)  # Recursive call

    return all_summaries

# --- Main Execution ---
if __name__ == "__main__":

    # --- Input Validation ---
    while True:
        research_topic = input("Enter your research topic: ")
        if research_topic.strip():
            break
        print("Please enter a valid research topic.")

    if not ZENROWS_API_KEY:
        print("WARNING: ZenRows API key not found.  Web scraping will likely fail.")
        print("Set the ZENROWS_API_KEY environment variable.")
    if not OPENAI_API_KEY:
        print("WARNING: OpenAI API key not found. Question generation and summarization will be limited.")
        print("Set the OPENAI_API_KEY environment variable.")
    # --- Perform Deep Research ---
    try:
        results = deep_research(research_topic)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
      
