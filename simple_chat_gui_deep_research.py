import sys
import json
import yaml
import os
import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextBrowser, QTextEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QMenuBar, QMenu, QFileDialog, QMessageBox,
    QDialog, QFormLayout, QLabel, QSlider, QComboBox,
    QDialogButtonBox, QGridLayout,
    QToolButton, QLineEdit
)
from PyQt6.QtCore import Qt, QDateTime, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QTextCursor, QFont
from urllib.parse import urlparse, quote_plus
from openai import OpenAI
import html
import datetime
import re
from bs4 import BeautifulSoup
from zenrows import ZenRowsClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Constants ---
MAX_SUBQUESTIONS = 5
MAX_RECURSION_DEPTH = 2
MAX_SUMMARY_LENGTH = 300
ZENROWS_CONCURRENCY = 5

class EmojiPickerDialog(QDialog):
    def __init__(self, parent=None, input_box=None):
        super().__init__(parent)
        self.setWindowTitle("Emoji Picker")
        self.input_box = input_box
        self.emoji_buttons = []
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout(self)
        emojis = [
            "😀", "😂", "😊", "😛", "❤️", "🎉", "🤔", "🚀", "🌟", "💡",
            "😞", "😡", "😴", "☕", "🍕", "🐶", "🐱", "🐰", "🌸", "🌈",
            "👍", "🙏", "😊", "😅", "😇", "🤑", "😢", "😄", "🥳", "😘",
            "🤗", "😂", "🍻", "🤩", "👏", "👌", "✌️", "☝️", "👎", "👋",
            "💪", "🫶", "🥱", "🤗", "🙄", "🤡", "💩", "😭", "😤", "😡",
            "☹️", "😣", "😖", "😫", "😎", "🤓", "🧐", "🤪", "😜", "😝",
            "😍", "🥰", "😚", "😋", "😀", "😃", "😄", "😁", "😆", "😅",
            "😂", "🤣", "😝", "😇", "😍", "🤩", "😘", "😗", "😚", "😋"
        ]
        row, col = 0, 0
        for emoji in emojis:
            button = QToolButton()
            button.setText(emoji)
            button.setFont(QFont("Arial", 14))
            button.clicked.connect(lambda checked, emoji=emoji: self.emoji_selected(emoji))
            layout.addWidget(button, row, col)
            self.emoji_buttons.append(button)
            col += 1
            if col > 8:
                col = 0
                row += 1

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.reject)
        layout.addWidget(close_button, row + 1, 0, 1, col + 1)

        self.setLayout(layout)

    def emoji_selected(self, emoji):
        if self.input_box:
            current_text = self.input_box.toPlainText()
            self.input_box.setText(current_text + emoji)
        # Removed self.accept() to prevent the dialog from closing


class DeepResearchThread(QThread):
    """
    Separate thread for performing deep research to prevent GUI freezing.
    """
    # Signals for communicating with the main thread
    research_started = pyqtSignal()  # Signal that research has started
    subquestion_generated = pyqtSignal(str, int)  # Signal sub-question & depth
    summary_ready = pyqtSignal(str)   # Signal when a summary is available
    synthesis_ready = pyqtSignal(str) # Signal when final synthesis is ready
    research_complete = pyqtSignal()  # Signal when research is finished
    research_error = pyqtSignal(str)  # Signal when research failed
    sources_ready = pyqtSignal(list) # Signal when the list of sources/urls is ready.

    def __init__(self, query, config):
        super().__init__()
        self.query = query
        self.config = config
        self.zenrows_client = None
        if self.config.get('ZenRows_API_Key'):
            self.zenrows_client = ZenRowsClient(self.config['ZenRows_API_Key'])

        if config.get('API_Key') and config.get('API_Url'):
            self.openai_client = OpenAI(api_key=config['API_Key'], base_url=config['API_Url'])
        else:
            self.openai_client = None


    def run(self):
        """
        Entry point for the thread. Executes the deep research.
        """
        try:
            self.research_started.emit()  # Notify the GUI
            if not self.openai_client:
                self.research_error.emit("OpenAI client not initialized.  Check API configuration.")
                return
            self.deep_research(self.query) # Start deep research
        except Exception as e:
            self.research_error.emit(str(e))
        finally:
            self.research_complete.emit()  # Always signal completion

    def generate_subquestions(self, query: str, existing_results: str = None, depth: int = 0) -> list:
        """
        Generate up to MAX_SUBQUESTIONS detailed sub-questions.
        """
        prompt = (f"Generate a list of up to {MAX_SUBQUESTIONS} detailed sub-questions for in-depth research on the topic: "
                  f"'{query}'. ")
        if existing_results:
            prompt += f"Consider these existing findings:\n{existing_results}\nGenerate NEW sub-questions that explore unanswered aspects or delve deeper."
        prompt += " Return the sub-questions as a numbered list."

        try:
            completion = self.openai_client.chat.completions.create(
                model=self.config.get('Model', "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(self.config.get('Temperature', 0.5)),
                max_tokens=200 * MAX_SUBQUESTIONS,
            )
            response_text = completion.choices[0].message.content
            subquestions = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', response_text)
            if not subquestions:
                subquestions = [q.strip() for q in response_text.split('\n') if q.strip()]
            logging.info(f"Generated sub-questions (depth {depth}): {subquestions}")
            return subquestions[:MAX_SUBQUESTIONS]
        except Exception as e:
            self.research_error.emit(f"Error generating sub-questions: {e}")
            return []

    def scrape_page(self, url: str) -> str:
        """
        Scrape a single page using ZenRows.
        """
        if not self.zenrows_client:
            self.research_error.emit("ZenRows client is not initialized. Check ZenRows API key.")
            return ""
        try:
            params = {
                "url": url,
                "js_render": "true",
                "antibot": "true",
                "premium_proxy": "true",
            }
            response = self.zenrows_client.get(url, params=params)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error scraping {url}: {e}")
            return ""

    def batch_scrape_pages(self, urls: list) -> list:
        """
        Scrape multiple pages concurrently.
        """
        responses_content = []
        with ThreadPoolExecutor(max_workers=ZENROWS_CONCURRENCY) as executor:
            future_to_url = {executor.submit(self.scrape_page, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    responses_content.append(content)
                except Exception as exc:
                    logging.error(f"Error during batch scraping of {url}: {exc}")
        return responses_content

    def summarize_content(self, text: str, query: str) -> str:
        """
        Summarize text focusing on its relation to the query.
        """
        prompt = (f"Summarize the following text in at most {MAX_SUMMARY_LENGTH} words, focusing on how it relates "
                  f"to the question: '{query}'.\n\nText:\n{text}")
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.config.get('Model', "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a concise and accurate summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(self.config.get('Temperature', 0.5)),
                max_tokens=MAX_SUMMARY_LENGTH * 2,
            )
            summary = completion.choices[0].message.content.strip()
            logging.info(f"Generated summary (length: {len(summary.split())} words)")
            return summary
        except Exception as e:
            self.research_error.emit(f"Error summarizing content: {e}")
            return "Error: Could not generate summary."

    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract and clean text from HTML.
        """
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text

    def synthesize_results(self, summaries: list, original_query: str) -> str:
        """
        Combine summaries into a comprehensive answer with citations.
        """
        prompt = f"Synthesize the following summaries into a comprehensive answer to the research question: '{original_query}'. " \
                 "Include citations referring to the summary numbers.\n\n"
        for i, summary in enumerate(summaries):
            prompt += f"{i+1}. {summary}\n"
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.config.get('Model', "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a research assistant that synthesizes information from multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(self.config.get('Temperature', 0.5)),
                max_tokens=500,
            )
            synthesis = completion.choices[0].message.content.strip()
            logging.info("Final synthesis generated.")
            return synthesis
        except Exception as e:
            self.research_error.emit(f"Error synthesizing results: {e}")
            return "Error: Could not synthesize results."

    def extract_search_result_links(self, html_content: str) -> list:
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

    def deep_research(self, query: str, depth: int = 0, prev_summaries: list = None):
        """
        Perform deep research recursively (adapted for GUI).
        """
        logging.info(f"Starting research on: '{query}' (depth {depth})")
        if depth > MAX_RECURSION_DEPTH:
            logging.info("Max recursion depth reached.")
            return

        # Generate sub-questions, send signal for each
        existing_results = " ".join(prev_summaries) if prev_summaries else None
        subquestions = self.generate_subquestions(query, existing_results, depth)
        if not subquestions:
            logging.info("No sub-questions generated.")
            return

        all_summaries = list(prev_summaries) if prev_summaries else []
        all_urls = []

        for idx, subq in enumerate(subquestions):
            self.subquestion_generated.emit(subq, depth)  # Emit sub-question

            logging.info(f"Searching for sub-question {idx+1}: {subq}")
            search_url = f"https://www.google.com/search?q={quote_plus(subq)}"
            search_results_html = self.scrape_page(search_url)
            if not search_results_html:
                continue

            result_links = self.extract_search_result_links(search_results_html)
            page_urls = result_links[:3]
            all_urls.extend(page_urls)
            logging.info(f"Found {len(page_urls)} relevant pages for sub-question {idx+1}.")

            scraped_pages = self.batch_scrape_pages(page_urls)
            for page_content in scraped_pages:
                if page_content:
                    clean_text = self.extract_text_from_html(page_content)
                    summary = self.summarize_content(clean_text, subq)
                    all_summaries.append(summary)
                    self.summary_ready.emit(summary)  # Emit summary

        if depth == 0:  # At top level, synthesize and emit
            final_answer = self.synthesize_results(all_summaries, query)
            self.synthesis_ready.emit(final_answer)  # Emit synthesis
            self.sources_ready.emit(all_urls)

        # Recurse for deeper research
        if depth < MAX_RECURSION_DEPTH:
            self.deep_research(query, depth + 1, all_summaries)


class AIChatApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Modern AI Chat App")
        self.setGeometry(100, 100, 800, 600)

        self.chat_log = []
        self.config = self.load_config()
        self.session_log_file = None

        self.emoji_dialog = EmojiPickerDialog(self, None)
        self.attached_file_content = None
        self.attached_file_name = None
        self.init_ui()
        self.emoji_dialog.input_box = self.input_box

        self.openai_client = None
        if (self.config):
            self.init_openai_client()
        else:
            QMessageBox.warning(self, "API Configuration", "API configuration not found.  Please configure the API.")

        self.create_session_log()

        self.css_style = """
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background-color: #e5ddd5;
                        color: #000;
                        padding: 20px;
                    }
                    .message {
                        background-color: #dcf8c6;
                        border-radius: 10px;
                        padding: 10px;
                        margin: 5px 0;
                        max-width: 70%;
                        word-wrap: break-word;
                    }
                    .user-message {
                        background-color: lightgreen;
                        text-align: right;
                    }
                    .ai-message {
                        background-color: lightblue;
                        text-align: left;
                    }
                    .error-message {
                        background-color: lightcoral;
                        text-align: left;
                    }
                    .bold { font-weight: bold; }
                    .italic { font-style: italic; }
                    .underline { text-decoration: underline; }
                    .strikethrough { text-decoration: line-through; }
                    .emoji { font-size: 1.2em; }
                """

    def init_openai_client(self):
        if self.config and self.config.get('API_Key') and self.config.get('API_Url'):
            self.openai_client = OpenAI(
                api_key=self.config['API_Key'],
                base_url=self.config['API_Url']
            )
        else:
            self.openai_client = None

    def init_ui(self):
        # --- Menu Bar ---
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu("File")
        settings_menu = menu_bar.addMenu("Settings")

        export_action = QAction("Export Chat History", self)
        export_action.triggered.connect(self.export_chat_history)
        file_menu.addAction(export_action)

        import_action = QAction("Import Chat History", self)
        import_action.triggered.connect(self.import_chat_history)
        file_menu.addAction(import_action)

        config_action = QAction("API Configuration", self)
        config_action.triggered.connect(self.show_config_dialog)
        settings_menu.addAction(config_action)

        # --- Chat Display Area ---
        self.chat_browser = QTextBrowser(self)
        self.chat_browser.setOpenExternalLinks(True)
        self.chat_browser.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_browser.customContextMenuRequested.connect(self.show_chat_context_menu)

        # --- Input Area ---
        self.input_box = QTextEdit(self)
        self.input_box.setPlaceholderText("Type your message...")
        self.input_box.setMaximumHeight(5 * self.input_box.fontMetrics().height())
        self.input_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.input_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.input_box.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.input_box.customContextMenuRequested.connect(self.show_input_context_menu)

        self.attach_button = QPushButton("Attach File", self)
        self.attach_button.clicked.connect(self.attach_file)
        self.emojis_button = QPushButton("Emojis", self)
        self.emojis_button.clicked.connect(self.show_emoji_picker)
        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        # --- Deep Research Button ---
        self.deep_research_button = QPushButton("Deep Research", self)
        self.deep_research_button.clicked.connect(self.start_deep_research)

        # Input layout
        input_hbox = QHBoxLayout()
        input_hbox.addWidget(self.input_box)
        input_hbox.addWidget(self.attach_button)
        input_hbox.addWidget(self.emojis_button)
        input_hbox.addWidget(self.send_button)
        input_hbox.addWidget(self.deep_research_button) # Add research button

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.chat_browser)
        main_layout.addLayout(input_hbox)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Status Label ---
        self.status_label = QLabel("Ready", self)  # For displaying status
        main_layout.addWidget(self.status_label)

    def show_emoji_picker(self):
        self.emoji_dialog.show()

    def show_chat_context_menu(self, position):
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.chat_browser.copy)
        menu.popup(self.chat_browser.viewport().mapToGlobal(position))

    def show_input_context_menu(self, position):
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.input_box.copy)
        paste_action = menu.addAction("Paste")
        paste_action.triggered.connect(self.input_box.paste)
        cut_action = menu.addAction("Cut")
        cut_action.triggered.connect(self.input_box.cut)
        select_all_action = menu.addAction("Select All")
        select_all_action.triggered.connect(self.input_box.selectAll)
        menu.popup(self.input_box.mapToGlobal(position))

    def export_chat_history(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Chat", "", "HTML Files (*.html)")
        if file_name:
            try:
                html_content = self.generate_html_chat_log()
                full_html = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>WhatsApp Formatter Output</title>
                    <style>{self.css_style}</style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                QMessageBox.information(self, "Export Successful", "Chat history exported to HTML.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export chat history: {e}")

    def generate_html_chat_log(self):
        html_log = ""
        for sender, message, timestamp in self.chat_log:
            formatted_message = self.format_whatsapp_text(message)
            if sender == "You":
                html_log += f"<div class='message user-message'><p>{html.escape(timestamp)} <b>{html.escape(sender)}:</b></p><p>{formatted_message}</p></div>"
            elif sender == "AI":
                html_log += f"<div class='message ai-message'><p>{html.escape(timestamp)} <b>{html.escape(sender)}:</b></p><p>{formatted_message}</p></div>"
            else:  # Error messages
                html_log += f"<div class='message error-message'><p>{html.escape(timestamp)} <b>{html.escape(sender)}:</b></p><p>{formatted_message}</p></div>"
        return html_log

    def import_chat_history(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Chat", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    self.chat_log = data.get('chat_log', [])
                    self.chat_browser.clear()
                    for entry in self.chat_log:
                        sender, message, timestamp = entry
                        self.display_message(sender, message, timestamp)
                QMessageBox.information(self, "Import Successful", "Chat history imported from JSON.")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Could not import chat history: {e}")

    def load_config(self):
        try:
            with open('api_configuration.yaml', 'r') as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            return {}  # Return empty dict if not found
        except yaml.YAMLError as e:
            QMessageBox.critical(self, "Configuration Error", f"Error reading configuration file: {e}")
            return {}

    def save_config(self, api_url, api_key, model, system_prompt, temperature, zenrows_key):
        if not api_url.strip():
            QMessageBox.warning(self, "Configuration Error", "API URL cannot be empty.")
            return False
        parsed_url = urlparse(api_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            QMessageBox.warning(self, "Configuration Error", "Invalid API URL format.")
            return False
        if not api_key.strip():
            QMessageBox.warning(self, "Configuration Error", "API Key cannot be empty.")
            return False
        temperature_val = float(temperature)
        if not 0.0 <= temperature_val <= 2.0:
            QMessageBox.warning(self, "Configuration Error", "Temperature must be between 0.0 and 2.0.")
            return False

        config = {
            'API_Url': api_url,
            'API_Key': api_key,
            'Model': model,
            'System_Prompt': system_prompt,
            'Temperature': temperature_val,
            'ZenRows_API_Key': zenrows_key  # Add ZenRows key
        }
        try:
            with open('api_configuration.yaml', 'w') as file:
                yaml.dump(config, file)
            self.config = config
            self.init_openai_client()
            QMessageBox.information(self, "Configuration Saved", "API configuration saved successfully.")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Could not save configuration: {e}")
            return False

    def show_config_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("API Configuration")
        form_layout = QFormLayout()

        api_url_input = QLineEdit(self.config.get('API_Url', '') if self.config else '')
        api_key_input = QLineEdit(self.config.get('API_Key', '') if self.config else '')

        self.model_combo = QComboBox(self)
        models = ["gpt-4o-mini", "o3-mini", "deepseek/deepseek-chat", "deepseek/deepseek-r1", "deepseek-reasoner",
                  "deepseek-chat"]
        self.model_combo.addItems(models)
        current_model = self.config.get('Model', 'gpt-3.5-turbo') if self.config else 'gpt-3.5-turbo'
        self.model_combo.setCurrentText(current_model)
        self.model_combo.setEditable(True)

        system_prompt_input = QLineEdit(self.config.get('System_Prompt', '') if self.config else '')
        temperature_slider = QSlider(Qt.Orientation.Horizontal)
        temperature_slider.setRange(0, 200)
        temperature_slider.setValue(int((self.config.get('Temperature', 0.5) if self.config else 0.5) * 100))
        temperature_label = QLabel(f"Temperature ({temperature_slider.value() / 100.0:.2f})")
        temperature_slider.valueChanged.connect(
            lambda value: temperature_label.setText(f"Temperature ({(value / 100.0):.2f})"))

        # --- ZenRows API Key Input ---
        zenrows_key_input = QLineEdit(self.config.get('ZenRows_API_Key', '') if self.config else '')
        form_layout.addRow("ZenRows API Key:", zenrows_key_input)


        form_layout.addRow("API URL:", api_url_input)
        form_layout.addRow("API Key:", api_key_input)
        form_layout.addRow("Model:", self.model_combo)
        form_layout.addRow("System Prompt:", system_prompt_input)
        form_layout.addRow("Temperature:", temperature_slider)
        form_layout.addRow("", temperature_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(lambda: self.save_config(api_url_input.text(), api_key_input.text(),
                                                          self.model_combo.currentText(), system_prompt_input.text(),
                                                          temperature_slider.value() / 100.0, zenrows_key_input.text()))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        form_layout.addRow(buttons)
        dialog.setLayout(form_layout)

        dialog.exec()

    def attach_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Attach File", "", "All Files (*)")
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
                preview_length = 500
                preview = content[:preview_length]
                if len(content) > preview_length:
                    preview += "...\n[Preview limited to first 500 characters]"
                self.attached_file_content = content
                self.attached_file_name = file_name

                self.display_file_attachment(file_name, preview)
            except Exception as e:
                QMessageBox.warning(self, "File Error", f"Could not read file: {e}")

    def display_file_attachment(self, file_name, preview):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        formatted_message = (
            f"<div style='text-align: right; background-color: lightyellow; padding: 5px; border-radius: 10px;'>"
            f"<small>[{timestamp}]</small> <b>You attached: {os.path.basename(file_name)}</b><br>"
            f"<pre style='white-space: pre-wrap; font-family: monospace;'>{preview}</pre></div>"
        )
        self.chat_browser.append(formatted_message)

    def send_message(self):
        user_input = self.input_box.toPlainText()
        full_message_content = user_input

        if self.attached_file_content:
            full_message_content += f"\n\n[Attached File: {os.path.basename(self.attached_file_name)}]\n{self.attached_file_content}"

        if not user_input and not self.attached_file_content:
            return

        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        self.display_message("You", user_input, timestamp, is_user=True,
                             file_attached=self.attached_file_name is not None)

        self.chat_log.append(["You", full_message_content, timestamp])

        if self.session_log_file:
            log_entry = ["You", full_message_content, timestamp]
            self.write_to_session_log(log_entry)

        self.input_box.clear()
        self.attached_file_content = None
        self.attached_file_name = None

        self.call_api(full_message_content)

    def display_message(self, sender, message, timestamp, is_user=False, file_attached=False, is_error=False):
        formatted_message = self.format_whatsapp_text(message)
        sender_display = "You" if is_user else "AI"
        if is_user:
            html_content = f"<div class='message user-message'><p>{html.escape(timestamp)} <b>{html.escape(sender_display)}:</b></p><p>{formatted_message}</p></div>"
        elif is_error:
            html_content = f"<div class='message error-message'><p>{html.escape(timestamp)} <b>{html.escape(sender_display)}:</b></p><p>{formatted_message}</p></div>"
        else:
            html_content = f"<div class='message ai-message'><p>{html.escape(timestamp)} <b>{html.escape(sender_display)}:</b></p><p>{formatted_message}</p></div>"

        cursor = self.chat_browser.textCursor()
        cursor.insertHtml(html_content)
        cursor.insertBlock()
        self.chat_browser.moveCursor(QTextCursor.MoveOperation.End)

    def format_whatsapp_text(self, text):
        text = html.escape(text)
        text = re.sub(r'\*\*\*\*(.*?)\*\*\*\*', r'<b>@#%@#%@#%\1@#%@#%@#%</b>', text)
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b>@#%@#%\1@#%@#%</b>', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>@#%\1@#%</b>', text)
        text = text.replace("\n", "<br>")
        text = text.encode('utf-16', 'surrogatepass').decode('utf-16')
        text = self.apply_whatsapp_tags(text)
                text = re.sub(r'@#%', r'*', text)
        return text

    def apply_whatsapp_tags(self, text):
        # Simplified placeholder method (robust and readable)
        text = text.replace("*", "###BOLD###")
        text = text.replace("###BOLD###", "<b>", 1)
        text = text.replace("###BOLD###", "</b>", 1)
        while "###BOLD###" in text:
            text = text.replace("###BOLD###", "<b>", 1)
            text = text.replace("###BOLD###", "</b>", 1)
        text = text.replace("<b></b>", "")

        text = text.replace("_", "###ITALIC###")
        text = text.replace("###ITALIC###", "<i>", 1)
        text = text.replace("###ITALIC###", "</i>", 1)
        while "###ITALIC###" in text:
            text = text.replace("###ITALIC###", "<i>", 1)
            text = text.replace("###ITALIC###", "</i>", 1)
        text = text.replace("<i></i>", "")

        text = text.replace("~", "###STRIKE###")
        text = text.replace("###STRIKE###", "<s>", 1)
        text = text.replace("###STRIKE###", "</s>", 1)
        while "###STRIKE###" in text:
            text = text.replace("###STRIKE###", "<s>", 1)
            text = text.replace("###STRIKE###", "</s>", 1)
        text = text.replace("<s></s>", "")

        return text

    def call_api(self, prompt):
        if not self.config:
            QMessageBox.warning(self, "API Error",
                                "API configuration is missing. Please configure API settings.")
            return
        if not self.openai_client:
            QMessageBox.warning(self, "API Error", "OpenAI client not initialized. Check API configuration.")
            return

        model = self.config.get('Model')
        system_prompt = self.config.get('System_Prompt')
        temperature = float(self.config.get('Temperature'))

        try:
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            ai_response_content = completion.choices[0].message.content
            timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
            self.display_message("AI", ai_response_content, timestamp)
            self.chat_log.append(["AI", ai_response_content, timestamp])

            if self.session_log_file:
                log_entry = ["AI", ai_response_content, timestamp]
                self.write_to_session_log(log_entry)

        except Exception as e:
            error_message = f"API request failed: {e}"
            self.display_message("AI", f"Error: {error_message}",
                                 QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]"),
                                 is_error=True)
            self.chat_log.append(
                ["AI", f"Error: {error_message}", QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")])

    def create_session_log(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_sn = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_dir = "chat_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, f"Chat_{timestamp}.json")

        try:
            self.session_log_file = open(log_file_path, 'w')
            log_data = {
                "session_name": f"Chat {timestamp_sn}",
                "chat_log": []
            }
            json.dump(log_data, self.session_log_file, indent=4)
            self.session_log_file.flush()
            self.session_log_file.close()
            self.session_log_file = open(log_file_path, 'r+')
        except Exception as e:
            QMessageBox.critical(self, "Logging Error", f"Could not create session log file: {e}")
            self.session_log_file = None

    def write_to_session_log(self, log_entry):
        if self.session_log_file:
            try:
                self.session_log_file.seek(0)
                data = json.load(self.session_log_file)
                data['chat_log'].append(log_entry)
                self.session_log_file.seek(0)
                self.session_log_file.truncate(0)
                json.dump(data, self.session_log_file, indent=4)
                self.session_log_file.flush()
            except Exception as e:
                QMessageBox.critical(self, "Logging Error", f"Could not write to session log file: {e}")

    def close_session_log(self):
        if self.session_log_file:
            try:
                self.session_log_file.seek(0)
                data = json.load(self.session_log_file)
                data["attached_files"] = {}
                data["created_at"] = datetime.datetime.now().timestamp()
                self.session_log_file.seek(0)
                self.session_log_file.truncate(0)
                json.dump(data, self.session_log_file, indent=4)
                self.session_log_file.close()
            except Exception as e:
                QMessageBox.critical(self, "Logging Error", f"Could not close session log file properly: {e}")

    def closeEvent(self, event):
        """Override closeEvent to ensure log file is closed."""
        self.close_session_log()
        super().closeEvent(event)

    # --- Deep Research Integration ---

    def start_deep_research(self):
        """
        Starts the deep research process.
        """
        query = self.input_box.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a research topic.")
            return

        # Clear existing chat log and display message.
        self.chat_log = []
        self.chat_browser.clear()
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        self.display_message("AI", f"Starting deep research on: {query}", timestamp)


        # Disable input and regular send, enable research button.
        self.input_box.setEnabled(False)
        self.send_button.setEnabled(False)
        self.deep_research_button.setEnabled(False)
        self.attach_button.setEnabled(False)

        # Create and start the research thread.
        self.research_thread = DeepResearchThread(query, self.config)
        self.research_thread.research_started.connect(self.on_research_started)
        self.research_thread.subquestion_generated.connect(self.on_subquestion_generated)
        self.research_thread.summary_ready.connect(self.on_summary_ready)
        self.research_thread.synthesis_ready.connect(self.on_synthesis_ready)
        self.research_thread.research_complete.connect(self.on_research_complete)
        self.research_thread.research_error.connect(self.on_research_error)
        self.research_thread.sources_ready.connect(self.on_sources_ready)
        self.research_thread.start()


    def on_research_started(self):
        """
        Called when the research thread starts.
        """
        self.status_label.setText("Research in progress...")

    def on_subquestion_generated(self, subquestion, depth):
        """
        Called when a sub-question is generated.
        """
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        # Display sub-questions at different depths with different colors
        if depth == 0:
          self.display_message("AI", f"Sub-question: {subquestion}", timestamp)
        elif depth == 1:
          self.display_message("AI", f"Deeper Sub-question: {subquestion}", timestamp)
        else:
          self.display_message("AI", f"Level {depth} Sub-question: {subquestion}", timestamp)

    def on_summary_ready(self, summary):
        """
        Called when a summary is ready.
        """
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        self.display_message("AI", f"Summary: {summary}", timestamp)

    def on_synthesis_ready(self, synthesis):
        """
        Called when the final synthesis is ready.
        """
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        self.display_message("AI", f"Synthesis: {synthesis}", timestamp)

    def on_sources_ready(self, sources):
        """
        Called when all sources are ready
        """
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        formatted_sources = "<br>".join([f"<a href='{url}'>{url}</a>" for url in sources])
        self.display_message("AI", f"Sources:<br>{formatted_sources}", timestamp)


    def on_research_complete(self):
        """
        Called when the research thread finishes.
        """
        self.status_label.setText("Research complete.")
        # Re-enable input and buttons.
        self.input_box.setEnabled(True)
        self.send_button.setEnabled(True)
        self.deep_research_button.setEnabled(True)
        self.attach_button.setEnabled(True)
        self.input_box.clear()


    def on_research_error(self, error_message):
        """
        Called if there's an error during research.
        """
        timestamp = QDateTime.currentDateTime().toString("[yyyy-MM-dd hh:mm:ss]")
        self.display_message("AI", f"Research Error: {error_message}", timestamp, is_error=True)
        # Re-enable controls.
        self.input_box.setEnabled(True)
        self.send_button.setEnabled(True)
        self.deep_research_button.setEnabled(True)
        self.attach_button.setEnabled(True)
        self.input_box.clear()
        self.status_label.setText("Ready")  # Reset status


if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_app = AIChatApp()
    chat_app.show()
    sys.exit(app.exec())
