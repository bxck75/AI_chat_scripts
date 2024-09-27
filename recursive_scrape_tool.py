# filename: recursive_scraper.py

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time

class RecursiveScraper:
    def __init__(self, base_url, max_depth=3):
        self.base_url = base_url
        self.visited = set()  # to track visited links
        self.max_depth = max_depth

    def fetch_page(self, url):
        """Fetch the HTML content of a page."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                time.sleep(1)  # Sleep for 1 second between requests
                return response.text
            else:
                print(f"Failed to fetch {url}, Status code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    def parse_page(self, html, url):
        """Parse the HTML content to extract text and Python code snippets."""
        soup = BeautifulSoup(html, 'html.parser')
        page_text = soup.get_text()
        python_code_snippets = self.extract_python_code(html)

        # Scrape text and Python code
        print(f"\nScraping URL: {url}")
        print("Text Content:", page_text[:500])  # printing first 500 chars of text
        print("\nPython Code Snippets:", python_code_snippets)

        # Extract links for recursive crawling
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if self.should_visit(full_url):
                self.crawl(full_url)

    def extract_python_code(self, html):
        """Extract Python code snippets using regex for <code> or <pre> tags."""
        python_code = re.findall(r'<(code|pre)>(.*?)</\1>', html, re.DOTALL)
        return [code for _, code in python_code]

    def should_visit(self, url):
        """Check if the link should be visited: within the base domain and not visited yet."""
        if urlparse(url).netloc != urlparse(self.base_url).netloc:
            return False
        if url in self.visited:
            return False
        return True

    def crawl(self, url, depth=0):
        """Recursively crawl and scrape content."""
        if depth > self.max_depth:
            return

        self.visited.add(url)
        html = self.fetch_page(url)
        if html:
            self.parse_page(html, url)

    def start(self):
        """Start the crawling process."""
        self.crawl(self.base_url)

# Usage
if __name__ == "__main__":
    base_url = "https://example.com"  # Replace with the starting URL
    scraper = RecursiveScraper(base_url, max_depth=3)
    scraper.start()
