import requests
from bs4 import BeautifulSoup
from cmab_epsilon_greedy import EpsilonGreedyAgent
from cmab_ucb import UCBAgent
from collections import deque 
from plot import save_plots
import os




class WikipediaCrawlerWithCMAB:
    def __init__(self, start_uri, subject, size_budget_kib, depth_limit=3, epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.05):
        """
        Initialize the Wikipedia Crawler with contextual multi-armed bandits (CMAB).
        
        Args:
            start_uri (str): Starting Wikipedia URI.
            subject (str): Subject to which the crawler will relate.
            size_budget_kib (int): Maximum memory budget for downloaded pages.
            depth_limit (int): Maximum depth for BFS traversal.
        """
        self.start_uri = start_uri
        self.subject = subject
        self.size_budget = size_budget_kib * 1024  # Convert KiB to bytes
        self.depth_limit = depth_limit
        self.visited_uris = set()

        # Create directories to save plots 
        os.makedirs("plots", exist_ok=True)

        self.metrics_eg = {'similarity_scores': [], 'cumulative_rewards': [], 'downloaded_count': 0, 'skipped_count': 0, 'downloaded_pages': []}
        self.metrics_ucb = {'similarity_scores': [], 'cumulative_rewards': [], 'downloaded_count': 0, 'skipped_count': 0, 'downloaded_pages': []}

    #-----------------------------------------------------------
    # Fetching the HTML content of a Wikipedia page.
    def fetch_page_content(self, uri):
        full_url = f"https://en.wikipedia.org{uri}"
        try:
            response = requests.get(full_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except requests.RequestException as e:
            print(f"Error fetching {full_url}: {e}")
            return None
        
    #-----------------------------------------------------------
    # Extracting valid Wikipedia links from the page.
    def extract_wiki_links(self, soup):
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/wiki/') and ':' not in href and href not in self.visited_uris:
                links.append(href)
        return links

    #-----------------------------------------------------------
    # Calculating the size of the page content in bytes.
    def get_page_size(self, page_content):
        return len(page_content.encode('utf-8'))

    #-----------------------------------------------------------
    # Crawling using hybrid BFS/DFS methods
    def bfs_crawl(self, agent, metrics, high_similarity_threshold=0.2):

        # Initialize a queue for BFS: each element is a tuple (uri, depth)
        queue = deque([(self.start_uri, 0)])

        while queue:
            # Get the next page to crawl and its depth
            uri, current_depth = queue.popleft()

            # Stop if depth limit is reached
            if current_depth > self.depth_limit:
                continue

            # Track total size of downloaded pages
            total_downloaded_size = sum([self.get_page_size(text) for _, text in metrics['downloaded_pages']])

            # Calculate remaining memory
            remaining_memory = self.size_budget - total_downloaded_size  # In bytes

            # If remaining memory is too small, stop all further exploration
            if remaining_memory < 4 * 1024:  # threshold of 4 KiB
                print(f"Stopping all exploration. Remaining memory is too small: {remaining_memory / 1024:.2f} KiB")
                break  

            # Calculate URL (title) similarity to the subject
            url_similarity = agent.calculate_url_similarity(uri)
            print(f"Title similarity for {uri}: {url_similarity}")

            # Fetch the content of the page
            soup = self.fetch_page_content(uri)
            if not soup:
                continue  

            page_text = soup.get_text()
            page_size = self.get_page_size(page_text)

            # If the page is too large to fit into remaining memory, skip downloading it
            if page_size > remaining_memory:
                metrics['skipped_count'] += 1
                continue

            # Calculate content similarity to the subject (context)
            content_similarity = agent.calculate_similarity(page_text)

            # Get the action from the agent considering both title and content similarity
            action = agent.choose_action(content_similarity, url_similarity)

            # Allow visiting pages but only download them once
            if action == 'download' and uri not in self.visited_uris and total_downloaded_size + page_size <= self.size_budget:
                # Add the URI to visited_uris to prevent future downloads
                self.visited_uris.add(uri)

                metrics['downloaded_count'] += 1
                metrics['downloaded_pages'].append((uri, page_text))  

                # Print downloaded size and remaining memory
                print(f"Downloaded: {uri} (size: {page_size / 1024:.2f} KiB), Remaining memory: {(remaining_memory - page_size) / 1024:.2f} KiB")

                # Track similarity for downloaded pages
                metrics['similarity_scores'].append(content_similarity)

                # If the page is highly relevant, explore links from it in a DFS manner
                if url_similarity > high_similarity_threshold :
                    print(f"High similarity detected ({url_similarity:.2f}) for {uri}. Going deeper...")
                    sub_links = self.extract_wiki_links(soup)
                    # Push sub-links with incremented depth to a temporary stack for DFS
                    stack = [(link, current_depth + 1) for link in sub_links]
                    
                    while stack:
                        sub_uri, sub_depth = stack.pop()  # DFS exploration
                        if sub_depth <= self.depth_limit and sub_uri not in self.visited_uris:
                            queue.appendleft((sub_uri, sub_depth))  # Add to the front of the queue
            else:
                metrics['skipped_count'] += 1

            # Extract valid Wikipedia links from the current page
            if content_similarity <= high_similarity_threshold:  # Regular BFS for less relevant links
                links = self.extract_wiki_links(soup)
                # Enqueue all links found on this page with an incremented depth
                for link in links:
                    queue.append((link, current_depth + 1))

            # Update rewards and track cumulative rewards
            agent.update_rewards(action, content_similarity)
            if action == 'download':
                if metrics['cumulative_rewards']:
                    metrics['cumulative_rewards'].append(metrics['cumulative_rewards'][-1] + agent.rewards[action][-1])
                else:
                    metrics['cumulative_rewards'].append(agent.rewards[action][-1])
            elif metrics['cumulative_rewards']:
                # If action is 'skip', append the last cumulative reward without change
                metrics['cumulative_rewards'].append(metrics['cumulative_rewards'][-1])

    #-----------------------------------------------------------
    # The function to run epsilon_greedy
    def run_epsilon_greedy(self, epsilon, epsilon_decay, min_epsilon):
        print(".................................Running Epsilon-Greedy Agent.................................")
        epsilon_greedy_agent = EpsilonGreedyAgent(self.subject, epsilon, epsilon_decay, min_epsilon)
        self.bfs_crawl(epsilon_greedy_agent, self.metrics_eg)
        print(f"\nEpsilon-Greedy Agent: Downloaded {self.metrics_eg['downloaded_count']} pages.")

    #-----------------------------------------------------------
    # The function to run ucb
    def run_ucb(self):
        print(".................................Running UCB Agent.................................")
        ucb_agent = UCBAgent(self.subject)
        self.bfs_crawl(ucb_agent, self.metrics_ucb)
        print(f"\nUCB Agent: Downloaded {self.metrics_ucb['downloaded_count']} pages.")

    #-----------------------------------------------------------
    # Ploting and Saving the plots (Do not show)
    def plot_results(self):
        save_plots(self.metrics_eg, self.metrics_ucb)



    #-------------------------------------------------------------
    # The function to show the downloaded results. 
    def display_downloaded_pages(self):
        print("\n--- Downloaded Pages Summary ---")
        for i, (uri, content) in enumerate(self.metrics_eg['downloaded_pages'], start=1):
            print(f"{i}. Topic: {uri}")
            print(f"   Content (first 500 characters):\n   {content[:500]}")
            print("-" * 80)

