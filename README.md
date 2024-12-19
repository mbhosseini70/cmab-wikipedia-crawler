# Smart Wikipedia Crawler Using Contextual Multi-Armed Bandit

## Project Overview
This project implements a smart Wikipedia crawler that leverages a Contextual Multi-Armed Bandit (CMAB) framework to efficiently explore Wikipedia pages while adhering to storage constraints. The system employs two reinforcement learning (RL) strategies—epsilon-greedy and Upper Confidence Bound (UCB)—to decide whether to download or skip a page. Semantic similarity ensures that only relevant pages are prioritized for download.

---

## Code Architecture
The project is modular and consists of the following core components:

### `main.py`
- Orchestrates the execution of the crawler.
- Coordinates the epsilon-greedy and UCB strategies.
- Executes crawling and plotting tasks.

### `webpage_crawler.py`
- Contains the `WikipediaCrawlerWithCMAB` class, which:
  - Navigates Wikipedia pages using a hybrid Breadth-First Search (BFS) and Depth-First Search (DFS) strategy.
  - Handles page fetching, content extraction, and link exploration.
  - Manages memory limits during page downloads.

### `cmab_epsilon_greedy.py`
- Implements the Epsilon-Greedy Agent for RL.
- Balances exploration and exploitation by probabilistically selecting actions based on the epsilon parameter.

### `cmab_ucb.py`
- Implements the UCB Agent for RL.
- Selects actions based on Upper Confidence Bound values, balancing confidence intervals and average rewards.

### `plot.py`
- Generates visualizations to compare the performance of the two agents.
- Saves plots depicting metrics such as pages downloaded vs skipped, similarity scores, and cumulative rewards over time.

---

## Key Features

### Content Extraction
- Uses BeautifulSoup to scrape Wikipedia content and extract valid internal links.
- Encodes page content and URLs using Sentence-BERT models for semantic similarity calculations.

### Hybrid Search Strategy
- Combines BFS and DFS to ensure broad exploration and targeted deep dives into relevant links.

### Memory Management
- Monitors downloaded content size and halts further downloads if storage limits are reached.

### CMAB Agents
#### Epsilon-Greedy Agent
- Uses an exploration rate (epsilon) to select actions probabilistically.
- Decays epsilon over time to reduce exploration as the model learns.

#### UCB Agent
- Calculates UCB values for each action based on confidence intervals and average rewards.
- Prioritizes actions with high UCB scores to maximize long-term reward.

### Reward Function
- Prioritizes downloads of pages with high semantic similarity to the subject.
- Provides incremental rewards based on similarity scores.

### Visualization
- Compares the performance of epsilon-greedy and UCB strategies across various metrics.
- Highlights cumulative rewards, downloaded page relevance, and decision efficiency.

---

## Workflow Explanation

### Initialization
- Initializes the crawler with a starting Wikipedia page URI, a subject of interest, and a memory budget.
- Creates CMAB agents with specific parameters (e.g., epsilon for epsilon-greedy, confidence intervals for UCB).

### Crawling Process
1. Fetches page content and calculates its size.
2. Computes semantic similarity between the page content, subject, and URL.
3. Decides whether to download or skip the page based on the CMAB agent's action.
4. Updates rewards and tracks metrics for evaluation.

### Agent Strategies
#### Epsilon-Greedy
- Selects random actions with a probability of epsilon and the best-known action otherwise.
- Gradually reduces epsilon to shift focus toward exploitation.

#### UCB
- Computes the UCB score for each action as the sum of average reward and a confidence interval.
- Selects the action with the highest UCB score.

### Evaluation and Visualization
- Records metrics such as downloaded page count, skipped pages, similarity scores, and cumulative rewards.
- Generates plots to compare the strategies.

---

## Tools and Libraries
- **BeautifulSoup:** For web scraping and HTML content parsing.
- **Sentence-BERT Models:**
  - `paraphrase-MiniLM-L6-v2`: For epsilon-greedy agent embeddings.
  - `all-MiniLM-L12-v2`: For UCB agent embeddings.
- **Matplotlib:** For visualizing and saving performance plots.
- **NumPy:** For numerical computations and reward tracking.
- **Requests:** For fetching Wikipedia pages.
- **Collections (Deque):** For hybrid BFS/DFS crawling.

---

## Challenges Faced

### Hyperparameter Tuning
- Adjusting epsilon decay, UCB confidence intervals, and reward thresholds required extensive experimentation.

### Efficient Exploration
- Balancing exploration depth and breadth while adhering to memory constraints posed significant challenges.

### Execution Time
- Long runtimes for crawling and agent training made hyperparameter tuning and result evaluation time-intensive.

### Embedding Model Selection
- Identifying models that balance accuracy and computational efficiency was critical to performance.

---

## Results and Observations

### Epsilon-Greedy Strategy
- Demonstrated superior performance in terms of cumulative reward and relevance of downloaded pages.
- Balanced exploration and exploitation effectively by dynamically adjusting epsilon.

### UCB Strategy
- Showed bolder exploration but struggled to match epsilon-greedy in cumulative rewards.

### Visualization Insights
- Epsilon-greedy exhibited steadier growth in cumulative rewards over time.
- UCB provided valuable insights into high-risk, high-reward scenarios.

---

## Conclusion
This project successfully demonstrates the application of contextual multi-armed bandits for building an intelligent Wikipedia crawler. By leveraging semantic similarity and reinforcement learning, the system effectively prioritizes relevant content while staying within resource constraints. The hybrid search strategy and well-tuned agents make it a robust solution for exploring and extracting structured information from large-scale datasets like Wikipedia.
