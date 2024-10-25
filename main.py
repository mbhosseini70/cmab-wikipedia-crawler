from webpage_crawler import WikipediaCrawlerWithCMAB

def main():
    """
    Main function to run the Wikipedia Crawler using CMAB agent.
    It initializes the crawler, sets parameters, and runs both epsilon-greedy 
    and UCB RL agents.
    """
    start_uri = '/wiki/Reinforcement_learning'
    subject = 'artificial Intelligence'
    size_budget_kib = 5 * 1024  # = KiB

    epsilon = 1  
    epsilon_decay = 0.999
    min_epsilon = 0.05  

    # Initialize Wikipedia crawler with CMAB
    crawler = WikipediaCrawlerWithCMAB(start_uri, subject, size_budget_kib)

    # Run epsilon-greedy CMAB agent
    print("______________________Running epsilon greedy RL agent _______________________________")
    crawler.run_epsilon_greedy(epsilon, epsilon_decay, min_epsilon)

    # Run UCB CMAB agent
    print("______________________Running UCB RL agent _______________________________")
    crawler.run_ucb()

    # Plot the results after running both agents
    crawler.plot_results()

if __name__ == "__main__":
    main()