import matplotlib.pyplot as plt
import numpy as np
import os


# saving the plots and comparing them. 
def save_plots(metrics_eg, metrics_ucb):

    os.makedirs("plots", exist_ok=True)

    # 1. Comparison of Pages Downloaded vs Skipped
    labels = ['Downloaded', 'Skipped']
    counts_eg = [metrics_eg['downloaded_count'], metrics_eg['skipped_count']]
    counts_ucb = [metrics_ucb['downloaded_count'], metrics_ucb['skipped_count']]

    x = np.arange(len(labels)) 
    width = 0.35  

    fig, ax = plt.subplots()
    
    # Strong green for downloaded and light green for skipped (Epsilon-Greedy)
    rects1 = ax.bar(x - width/2, counts_eg, width, 
                    label='Epsilon-Greedy', 
                    color=['#228B22', '#98FB98'])  # Dark green and light green

    # Strong blue for downloaded and light blue for skipped (UCB)
    rects2 = ax.bar(x + width/2, counts_ucb, width, 
                    label='UCB', 
                    color=['#0000FF', '#ADD8E6'])  # Blue and light blue

    # Add some text for labels, title, and axes ticks
    ax.set_ylabel('Count')
    ax.set_title('Number of Pages Downloaded vs Skipped')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add numbers on top of the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels(rects1)
    add_labels(rects2)

    fig.tight_layout()
    plt.savefig('plots/downloaded_vs_skipped_comparison.png')
    plt.close()

    # 2. Comparison of Similarity Scores Over Time
    plt.figure()
    plt.plot(metrics_eg['similarity_scores'], label='Epsilon-Greedy', color='green')
    plt.plot(metrics_ucb['similarity_scores'], label='UCB', color='blue')
    plt.title('Similarity of Downloaded Pages Over Time')
    plt.xlabel('Downloaded Pages')
    plt.ylabel('Similarity')
    plt.legend()
    plt.savefig('plots/similarity_over_time_comparison.png')
    plt.close()

    # 3. Comparison of Cumulative Reward Over Time
    plt.figure()
    plt.plot(metrics_eg['cumulative_rewards'], label='Epsilon-Greedy', color='green')
    plt.plot(metrics_ucb['cumulative_rewards'], label='UCB', color='blue')
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Pages')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig('plots/cumulative_reward_comparison.png')
    plt.close()