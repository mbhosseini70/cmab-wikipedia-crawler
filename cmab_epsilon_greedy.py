from sentence_transformers import SentenceTransformer, util
import random
import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, subject, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.05):
        self.subject = subject
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.counts = {'download': 0, 'skip': 0}
        self.rewards = {'download': [], 'skip': []}
        
        # Load Sentence-BERT model for sentence embeddings
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    #-----------------------------------------------------------
    # Function for calculating the similarity between the page context and subject
    # helping in calculating the reward
    def calculate_similarity(self, page_content, chunk_size=512):

        # Encode the subject into embeddings
        subject_embedding = self.model.encode(self.subject, convert_to_tensor=True)

        # Encode the context into embeddings
        content_tokens = page_content.split()  
        first_chunk = " ".join(content_tokens[:chunk_size])  # Take the first chunk
        content_embedding = self.model.encode([first_chunk], convert_to_tensor=True)

        # Calculate the cosine similarity between subject and content embeddings
        similarity = util.pytorch_cos_sim(subject_embedding, content_embedding)

        # Calculate the cosine similarity between subject and content embeddings
        similarity = util.pytorch_cos_sim(subject_embedding, content_embedding)
        single_similarity = similarity.item()
        print(f"Similarity score between the subject and the first chunk of the page content: {single_similarity}")
        return single_similarity
    
    #-----------------------------------------------------------
    # The function to check the similarity of uri with subject. 
    # use to filter unrelated links.
    def calculate_url_similarity(self, uri):
        url_text = uri.replace('_', ' ').replace('/wiki/', '')
        subject_embedding = self.model.encode(self.subject, convert_to_tensor=True)
        url_embedding = self.model.encode(url_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(subject_embedding, url_embedding).item()
        return similarity


    #-----------------------------------------------------------
    # The function for choosing action by agnet.
    def choose_action(self, page_content):
        if random.random() < self.epsilon:
            action = random.choice(['download', 'skip'])
        else:
            download_reward = np.mean(self.rewards['download']) if self.rewards['download'] else 0
            skip_reward = np.mean(self.rewards['skip']) if self.rewards['skip'] else 0
            action = 'download' if download_reward >= skip_reward else 'skip'
        # decaying the epsilon    
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action


    #-----------------------------------------------------------
    # The function for calculting the reward for agent's action.
    def update_rewards(self, action, similarity):
        if action == 'download' and similarity >= 0.3:
            reward = similarity
        else:
            reward = 0
        self.rewards[action].append(reward)
        self.counts[action] += 1
