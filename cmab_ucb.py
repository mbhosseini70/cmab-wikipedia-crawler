from sentence_transformers import SentenceTransformer, util
import numpy as np

class UCBAgent:
    def __init__(self, subject):
        self.subject = subject
        self.t = 0
        self.counts = {'download': 0, 'skip': 0}
        self.rewards = {'download': [], 'skip': []}
        
        
        self.model = SentenceTransformer('all-MiniLM-L12-v2')



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
    # make decision based on the highest UCB value
    # The UCB value is computed as the sum of the average reward and a confidence interval.
    def choose_action(self, page_content):
        self.t += 1
        ucb_values = {}
        for action in ['download', 'skip']:
            count = self.counts[action]
            if count == 0:
                ucb_values[action] = float('inf')
            else:
                mean_reward = np.mean(self.rewards[action]) if self.rewards[action] else 0
                ucb_values[action] = mean_reward + np.sqrt(2 * np.log(self.t) / count)
        return max(ucb_values, key=ucb_values.get)



    #-----------------------------------------------------------
    # The function for calculting the reward for agent's action.
    def update_rewards(self, action, similarity):
        if action == 'download' and similarity >= 0.3:
            reward = similarity
        else:
            reward = 0
        self.rewards[action].append(reward)
        self.counts[action] += 1
