import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

import torch
from torch import nn


class Loss():

    def __init__(self, args):
        self.args = args
        self.cos = nn.CosineSimilarity(dim=-1)

    def train_loss_fct(self, criterion, a, p, n, neg_weight=0):

        positive_similarity = self.cos(a.unsqueeze(1), p.unsqueeze(0)) / self.args['temperature']
        negative_similarity = self.cos(a.unsqueeze(1), n.unsqueeze(0)) / self.args['temperature']
        
        cosine_similarity = torch.cat([positive_similarity, negative_similarity], dim=1).to(self.args['device'])

        labels = torch.arange(cosine_similarity.size(0)).long().to(self.args['device'])

        # Calculate loss with hard negatives
        weights = torch.tensor(
            [[0.0] * (cosine_similarity.size(-1) - negative_similarity.size(-1)) + [0.0] * i + [neg_weight] + [0.0] * (negative_similarity.size(-1) - i - 1) for i in range(negative_similarity.size(-1))]
        ).to(self.args['device'])

        cosine_similarity = cosine_similarity + weights
        loss = criterion(cosine_similarity, labels)

        return loss

    def evaluation(self, embeddings1, embeddings2, labels):
        
        embeddings1 = embeddings1.cpu().numpy()
        embeddings2 = embeddings2.cpu().numpy()
        labels = labels.cpu().numpy().flatten()

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        score = {'eval_pearson_cosine': eval_pearson_cosine,
                 'eval_spearman_cosine': eval_spearman_cosine,
                 'eval_pearson_manhattan': eval_pearson_manhattan,
                 'eval_spearman_manhattan': eval_spearman_manhattan,
                 'eval_pearson_euclidean': eval_pearson_euclidean,
                 'eval_spearman_euclidean': eval_spearman_euclidean,
                 'eval_pearson_dot': eval_pearson_dot,
                 'eval_spearman_dot': eval_spearman_dot}

        return  score