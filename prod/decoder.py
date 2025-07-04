
import torch
import torch.nn as nn
import numpy as np
import math


class GreedyCTCDecoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        labels_int = np.arange(n_classes).tolist()
        self.labels = [f"{i}" for i in labels_int]
        self.blank = 0
    
    def forward(self, emission):
        probs = torch.exp(emission)
        max_probs, indices = torch.max(probs, dim=-1)

        # Mask for payload motif indices (1 to 8)
        mask = (indices > 0) & (indices < 9)

        # Select relevant probabilities
        selected_probs = max_probs[mask]

        # Compute quality only if we have selected any
        if selected_probs.numel() > 0:
            prob_score = selected_probs.sum()
            counter = selected_probs.numel()
            quality = -10 * math.log10(
                1 - (prob_score / counter).item())
            individual_qualities = -10 * torch.log10(1 - selected_probs)
        else:
            quality = 0  # or some defined fallback value

        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels[i] for i in indices])

        greedy_transcript = joined.replace(
            "|", " ").strip().split()

        return greedy_transcript, quality, individual_qualities