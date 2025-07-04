
import torch
import torch.nn as nn
import math


class GreedyCTCDecoder(nn.Module):

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels=labels
        self.blank = blank

    def forward(self, emission:torch.Tensor):
        """Given a sequence emission over labels, get the best path"""

        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    
    def forward_2(self, emission, prob_threshold):
        probs = torch.exp(emission)  # shape: (T, C)
        max_probs, indices = torch.max(probs, dim=-1)  # get max prob and corresponding index at each timestep

        # Apply probability threshold
        indices = torch.where(max_probs >= prob_threshold, indices, torch.tensor(self.blank, device=indices.device))

        # Collapse repeated tokens and remove blanks
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        # Convert to label string
        joined = " ".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    
    def forward_with_quality(
            self, emission, batch_size=1, prob_threshold=0, full_qualities=False):
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
            quality = -10 * math.log10(1 - (prob_score / counter).item())
            individual_qualities = -10 * torch.log10(1 - selected_probs)
        else:
            quality = 0  # or some defined fallback value

        # Optional: get full payload predictions if still needed
        # payload_predictions = probs[mask]  # optional, only if needed
        # Apply probability threshold
        indices = torch.where(max_probs >= prob_threshold, indices, torch.tensor(self.blank, device=indices.device))
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels[i] for i in indices])

        greedy_transcript = joined.replace("|", " ").strip().split()


        if full_qualities:
            return greedy_transcript, quality, individual_qualities
        return greedy_transcript, quality

