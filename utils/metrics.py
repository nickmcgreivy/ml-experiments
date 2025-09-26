from collections import defaultdict
import math
import torch

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Neither sequence is tokenized.
    
    Args:
        pred_seq (str): preprocessed prediction sentence
        label seq (str): preprocessed target sentence
    
    Returns:
        bleu (float): BLEU metric"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def accuracy(logits, y):
    """Assumes logits of shape (B, num_classes), y of shape (B,)"""
    predicted_label = torch.argmax(logits, dim=-1)
    return (predicted_label == y).mean()

def topk_accuracy(logits, y, k=5):
    """Same assumptions as accuracy"""
    _, topk_labels = torch.topk(logits, k)
    return (topk_labels == y[..., None]).sum(dim=1).mean()