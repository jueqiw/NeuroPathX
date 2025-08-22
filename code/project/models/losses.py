import sys
import random
from typing import Tuple

import torch
import numpy as np
from torch.nn import functional as F

# with help from ChatGPT and DeepSeek


def generate_pairs(train_labels, num_pairs=10):
    same_group_pairs = []
    different_group_pairs = []

    # Separate indices by group
    asd_indices = [i for i, label in enumerate(train_labels) if label == 1]
    control_indices = [i for i, label in enumerate(train_labels) if label == 0]

    for _ in range(num_pairs // 2):
        group = random.choice([asd_indices, control_indices])
        pair = random.sample(group, 2)
        same_group_pairs.append((pair[0], pair[1], "same"))

    for _ in range(num_pairs // 2):
        pair = (random.choice(asd_indices), random.choice(control_indices))
        different_group_pairs.append((pair[0], pair[1], "different"))

    return same_group_pairs + different_group_pairs


def pattern_loss(
    attention_matrix: torch.Tensor,
    group_pairs: list,
    metric="euclidean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the pattern loss for the given pairs of samples.
    Args:
        attention_matrix (np.ndarray): Cross-attention matrix (e.g., 177x210).
        pairs (list): List of tuples containing pairs of indices and their group type (same or different).
        margin (float): Minimum gap to enforce between same and different group pairs.
        top_k (int): Number of top values to consider in the attention matrix.

    Returns:
        float: Total loss value.
    """
    same_pair_loss = torch.tensor(0.0).to(attention_matrix.device)
    diff_pair_loss = torch.tensor(0.0).to(attention_matrix.device)

    for pair in group_pairs:
        idx1, idx2, group_type = pair

        if group_type == "different":
            attention1 = attention_matrix[idx1].sum(dim=-1)
            attention2 = attention_matrix[idx2].sum(dim=-1)

            if metric == "L1":
                gap = torch.norm(attention1 - attention2, p=1)
            elif metric == "cosine":
                gap = 1 - F.cosine_similarity(attention1, attention2, dim=0)
            elif metric == "euclidean":
                gap = torch.norm(attention1 - attention2, p=2)

    return gap


def groupwise_attention_loss(attention_matrices, group_labels, k=10, margin=1.0):
    """
    Group-wise loss for cross-attention matrices.

    Args:
        attention_matrices (torch.Tensor): Cross-attention matrices of shape (B, n, m).
        group_labels (torch.Tensor): Tensor of shape (B,) containing group labels (0=control, 1=ASD).
        k (int): Number of top values to consider in the attention matrix.
        margin (float): Margin for dissimilarity between group means.

    Returns:
        torch.Tensor: Computed loss value.
    """
    # Separate attention matrices by group
    asd_attention = attention_matrices[group_labels == 1]
    control_attention = attention_matrices[group_labels == 0]

    # Compute group means
    asd_mean = asd_attention.mean(dim=0)
    control_mean = control_attention.mean(dim=0)

    # Compute similarity loss within each group
    asd_loss = torch.norm(asd_attention - asd_mean, dim=(1, 2)).mean()
    control_loss = torch.norm(control_attention - control_mean, dim=(1, 2)).mean()

    # Compute dissimilarity loss between groups
    inter_group_loss = max(0, margin - torch.norm(asd_mean - control_mean))

    # Combine losses
    total_loss = asd_loss + control_loss + inter_group_loss
    return total_loss


# Code is adpated from: https://github.com/wangz10/contrastive_loss/blob/master/losses.py#L50
def pdist_euclidean(z):
    """
    Computes pairwise Euclidean distance matrix.
    """
    dist_matrix = torch.cdist(z, z, p=2)
    return dist_matrix


def square_to_vec(D):
    """
    Converts a square distance matrix to a vector form.
    """
    return D[torch.triu_indices(D.size(0), D.size(1), offset=1).unbind()]


def get_contrast_batch_labels(y):
    """
    Generates contrastive labels for pairs.
    """
    if y.dim() > 1:
        y = y.squeeze()  # Ensure y is 1D

    y_i, y_j = torch.meshgrid(y, y, indexing="ij")
    return (y_i == y_j).float()[
        torch.triu_indices(y.size(0), y.size(0), offset=1).unbind()
    ]


def max_margin_contrastive_loss(
    z, y, margin=1.0, metric="euclidean", if_matrix=False, tau=0.07
):
    z = z.view(z.size(0), -1)  # Flatten to [bsz, n_features * m_features]
    y = y.view(-1)

    if metric == "euclidean":
        D = pdist_euclidean(z)
    elif metric == "cosine":
        D = 1 - torch.mm(F.normalize(z, p=2, dim=1), F.normalize(z, p=2, dim=1).T) / tau
    else:
        raise ValueError("Unsupported metric")

    d_vec = square_to_vec(D)
    y_contrasts = get_contrast_batch_labels(y)

    loss = (
        y_contrasts * d_vec.pow(2) + (1 - y_contrasts) * F.relu(margin - d_vec).pow(2)
    ).mean()

    return loss


# some code is modified from https://github.com/rongzhou7/MCLCA/blob/main/MCLCA.py#L9
def full_contrastive_loss(z_alpha, z_beta, tau=0.07, lambda_param=0.5):
    """
    Compute the full contrastive loss considering all negative samples explicitly,
    without normalizing by batch size.
    """
    # Normalize the embedding vectors
    z_alpha_norm = F.normalize(z_alpha, p=2, dim=1)
    z_beta_norm = F.normalize(z_beta, p=2, dim=1)

    # Calculate the cosine similarity matrix
    sim_matrix = torch.mm(z_alpha_norm, z_beta_norm.t()) / tau
    # Extract similarities of positive pairs (same index pairs)
    positive_examples = torch.diag(sim_matrix)
    # Apply exponential to the similarity matrix for negative pairs handling
    exp_sim_matrix = torch.exp(sim_matrix)
    # Create a mask to zero out positive pair contributions in negative pairs sum
    mask = torch.eye(z_alpha.size(0)).to(z_alpha.device)
    exp_sim_matrix -= mask * exp_sim_matrix
    # Sum up the exponentiated similarities for negative pairs
    negative_sum = torch.sum(exp_sim_matrix, dim=1)

    # Calculate the contrastive loss for one direction (alpha as anchor)
    L_alpha_beta = -torch.sum(torch.log(positive_examples / negative_sum))

    # Repeat the steps for the other direction (beta as anchor)
    sim_matrix_T = sim_matrix.t()
    positive_examples_T = torch.diag(sim_matrix_T)
    exp_sim_matrix_T = torch.exp(sim_matrix_T)
    exp_sim_matrix_T -= mask * exp_sim_matrix_T
    negative_sum_T = torch.sum(exp_sim_matrix_T, dim=1)
    L_beta_alpha = -torch.sum(torch.log(positive_examples_T / negative_sum_T))

    # Combine the losses from both directions, balanced by lambda
    loss = lambda_param * L_alpha_beta + (1 - lambda_param) * L_beta_alpha
    return loss  # Return the unnormalized total loss


def contrastive_loss(z_alpha, z_beta, lambda_param, tau=0.07):
    """
        Compute the contrastive loss L_cont(α, β) for two sets of embeddings.

        Parameters:
        - z_alpha: Embeddings from modality α, tensor of shape (batch_size, embedding_size)
        - z_beta: Embeddings from modality β, tensor of shape (batch_size, embedding_size)
        - tau: Temperature parameter for scaling the cosine similarity
        - lambda_param: Weighting parameter to balance the loss terms
    x
        Returns:
        - loss: The computed contrastive loss
    """

    # Compute the cosine similarity matrix
    sim_matrix = (
        torch.mm(F.normalize(z_alpha, p=2, dim=1), F.normalize(z_beta, p=2, dim=1).t())
        / tau
    )
    # Diagonal elements are positive examples
    positive_examples = torch.diag(sim_matrix)
    # Compute the log-sum-exp for the denominator
    sum_exp = torch.logsumexp(sim_matrix, dim=1)

    # Loss for one direction (α anchoring and contrasting β)
    L_alpha_beta = -torch.mean(positive_examples - sum_exp)

    # Loss for the other direction (β anchoring and contrasting α)
    L_beta_alpha = -torch.mean(
        torch.diag(
            torch.mm(
                F.normalize(z_beta, p=2, dim=1), F.normalize(z_alpha, p=2, dim=1).t()
            )
            / tau
        )
        - torch.logsumexp(sim_matrix.t(), dim=1)
    )

    # Combined loss, balanced by lambda
    loss = lambda_param * L_alpha_beta + (1 - lambda_param) * L_beta_alpha

    return loss
