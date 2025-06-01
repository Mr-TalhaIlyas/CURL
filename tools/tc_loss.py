import math
import sys

import torch
import torch.nn as nn
import numpy as np

def KMeans(x, K=10, Niters=10, verbose=False):
    N, D = x.shape

    if N == 0:
        # Return empty assignments and zero centroids if input is empty
        # This case should ideally be caught before calling KMeans if K > 0
        return torch.empty((0,), dtype=torch.long, device=x.device), \
               torch.zeros((K, D), dtype=x.dtype, device=x.device)

    # The calling function (temporal_contrastive_loss) will ensure N >= K.
    # If N < K, x[:K, :] would error.
    
    _device = x.device
    c = x[:K, :].clone()  # Initialize centroids from the first K samples
    x_i = x[:, None, :]  # Reshape x for broadcasting

    for i in range(Niters):
        c_j = c[None, :, :]  # Reshape centroids for broadcasting

        D_ij = ((x_i - c_j) ** 2).sum(-1)  # Calculate squared Euclidean distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Assign samples to closest centroids

        # Update centroids
        c_new = torch.zeros((K, D), dtype=x.dtype, device=_device)
        # Sum features for each cluster
        c_new.scatter_add_(0, cl.unsqueeze(1).expand(-1, D), x)

        # Count samples in each cluster
        counts = torch.bincount(cl, minlength=K).float().to(_device)
        
        # Avoid division by zero for empty clusters (centroid becomes 0)
        non_zero_counts = counts.unsqueeze(1).clamp(min=1.0) # Shape [K, 1]
        c = c_new / non_zero_counts
        
        if verbose and i % (Niters // 10 if Niters >=10 else 1) == 0:
            print(f"KMeans iter {i}: counts {counts.cpu().numpy()}")


    return cl, c

def IID_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    """
    This code is from IIC github
    """

    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss

def _calculate_cld_and_softmax_affinity(features, centroids, cluster_labels, T, criterion, softmax, num_clusters, feature_name_for_log=""):
    """
    Helper function to calculate CLD loss for one direction and softmax of raw affinity for IID.
    Returns: (cld_loss, softmax_affinity_for_iid, error_occurred)
    """
    if features.shape[0] == 0: # Should be caught by main function's checks
        # This case implies no data to compute loss for this path.
        # Return 0 loss and empty affinity, no error.
        return torch.tensor(0.0, device=features.device, requires_grad=True), \
               torch.empty((0, num_clusters), device=features.device), \
               False

    affinity_raw = torch.mm(features, centroids.t())

    if torch.isnan(affinity_raw).any():
        print(f"Warning: NaN detected in raw affinity calculation for {feature_name_for_log}. Centroids might be problematic.")
        return None, None, True # error_occurred = True

    # Validate cluster_labels (targets for CrossEntropyLoss)
    # cluster_labels come from KMeans on the *other* feature set (K=num_clusters)
    # So, their values should be in [0, num_clusters-1]. centroids.shape[0] is num_clusters.
    if cluster_labels.numel() > 0: # Only check if there are labels
        if cluster_labels.min() < 0 or cluster_labels.max() >= centroids.shape[0]:
            print(f"Warning: Invalid cluster labels for {feature_name_for_log}. Min: {cluster_labels.min()}, Max: {cluster_labels.max()}, Num Clusters: {centroids.shape[0]}")
            return None, None, True # error_occurred = True
    elif features.shape[0] > 0 : # features exist, but no labels (e.g. other feature set was empty)
        print(f"Warning: Features present for {feature_name_for_log} but no cluster labels provided (likely other feature set was empty).")
        return None, None, True # error_occurred = True


    loss_cld = criterion(affinity_raw / T, cluster_labels)
    softmax_affinity_for_iid = softmax(affinity_raw) # IID uses softmax on raw affinities (before T scaling)

    return loss_cld, softmax_affinity_for_iid, False # No error

def compute_joint(x_out, x_tf_out):
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def temporal_contrastive_loss(features1, features2, T, args):
    """
    Temporal Contrastive Loss based on Cross-Level Distillation (CLD) and IID regularization.
    Handles features from sequences like ultrasound videos.
    """
    _device = features1.device if isinstance(features1, torch.Tensor) else \
              (features2.device if isinstance(features2, torch.Tensor) else args.default_device)

    # --- Input Validations ---
    if not isinstance(features1, torch.Tensor) or not isinstance(features2, torch.Tensor):
        print("Warning: features1 and features2 must be PyTorch Tensors. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)
    if features1.ndim != 2 or features2.ndim != 2:
        print("Warning: features1 and features2 must be 2D Tensors (batch_size, feature_dim). Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)
    if features1.shape[0] > 0 and features2.shape[0] > 0 and features1.shape[1] != features2.shape[1]:
        print("Warning: Feature dimensions of features1 and features2 must match. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)
    
    if torch.isnan(features1).any() or torch.isinf(features1).any() or \
       torch.isnan(features2).any() or torch.isinf(features2).any():
        print("Warning: NaN/Inf detected in input features. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)

    if args.clusters <= 0:
        print(f"Error: args.clusters must be positive, got {args.clusters}. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)
    if args.num_iters <= 0:
        print(f"Error: args.num_iters for KMeans must be positive, got {args.num_iters}. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)
    if T <= 1e-8: # Temperature should be strictly positive
        print(f"Error: Temperature T must be positive, got {T}. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)

    # Handle cases with insufficient samples for KMeans or empty features
    f1_empty = features1.shape[0] == 0
    f2_empty = features2.shape[0] == 0
    
    if f1_empty and f2_empty:
        # print("Debug: Both feature sets empty. Returning 0 loss.")
        return torch.tensor(0.0, device=_device, requires_grad=True)

    # If one is empty, CLD is not well-defined in its symmetric form.
    # Depending on desired behavior, could return 0 or NaN.
    # For now, if one set is empty, we can't form meaningful cross-cluster assignments.
    if f1_empty or f2_empty:
        # print(f"Warning: One feature set is empty (f1_empty: {f1_empty}, f2_empty: {f2_empty}). Returning 0 loss as no contrast can be made.")
        return torch.tensor(0.0, device=_device, requires_grad=True)

    if features1.shape[0] < args.clusters or features2.shape[0] < args.clusters:
        print(f"Warning: Not enough samples for KMeans (K={args.clusters}, N1={features1.shape[0]}, N2={features2.shape[0]}). "
              "KMeans requires N >= K. Returning 0 loss.")
        return torch.tensor(0.0, device=_device, requires_grad=True)

    # --- Initialization ---
    criterion = nn.CrossEntropyLoss().to(_device)
    softmax = nn.Softmax(dim=1).to(_device)

    # --- K-Means Clustering ---
    try:
        cluster_label1, centroids1 = KMeans(features1, K=args.clusters, Niters=args.num_iters)
        cluster_label2, centroids2 = KMeans(features2, K=args.clusters, Niters=args.num_iters)
    except Exception as e:
        print(f"Error during KMeans: {e}. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)

    # --- CLD and IID Affinity Calculation ---
    cld_loss1, affinity1_softmax_iid, err1 = _calculate_cld_and_softmax_affinity(
        features1, centroids2, cluster_label2, T, criterion, softmax, args.clusters, "features1 vs centroids2"
    )
    cld_loss2, affinity2_softmax_iid, err2 = _calculate_cld_and_softmax_affinity(
        features2, centroids1, cluster_label1, T, criterion, softmax, args.clusters, "features2 vs centroids1"
    )

    if err1 or err2:
        print("Error in calculating CLD components. Returning NaN loss.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)

    cld_loss = (cld_loss1 + cld_loss2) / 2.0

    # --- IID Loss ---
    iid_loss_val = torch.tensor(0.0, device=_device) # Default to 0 if not calculated
    if args.do_entro:
        # Ensure affinities for IID loss are not empty (can happen if one feature set was empty, though handled above)
        if affinity1_softmax_iid.shape[0] > 0 and affinity2_softmax_iid.shape[0] > 0 :
            iid_loss_val = IID_loss(affinity1_softmax_iid, affinity2_softmax_iid)
            if torch.isnan(iid_loss_val) or torch.isinf(iid_loss_val):
                print("Warning: IID loss is NaN or Inf. Setting to 0 for this component.")
                iid_loss_val = torch.tensor(0.0, device=_device)
        else:
            # This case should ideally not be reached due to earlier checks for empty features1/2
            print("Warning: Skipping IID loss due to empty softmax affinities (should have been caught earlier).")


    # --- Combine Losses ---
    total_loss = cld_loss + iid_loss_val if args.do_entro else cld_loss
        
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: Final temporal_contrastive_loss is NaN or Inf. Returning NaN.")
        return torch.tensor(float('nan'), device=_device, requires_grad=True)

    return total_loss
    