import math
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from pathlib import Path
from typing import Optional
from contextlib import nullcontext

import torch
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
    create_block_mask,
    or_masks,
    and_masks,
)

# TODO This was moved on nightly, this enables 2.5 and 2.6 | we should remove this once 2.5 is no longer supported
try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature],
    mask_mod: Optional[_mask_mod_signature],
    device: str = "cuda",
    _compile: bool = False,
    scale: Optional[float] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    B = 1
    H = 1
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device) + batch_idx
    h = torch.arange(0, H, device=device) + head_idx
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if type == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if type == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)
        scores *= scale_factor
        scores = scores.view(1, 1, M, N)
        if type == _ModificationType.SCORE_MOD:
            out = mod(scores, b, h, m, n)
        else:
            out = mod(b, h, m, n)

    return out


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Optional[Path] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: Optional[float] = None,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

    Returns:
        None
    """
    assert score_mod is not None or mask_mod is not None, (
        "Must provide either score_mod or mask_mod"
    )
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )
    # If both score_mod and mask_mod are provided, apply both
    if score_mod is not None and mask_mod is not None:
        mask_viz = create_score_mod(
            query,
            key,
            score_mod=None,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )
        # Apply mask by setting masked positions to -inf
        scores_viz = torch.where(mask_viz == 0, float("-inf"), scores_viz)

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10))
    color = "viridis" if score_mod is not None else "cividis"
    if score_mod is not None and mask_mod is not None:
        color = "plasma"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved as {file_path}")


def create_causal_mask_sdpa(seq_len):
    """
    Create a causal mask for self-attention. same as is_causal=True in torch.

    Args:
    - seq_len (int): Total sequence length
    Returns:
    - mask (torch.Tensor): Attention mask of shape (seq_len, seq_len)
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)


def create_block_causal_mask_sdpa(seq_len, block_size):
    """
    Create a block diagonal mask combined with a causal mask for self-attention.

    Args:
    - seq_len (int): Total sequence length
    - block_size (int): Size of each attention block

    Returns:
    - mask (torch.Tensor): Attention mask of shape (seq_len, seq_len)
    """
    # Block mask (bool tensor)
    block_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(0, seq_len, block_size):
        block_mask[i:i+block_size, i:i+block_size] = True  # Allow intra-block attention

    # Causal mask (bool tensor)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # Combine using logical OR
    combined_mask = block_mask | causal_mask  # Result is a bool tensor
    return combined_mask

def create_block_cross_attn_causal_mask_sdpa(nT, blocksize, blocksize1):
    """
    Create a block diagonal mask combined with a causal mask for self-attention.

    Args:
    - block_size (int): Size of each attention block

    Returns:
    - mask (torch.Tensor): Attention mask of shape (seq_len, seq_len)
    """
    # Block mask (bool tensor)
    seq_len = nT * blocksize
    seq_len1 = nT * blocksize1
    block_mask = torch.zeros(seq_len, seq_len1, dtype=torch.bool)
    for i in range(nT):
        for j in range(i + 1):
            block_mask[i*blocksize:(i+1)*blocksize, j*blocksize1:(j+1)*blocksize1] = True

    # Combine using logical OR
    combined_mask = block_mask.bool()
    return combined_mask

def create_causal_mask_flex(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def create_block_causal_mask_flex(block_size: int) -> _mask_mod_signature:
    """
    Create a block diagonal mask combined with a causal mask for FlexAttention.

    Args:
    - block_size (int): Size of each attention block

    Returns:
    - A callable mask function for FlexAttention
    """
    def mask_fn(b, h, q_idx, kv_idx):
        # Block attention within each block
        in_same_block = (q_idx // block_size) == (kv_idx // block_size)
        # Causal mask: allow only attending to past and current tokens
        is_causal = q_idx >= kv_idx
        return in_same_block | is_causal  # Logical OR for combining masks

    return mask_fn

def create_block_causal_in_context_mask_flex(block_size: int, ori_seq_len: int) -> _mask_mod_signature:
    """
    Create a block diagonal mask combined with a causal mask for FlexAttention.

    Args:
    - block_size (int): Size of each attention block

    Returns:
    - A callable mask function for FlexAttention
    """
    def mask_fn(b, h, q_idx, kv_idx):
        # Block attention within each block
        in_same_block = (q_idx // block_size) == (kv_idx // block_size)
        # Causal mask: allow only attending to past and current tokens
        is_causal = q_idx >= kv_idx

        # postfix
        is_postfix = q_idx >= ori_seq_len
        post_fix_causal = (kv_idx >= ori_seq_len) & ((kv_idx - ori_seq_len) <= (q_idx // block_size))
        return in_same_block | is_causal | is_postfix | post_fix_causal  # Logical OR for combining masks

    return mask_fn


def create_prefixlm_causal_mask_flex(prefix_length: int) -> _mask_mod_signature:
    """Generates a prefix LM causal attention mask.

    Args:
        prefix_length: The length of the prefix.

    Note:
        This mask allows full attention within the prefix (first PREFIX_LENGTH tokens)
        and causal attention for the rest of the sequence.
    """

    def prefix_mask(b, h, q_idx, kv_idx):
        return kv_idx < prefix_length

    prefixlm_causal_mask = or_masks(prefix_mask, create_causal_mask_flex)
    prefixlm_causal_mask.__name__ = f"prefixlm_causal_mask_{prefix_length}"
    return prefixlm_causal_mask


def create_sliding_window_mask_flex(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size

    sliding_window_mask = and_masks(sliding_window, create_causal_mask_flex)
    sliding_window_mask.__name__ = f"sliding_window_mask_{window_size}"
    return sliding_window_mask


def create_cycle_forcing_block_mask_flex(nT: int, block_size: int, p: int):
    """
    Create the cycle-forcing mask where GT frames are repeated based on p.

    Logic:
    - The last p frames of GT are stretched to cover the nT slots.
    - Each GT frame is repeated (nT // p) times.
    - Visibility is determined by the identity of the GT frame.
      If a slot contains GT_k, it sees Rollout[0...k-1].

    Args:
        nT: Total number of frames.
        block_size: Tokens per frame.
        p: Number of distinct GT frames to use (must divide nT).
           p=nT: GT1, GT2, GT3, GT4
           p=2 : GT3, GT3, GT4, GT4
           p=1 : GT4, GT4, GT4, GT4
    """
    assert nT % p == 0, f"nT ({nT}) must be divisible by p ({p})"

    repeats = nT // p
    start_gt_idx = nT - p  # 0-based index of the first GT frame to use

    def mask_fn(b, h, q_idx, kv_idx):
        q_frame = q_idx // block_size
        kv_frame = kv_idx // block_size

        # --- 1. Rollout Region (Standard) ---
        # Frame indices: 0 to nT-1
        is_last_rollout = (q_frame == (nT - 1))

        q_in_rollout = q_frame < (nT - 1)
        kv_in_rollout = kv_frame < nT

        # Rollout sees history (causal)
        rollout_can_see = q_in_rollout & kv_in_rollout & (kv_frame <= q_frame)

        # --- 2. GT Region (Repeated Logic) ---
        # Frame indices: nT to 2*nT - 1
        q_in_gt = q_frame >= nT

        # Calculate which slot in the GT sequence we are in (0 to nT-1)
        gt_slot_idx = q_frame - nT

        # Map slot to the actual Frame ID (0-based)
        # e.g., if nT=4, p=2: slots 0,1 -> frame 2 (GT3); slots 2,3 -> frame 3 (GT4)
        effective_frame_id = start_gt_idx + (gt_slot_idx // repeats)

        # Condition A: GT sees Rollout
        # If effective_frame_id is k, it sees Rollouts 0 to k-1.
        # So kv_frame must be < effective_frame_id
        gt_see_rollout = q_in_gt & kv_in_rollout & (kv_frame < effective_frame_id)

        # Condition B: GT sees Self (Strictly Diagonal)
        # Even if GT3 is repeated, slot i only sees slot i
        gt_see_self = q_in_gt & (kv_frame == q_frame)

        mask = (~is_last_rollout) & (rollout_can_see | gt_see_rollout | gt_see_self)

        return mask

    return mask_fn

def create_cycle_forcing_block_mask_sdpa(nT: int, block_size: int, p: int, device: str = "cuda"):
    """
    Tensor version (SDPA) for verification.
    """
    assert nT % p == 0, "nT must be divisible by p"

    seq_len = 2 * nT * block_size
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    repeats = nT // p
    start_gt_idx = nT - p # 0-based

    for q_idx in range(seq_len):
        q_frame = q_idx // block_size

        # Skip last rollout frame (masked)
        if q_frame == (nT - 1):
            continue

        # Rollout Section
        if q_frame < (nT - 1):
            rollout_end = (q_frame + 1) * block_size
            mask[q_idx, :rollout_end] = True

        # GT Section
        elif q_frame >= nT:
            gt_slot_idx = q_frame - nT

            # Determine identity: which GT frame is this?
            effective_frame_id = start_gt_idx + (gt_slot_idx // repeats)

            # 1. See Rollout History (0 ... effective_frame_id - 1)
            # This works because effective_frame_id is 0-based index,
            # so it equals the count of previous frames.
            if effective_frame_id > 0:
                rollout_end = effective_frame_id * block_size
                mask[q_idx, :rollout_end] = True

            # 2. See Self (Strictly current frame)
            frame_start = q_frame * block_size
            frame_end = frame_start + block_size
            mask[q_idx, frame_start:frame_end] = True

    return mask

def visualize_square_mask_with_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    nT: int,
    block_size: int,
    mask_mod: Optional[_mask_mod_signature] = None,
    mask_tensor: Optional[torch.Tensor] = None,
    device: str = "cuda",
    name: str = "cycle_forcing_square_mask",
    path: Optional[Path] = None,
    show_token_labels: bool = False,
):
    """
    Visualize the square cycle-forcing mask with block boundaries.

    Args:
        query: (B, H, seq_len, HEAD_DIM)
        key: (B, H, seq_len, HEAD_DIM)
        nT: Number of frames
        block_size: Tokens per frame
        mask_mod: Mask function for FlexAttention
        mask_tensor: Pre-computed mask tensor (for SDPA)
        device: Device to use
        name: Output filename
        path: Output path
        show_token_labels: Show individual token labels
    """

    # Use pre-computed tensor if provided, otherwise generate from mask_mod
    if mask_tensor is not None:
        scores_viz = mask_tensor.float().unsqueeze(0).unsqueeze(0)
    else:
        # For FlexAttention mask_mod, manually generate the mask
        seq_len = query.shape[2]
        mask_matrix = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        for q_idx in range(seq_len):
            for kv_idx in range(seq_len):
                # Call mask_mod with scalar indices (b=0, h=0)
                mask_matrix[q_idx, kv_idx] = mask_mod(
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(q_idx),
                    torch.tensor(kv_idx)
                )

        scores_viz = mask_matrix.float().unsqueeze(0).unsqueeze(0)

    seq_len = 2 * nT * block_size
    rollout_len = nT * block_size

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap="cividis")
    fig.colorbar(im, label="Attention Mask (1=attend, 0=mask)")

    ax.set_title(f"Cycle-Forcing Square Mask (nT={nT}, block_size={block_size})", fontsize=20)
    ax.set_xlabel("Key Tokens (Rollout + GT)", fontsize=18)
    ax.set_ylabel("Query Tokens (Rollout + GT)", fontsize=18)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Token-level grid
    token_boundaries = np.arange(-0.5, seq_len, 1)
    ax.set_xticks(token_boundaries, minor=True)
    ax.set_yticks(token_boundaries, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.5)

    # Block-level grid (red lines)
    block_boundaries = [i * block_size - 0.5 for i in range(2 * nT + 1)]
    for pos in block_boundaries:
        ax.axvline(x=pos, color="red", linestyle="-", linewidth=2.5)
        ax.axhline(y=pos, color="red", linestyle="-", linewidth=2.5)

    # Rollout/GT separator (blue line)
    separator_pos = rollout_len - 0.5
    ax.axvline(x=separator_pos, color="blue", linestyle="--", linewidth=3, label="Rollout/GT boundary")
    ax.axhline(y=separator_pos, color="blue", linestyle="--", linewidth=3)

    # Labels
    if show_token_labels and seq_len <= 32:
        labels = []
        for i in range(nT):
            for j in range(block_size):
                labels.append(f"R{i+1}_{j}")
        for i in range(nT):
            for j in range(block_size):
                labels.append(f"GT{i+1}_{j}")

        ax.set_xticks(range(seq_len), minor=False)
        ax.set_yticks(range(seq_len), minor=False)
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        centers = [(i * block_size + (i + 1) * block_size - 1) / 2 for i in range(2 * nT)]
        labels = [f"R{i+1}" for i in range(nT)] + [f"GT{i+1}" for i in range(nT)]

        ax.set_xticks(centers, minor=False)
        ax.set_yticks(centers, minor=False)
        ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    file_path = Path(name).with_suffix(".png") if path is None else path
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Visualization saved as {file_path}")


# ========== Test Code ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Test: Cycle-Forcing Square Mask with Block-wise Attention")
    print("="*60)

    nT = 8
    block_size = 4
    p = 8
    SEQ_LEN = 2 * nT * block_size
    B, H, HEAD_DIM = 1, 1, 8
    device = "cuda"

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    # Test SDPA version
    print("\n📊 Testing SDPA version...")
    mask_sdpa = create_cycle_forcing_block_mask_sdpa(nT, block_size, p, device=device)
    print(f"Mask shape: {mask_sdpa.shape}")

    visualize_square_mask_with_blocks(
        query, key,
        nT=nT,
        block_size=block_size,
        mask_tensor=mask_sdpa,
        device=device,
        name="create_cycle_forcing_block_mask_sdpa",
        show_token_labels=False
    )

    # Test FlexAttention version
    print("\n📊 Testing FlexAttention version...")
    mask_flex = create_cycle_forcing_block_mask_flex(nT, block_size, p)

    visualize_square_mask_with_blocks(
        query, key,
        nT=nT,
        block_size=block_size,
        mask_mod=mask_flex,
        device=device,
        name="create_cycle_forcing_block_mask_flex",
        show_token_labels=False
    )


# if __name__ == "__main__":

#     # prepare for visualization
#     B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8
#     # B, H, SEQ_LEN, HEAD_DIM = 1, 1, 15, 8
#     device = "cuda"
#     def make_tensor():
#         return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)
#     query, key = make_tensor(), make_tensor()

#     # test create_causal_mask_flex
#     #visualize_attention_scores(query, key, mask_mod=create_causal_mask_flex, device=device, name="causal_mask")

#     # test create_block_causal_mask_flex
#     # block_causal_mask = create_block_causal_in_context_mask_flex(4, 12)
#     block_causal_mask = create_block_causal_mask_flex(3)
#     visualize_attention_scores(
#         query, key, mask_mod=block_causal_mask, device=device, name="block_causal_mask"
#     )

#     '''
#     # test create_prefixlm_causal_mask_flex
#     prefixlm_causal_mask = create_prefixlm_causal_mask_flex(4)
#     visualize_attention_scores(
#         query, key, mask_mod=prefixlm_causal_mask, device=device, name="prefixlm_causal_mask_4"
#     )

#     # test create_sliding_window_mask_flex
#     sliding_window_mask = create_sliding_window_mask_flex(3)
#     visualize_attention_scores(
#         query, key, mask_mod=sliding_window_mask, device=device, name="sliding_window_mask"
#     )
#     '''
