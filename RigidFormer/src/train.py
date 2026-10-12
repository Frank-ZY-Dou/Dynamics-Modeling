"""
Main training script for RigidFormer, a multi-body rigid dynamics model.

The network predicts per-anchor accelerations, which are integrated with a
Verlet scheme and projected onto rigid motions via Kabsch alignment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random
import subprocess
import json

try:
    import roma  # RoMa library for differentiable rigid registration
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    roma = None


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.multi_dataloader import create_multi_dataloaders
from src.models.encoder import RigidBodyEncoder, EncoderPack
from src.models.state import PersistentStateManager
from src.models.decoder import BidirectionalDecoder
from src.models.predictor import DynamicsPredictor
from src.models.transform_layer import create_transform_layer

from src.utils.intersection_handler import handle_intersections
from src.models.physformer_rope import rope_memo_reset
from src.utils.weight_ema import WeightEMA

# Require CUDA KNN (no fallback). If unavailable, raise immediately.
try:
    from src.utils.cuda_knn import compute_nearest_distance_to_list_cuda, compute_knn_all_objects_vectorized, \
        knn_pack, compute_knn_packed
except Exception as e:
    raise RuntimeError(
        "CUDA KNN is required but not available. Please build/enable src.utils.cuda_knn."
    ) from e
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Per-step NaN/Inf diagnostics inside the training forward (each costs host syncs). They only log, so they are
# opt-in: DEBUG_NAN=1. Non-finite losses are still caught by the training loop.
DEBUG_NAN = os.environ.get('DEBUG_NAN', '0') == '1'


def distance_2_plane(points, plane_point, plane_axis, body_dist_vec):
    """
    Check if points are closer to a plane than to another body, and update distance vectors.
    
    Args:
        points: Tensor of shape (B, M, 3) containing point positions
        plane_point: List [x, y, z] defining a point on the plane (e.g., [0,0,0])
        plane_axis: String "x", "y", or "z" defining the plane normal direction
        body_dist_vec: Tensor of shape (B, M, 3) containing vectors to nearest points on other body
    
    Returns:
        Updated distance vectors (B, M, 3) where vectors to plane replace body vectors when closer
    """
    B, M, _ = points.shape
    device = points.device
    
    # Get the axis index
    axis_map = {"x": 0, "y": 1, "z": 2}
    if plane_axis not in axis_map:
        raise ValueError(f"Invalid plane_axis: {plane_axis}. Must be 'x', 'y', or 'z'")
    axis_idx = axis_map[plane_axis]
    
    # Get plane position on the specified axis
    plane_pos = plane_point[axis_idx]
    
    # Calculate distance to plane for each point
    # Distance is just the absolute difference in the specified axis
    point_axis_coords = points[:, :, axis_idx]  # (B, M)
    dist_to_plane = torch.abs(point_axis_coords - plane_pos)  # (B, M)
    
    # Calculate distance from body vectors (norm of the vectors)
    dist_to_body = torch.norm(body_dist_vec, dim=2)  # (B, M)
    
    # Create vectors from points to plane
    # The vector points from the point to its projection on the plane
    plane_vectors = torch.zeros_like(points)  # (B, M, 3)
    plane_vectors[:, :, axis_idx] = plane_pos - point_axis_coords  # Only non-zero in the plane normal direction
    
    # Create mask for points closer to plane than to body
    closer_to_plane = dist_to_plane < dist_to_body  # (B, M)
    
    # Expand mask to match vector dimensions
    closer_to_plane_expanded = closer_to_plane.unsqueeze(-1).expand_as(body_dist_vec)  # (B, M, 3)
    
    # Update vectors: use plane vectors where closer to plane, otherwise keep body vectors
    updated_vectors = torch.where(closer_to_plane_expanded, plane_vectors, body_dist_vec)

    return updated_vectors


def _compute_3d_grid(n):
    """Compute 3D grid dimensions for n tokens (same logic as StateRepresentation)."""
    cube_root = int(round(n ** (1/3)))
    for x in range(cube_root, 0, -1):
        if n % x == 0:
            remaining = n // x
            sqrt_rem = int(round(remaining ** 0.5))
            for y in range(sqrt_rem, 0, -1):
                if remaining % y == 0:
                    z = remaining // y
                    return (x, y, z)
    return (n, 1, 1)


def _adapt_state_tokens_if_needed(state_dict, model, logger):
    """
    Adapt state tokens in checkpoint if size differs from current model.
    Uses 3D trilinear interpolation to preserve learned spatial patterns.

    Args:
        state_dict: Checkpoint state dict
        model: Current model (may be wrapped in DDP)
        logger: Logger instance

    Returns:
        Adapted state_dict
    """
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Get the actual model (unwrap DDP if needed)
    model_ref = model.module if isinstance(model, DDP) else model

    # If model has no state_repr (no_global_state=true), remove state keys from checkpoint
    if model_ref.state_repr is None:
        state_keys_to_remove = [k for k in state_dict if 'state_repr.' in k]
        if state_keys_to_remove:
            for k in state_keys_to_remove:
                del state_dict[k]
            logger.info(f"[Checkpoint Adapt] no_global_state=true: removed {len(state_keys_to_remove)} state_repr keys from checkpoint")
        return state_dict

    # Find state token keys (handle both DDP and non-DDP checkpoints)
    state_key_patterns = [
        'state_repr.register_tokens.weight',
        'module.state_repr.register_tokens.weight'
    ]

    # Determine the correct key based on what's in the state_dict
    ckpt_key = None
    for pattern in state_key_patterns:
        if pattern in state_dict:
            ckpt_key = pattern
            break

    if ckpt_key is None:
        logger.info("[Checkpoint Adapt] No state tokens found in checkpoint, skipping adaptation")
        return state_dict

    # Get checkpoint and model state token shapes
    ckpt_tokens = state_dict[ckpt_key]
    ckpt_num_tokens, ckpt_dim = ckpt_tokens.shape

    # Get model's expected shape
    model_num_tokens = model_ref.state_repr.num_tokens
    model_dim = model_ref.state_repr.token_dim

    # Check if adaptation is needed
    if ckpt_num_tokens == model_num_tokens and ckpt_dim == model_dim:
        logger.info(f"[Checkpoint Adapt] State tokens match: ({ckpt_num_tokens}, {ckpt_dim}), no adaptation needed")
        return state_dict

    if ckpt_dim != model_dim:
        logger.warning(f"[Checkpoint Adapt] State token_dim mismatch: checkpoint={ckpt_dim}, model={model_dim}")
        logger.warning("[Checkpoint Adapt] Cannot adapt token_dim, state tokens will be randomly initialized")
        # Remove the mismatched key so it will be randomly initialized
        del state_dict[ckpt_key]
        return state_dict

    # Adapt num_tokens using 3D interpolation
    logger.info(f"[Checkpoint Adapt] State tokens size changed: {ckpt_num_tokens} -> {model_num_tokens}")

    # Compute 3D grid dimensions
    ckpt_grid = _compute_3d_grid(ckpt_num_tokens)
    model_grid = model_ref.state_repr.grid_3d if hasattr(model_ref.state_repr, 'grid_3d') else _compute_3d_grid(model_num_tokens)

    logger.info(f"[Checkpoint Adapt] Grid: {ckpt_grid} -> {model_grid}")

    # Reshape to 3D grid: (N, D) -> (1, D, X, Y, Z)
    ckpt_3d = ckpt_tokens.T.reshape(1, ckpt_dim, *ckpt_grid)  # (1, D, X, Y, Z)

    # Trilinear interpolation to new grid size
    adapted_3d = F.interpolate(
        ckpt_3d.float(),
        size=model_grid,
        mode='trilinear',
        align_corners=True
    )  # (1, D, X', Y', Z')

    # Reshape back to (N', D)
    adapted_tokens = adapted_3d.reshape(ckpt_dim, model_num_tokens).T  # (N', D)

    # Convert back to original dtype
    adapted_tokens = adapted_tokens.to(ckpt_tokens.dtype)

    # Update state_dict
    state_dict[ckpt_key] = adapted_tokens

    logger.info(f"[Checkpoint Adapt] State tokens adapted: {ckpt_tokens.shape} -> {adapted_tokens.shape}")

    return state_dict



def _object_centered(ref_pose: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
    """ref_pose (B, Nobj, Nv, 3) minus each object's x/y centroid over valid vertices vmask (B, Nobj, Nv).

    Only x and y are centred: horizontal translation is a symmetry of the dynamics, height above
    the ground plane (z = 0) is not, so z is kept absolute."""
    m = vmask.to(ref_pose.dtype).unsqueeze(-1)
    c = (ref_pose * m).sum(dim=2, keepdim=True) / m.sum(dim=2, keepdim=True).clamp_min(1.0)
    c = torch.cat([c[..., :2], torch.zeros_like(c[..., 2:3])], dim=-1)
    return ref_pose - c


class CUT3RDynamicsModel(nn.Module):
    """
    Complete model combining encoder, state management, decoder, and predictor.
    """
    
    def __init__(self, config):
        super().__init__()

        # Store config for later use
        self.config = config

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Physics-parameter and time-step flags (read before submodule construction)
        # Generalization arm: anchor reference poses expressed relative to each object's
        # frame-0 centroid (removes absolute translation from the predictor query), and
        # rotary pairing that matches rotate_half so attention depends on displacement only.
        self._ref_pose_object_centered = bool(config['model'].get('ref_pose_object_centered', False))
        from src.models.physformer_rope import PhysFormerRoPE as _RopeCls
        _RopeCls.INTERLEAVED_PAIRING = bool(config['model'].get('rope_interleaved_pairing', False))

        self._if_phys_param = config.get('model', {}).get('if_phys_param', False)
        self._if_time_step = config.get('model', {}).get('if_time_step', False)
        self._film_conditioning = config.get('model', {}).get('film_conditioning', False)
        if self._if_phys_param:
            print(f"[Model] Physics parameters enabled: mass, friction, restitution (+3D to encoder/predictor)")
        if self._if_time_step:
            if self._film_conditioning:
                print(f"[Model] Time step enabled via FiLM only (NOT concatenated to encoder input)")
            else:
                print(f"[Model] Time step enabled: skip_step_size (s and s², +2D to encoder/predictor)")

        # Rollout state reset: reset state tokens at each step to avoid train-test mismatch
        # When training with short sequences (e.g., 4 frames), state tokens only see 2 updates.
        # During long rollouts (e.g., 400 frames), state drift causes OOD behavior.
        # Setting this to True resets state tokens at each rollout step to match training distribution.
        self._rollout_state_reset = config.get('model', {}).get('rollout_state_reset', False)
        if self._rollout_state_reset:
            print(f"[Model] Rollout state reset ENABLED: state tokens will be reset at each rollout step")

        # KNN acceleration: GPU-parallel grouped processing
        self._knn_num_groups = config.get('data', {}).get('knn_num_groups', 1)
        # Valid-vertex packing (model.knn_packed, model.encoder.packed): the distance features and the encoder's
        # pointwise layers run over the valid vertices only instead of every padded (object slot x vertex) entry.
        # Equal to the padded paths up to floating-point rounding; the packs are built once per batch / rollout.
        self._knn_packed = bool(config.get('model', {}).get('knn_packed', False))
        self._encoder_packed = bool(config.get('model', {}).get('encoder', {}).get('packed', False))
        if self._knn_packed or self._encoder_packed:
            print(f"[Model] valid-vertex packing: knn_packed={self._knn_packed}, encoder.packed={self._encoder_packed}")
        if self._knn_packed and self._knn_num_groups > 1:
            raise ValueError('model.knn_packed replaces data.knn_num_groups; set knn_num_groups to 1')
        # model.encoder.packed_pooling: 'segment' pools the packed rows directly (segment max / mean) instead of
        # scattering every level back to the padded layout; same values up to fp32 summation order in the mean.
        if (config.get('model', {}).get('encoder', {}).get('packed_pooling', 'scatter') != 'scatter'
                and not self._encoder_packed):
            raise ValueError('model.encoder.packed_pooling applies to the packed encoder path; set model.encoder.packed')
        if self._knn_num_groups > 1:
            print(f"[Model] KNN acceleration enabled: num_groups={self._knn_num_groups}")

        # model.kabsch_reference: the mesh the per-step Kabsch fit transforms. 'advancing' (default) = the
        # previous step's result, so every step re-transforms an already transformed mesh and rounding accumulates;
        # 'fixed' = one unchanged reference shape for the whole sequence / rollout (the first current frame), equal
        # in exact arithmetic (the fit is equivariant) without the accumulation.
        _kref = config.get('model', {}).get('kabsch_reference', 'advancing')
        if _kref not in ('fixed', 'advancing'):
            raise ValueError(f"model.kabsch_reference must be 'fixed' or 'advancing', got {_kref!r}")
        self._kabsch_reference_fixed = _kref == 'fixed'
        if self._kabsch_reference_fixed:
            print("[Model] Kabsch reference FIXED: every step fits the same reference mesh")

        # model.intersection_handling.after_kabsch_loss: which state the position-after-Kabsch loss supervises.
        # 'rigid' = the Kabsch output (the state the acceleration-after-Kabsch loss already uses), 'handled'
        # (default) = the state after the ground/object correction. The rollout state is the corrected one either way.
        _akl = config.get('model', {}).get('intersection_handling', {}).get('after_kabsch_loss', 'handled')
        if _akl not in ('rigid', 'handled'):
            raise ValueError(f"model.intersection_handling.after_kabsch_loss must be 'rigid' or 'handled', got {_akl!r}")
        self._after_kabsch_loss_rigid = _akl == 'rigid'
        if self._after_kabsch_loss_rigid:
            print("[Model] position-after-Kabsch loss supervises the Kabsch output (before intersection handling)")

        # Time-batched training forward (training.time_batched_forward): the network part of all teacher-forced
        # steps runs as one batch (see _forward_multi_object). Only valid when no step's network input depends on
        # an earlier step's prediction or state; the settings that would create such a dependency are refused.
        self._time_batched_forward = bool(config.get('training', {}).get('time_batched_forward', False))
        if self._time_batched_forward:
            _enc = config.get('model', {}).get('encoder', {})
            _bad = []
            if config.get('model', {}).get('if_penetration', config.get('model', {}).get('if_directional_distance', False)):
                _bad.append('model.if_penetration/if_directional_distance (uses the previous step\'s decoder output)')
            if not _enc.get('masked_pooling', False):
                _bad.append('model.encoder.masked_pooling=false (BatchNorm statistics would pool across steps)')
            if _enc.get('use_stn', False) or _enc.get('feature_transform', False):
                _bad.append('encoder STN / feature transform (their BatchNorms would pool across steps)')
            if _bad:
                raise ValueError("training.time_batched_forward is incompatible with: " + '; '.join(_bad))
            print("[Model] time-batched training forward ACTIVE (network part of all teacher-forced steps in one "
                  "batch; per-step BatchNorm statistics)", flush=True)

        # Debug mode: from data.random_step_size.if_debug or training.loss_weights.if_debug
        self._debug_mode = (
            config.get('data', {}).get('random_step_size', {}).get('if_debug', False) or
            config.get('training', {}).get('loss_weights', {}).get('if_debug', False)
        )

        # Config Consistency Check: Multi-timescale training requires time_step feature
        random_step_size_config = config.get('data', {}).get('random_step_size', {})
        if random_step_size_config.get('enable', False):
            if not self._if_time_step:
                raise ValueError(
                    "Config inconsistency detected!\n"
                    "  data.random_step_size.enable = True  (multi-timescale training enabled)\n"
                    "  model.if_time_step = False  (time step feature disabled)\n"
                    "\n"
                    "Multi-timescale training requires the network to receive step_size as input.\n"
                    "Please set model.if_time_step = true in your config file."
                )
            print(f"[Model] Multi-timescale training enabled: step_size will be used as condition feature")

        # Initialize components (multi-object only)

        # ============================================================
        # ARCHITECTURAL INVARIANTS — Translation Consistency & Permutation Equivariance
        # These properties are fundamental to the model. Enforce at init time.
        # ============================================================

        # 1. State tokens permanently removed — fixed 3D grid breaks translation consistency
        self._no_global_state = True
        self.state_repr = None
        assert config['model'].get('no_global_state', True) is True, \
            "no_global_state must be True: state tokens on a fixed 3D grid break translation consistency"

        # 2. rope_set_aggregation must be True — concat mode is anchor-order sensitive
        assert config['model']['decoder'].get('rope_set_aggregation', False) is True, \
            "rope_set_aggregation must be True: the concatenation mode is anchor-order sensitive, " \
            "breaking permutation invariance of ARoPE within each object"

        # 3. No sequence-index positional embeddings — would break object permutation equivariance
        assert not config['model']['decoder'].get('use_learnable_pe', False), \
            "use_learnable_pe must be False: sequence-index PE breaks object permutation equivariance"

        # ============================================================

        # Multi-scale predictor reads features from selected decoder layers (CUT3R-style)
        from src.models.multi_scale_predictor import QueryBasedMultiScalePredictor

        # Adjust layer indices based on actual number of decoder layers (needed before decoder init)
        num_decoder_layers = config['model']['decoder'].get('num_layers', 12)
        if num_decoder_layers >= 12:
            layer_indices = [0, 6, 9, 12]  # CUT3R's layers for 12-layer model
        elif num_decoder_layers >= 6:
            layer_indices = [0, 2, 4, num_decoder_layers]  # For 6+ layer model
        elif num_decoder_layers >= 4:
            # For 4-layer model
            layer_indices = [0, 1, 2, num_decoder_layers]  # Use all available layers
        else:
            # For 2-layer model (tiny), use all available layers
            layer_indices = list(range(num_decoder_layers + 1))  # [0, 1, 2] for 2 layers

        # Verify that all layer indices are valid for the decoder
        max_valid_index = num_decoder_layers  # 0 to num_decoder_layers inclusive (initial + N layers)
        for idx in layer_indices:
            assert 0 <= idx <= max_valid_index, \
                f"Invalid layer index {idx} for decoder with {num_decoder_layers} layers. " \
                f"Valid range: [0, {max_valid_index}]"

        # Store layer indices for decoder to only save needed layers
        self.predictor_layer_indices = set(layer_indices)  # Use set for O(1) lookup

        # Create mapping from original layer index to position in saved list
        # For example: if layer_indices = [0, 1, 2, 4], then:
        # layer_index_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
        self.layer_index_mapping = {idx: i for i, idx in enumerate(sorted(layer_indices))}

        # Gated Attention configuration (Qwen3-style)
        gated_attention_config = config['model']['decoder'].get('gated_attention', {})
        gated_attention_enable = gated_attention_config.get('enable', False)
        gated_attention_type = gated_attention_config.get('type', 'elementwise')
        gated_attention_debug = gated_attention_config.get('if_debug', False)
        # Currently only support elementwise gate
        if gated_attention_enable and gated_attention_type != 'elementwise':
            raise ValueError(f"Unsupported gated_attention type: {gated_attention_type}. Only 'elementwise' is supported.")
        self._elementwise_gate = gated_attention_enable and gated_attention_type == 'elementwise'

        # Debug print for gated attention (print to console since FileHandler not yet set)
        if gated_attention_debug:
            msg = (f"[Gated Attention] enable={gated_attention_enable}, type={gated_attention_type}, "
                   f"elementwise_gate={self._elementwise_gate}")
            print(msg)  # Console output (FileHandler not ready at model init time)
            logger.info(msg)  # Also log (will appear in file after Trainer sets FileHandler)

        # State update mode: when False, state is just learnable embeddings (no bidirectional update)
        # Use with rollout_state_reset=True to avoid train-test mismatch
        self._state_update_enabled = config['model']['decoder'].get('state_update_enabled', True)

        # Objects Self-Attention: Enable direct object-to-object interaction
        # Useful for physics simulation where objects need to exchange information
        self._objects_self_attention = config['model']['decoder'].get('objects_self_attention', False)

        # PE type must be read before decoder init (use_se3 flag needs it)
        self._pe_type = config['model']['decoder'].get('pe_type', 'arope')

        # Initialize decoder with layer_indices_to_save for memory efficiency
        # Block AttnRes config (Kimi 2026)
        _block_attn_res_cfg = config['model']['decoder'].get('block_attn_res', {})
        _block_attn_res_enable = _block_attn_res_cfg.get('enable', False) if isinstance(_block_attn_res_cfg, dict) else bool(_block_attn_res_cfg)
        _block_attn_res_block_size = _block_attn_res_cfg.get('block_size', 4) if isinstance(_block_attn_res_cfg, dict) else 4
        _block_attn_res_final = bool(_block_attn_res_cfg.get('final_aggregate', False)) if isinstance(_block_attn_res_cfg, dict) else False

        self.decoder = BidirectionalDecoder(
            state_dim=config['model']['state'].get('token_dim', 768),
            num_layers=config['model']['decoder'].get('num_layers', 12),  # CUT3R uses 12
            num_heads=config['model']['decoder'].get('num_heads', 12),
            feedforward_dim=config['model']['decoder'].get('feedforward_dim', 3072),
            dropout=config['model']['decoder']['dropout'],
            return_all_layers=True,  # For multi-scale fusion
            use_self_attention=config['model']['decoder'].get('use_self_attention', False),  # Optional state SA
            center_rope_positions=config['model'].get('rope_center_positions', False),
            rope_set_aggregation=config['model']['decoder'].get('rope_set_aggregation', False),  # Set aggregation for RoPE
            rope_dim=config['model']['decoder'].get('rope_dim', None),  # Custom RoPE dimension (None = auto-compute)
            dim_per_coord_3d=config['model']['decoder'].get('dim_per_coord_3d', None),  # Dim per coordinate for single-point (3D) RoPE
            dim_per_coord_multi=config['model']['decoder'].get('dim_per_coord_multi', None),  # Dim per coordinate for multi-point RoPE
            layer_indices_to_save=self.predictor_layer_indices,  # Only save layers needed by predictor
            elementwise_gate=self._elementwise_gate,  # Qwen3: elementwise gated attention
            state_update_enabled=self._state_update_enabled,  # When False, state is learnable embeddings only
            objects_self_attention=self._objects_self_attention,  # Enable Objects Self-Attention for object-object interaction
            use_se3=(self._pe_type == 'se3'),  # SE(3) pairwise bias for obj self-attention
            block_attn_res=_block_attn_res_enable,  # Block Attention Residuals (Kimi 2026)
            block_attn_res_block_size=_block_attn_res_block_size,  # Layers per block
            block_attn_res_final_aggregate=_block_attn_res_final,  # final depth aggregation as the last features
            film_conditioning=(config['model']['decoder'].get('film_conditioning', False) and self._film_conditioning and self._if_time_step),  # Per-layer FiLM: requires decoder.film_conditioning + model.film_conditioning + model.if_time_step
        )

        # ============ Anchor-Vertex Pooling (AVP) ============
        avp_cfg = config.get('model', {}).get('avp', {})
        self._avp_enabled = avp_cfg.get('enable', False)
        self._avp_output_dim = avp_cfg.get('output_dim', 256)
        if self._avp_enabled:
            from src.models.anchor_vertex_pooling import AnchorVertexPooling
            _avp_vertex_dim = config.get('model', {}).get('encoder', {}).get(
                'backbone_dim', config.get('model', {}).get('encoder', {}).get('output_dim', 1024))
            self.avp = AnchorVertexPooling(
                vertex_dim=_avp_vertex_dim,
                output_dim=self._avp_output_dim,
                init_sigma=avp_cfg.get('init_sigma', 0.1)
            )
            print(f"[Model] AVP ENABLED: vertex_dim={_avp_vertex_dim}, output_dim={self._avp_output_dim}, "
                  f"init_sigma={avp_cfg.get('init_sigma', 0.1)}")
        else:
            self.avp = None

        # Dynamic predictor input_dim calculation:
        # Base: 12D (9D query: dist_vec+vel+rel_pos + 3D ref_pose)
        # + phys_params (if if_phys_param=true): +3D = 15D
        # + time_step (if if_time_step=true AND NOT film): +2D = 17D
        # + AVP (if avp.enable=true): +avp_output_dim
        predictor_input_dim = 12  # Base: 9D query + 3D ref_pose
        if self._if_phys_param:
            predictor_input_dim += 3  # Physics params
        if self._if_time_step and not self._film_conditioning:
            predictor_input_dim += 2  # Time step (s and s²) — skipped when FiLM handles it
        if self._avp_enabled:
            predictor_input_dim += self._avp_output_dim  # AVP features per anchor
            print(f"[Model] Predictor input_dim: {predictor_input_dim} "
                  f"(base 12 + phys {3 if self._if_phys_param else 0} + AVP {self._avp_output_dim})")

        self.predictor = QueryBasedMultiScalePredictor(
            state_dim=config['model']['state'].get('token_dim', 768),
            query_dim=predictor_input_dim,  # Dynamically computed based on flags
            hidden_dim=config['model']['predictor'].get('hidden_dims', [512, 256])[0],
            output_dim=config['model']['predictor']['output_dim'],
            layer_indices=layer_indices,
            layer_index_mapping=self.layer_index_mapping,  # Pass the mapping for memory-optimized decoder
            num_heads=config['model']['predictor'].get('num_heads', config['model']['decoder'].get('num_heads', 8)),  # Use predictor's num_heads if specified
            dropout=config['model']['predictor']['dropout'],
            use_ref_pose=config['model']['predictor'].get('use_ref_pose', True),  # Enable ref pose
            norm_type=config['model']['predictor'].get('norm_type', 'layer_norm'),
            qk_norm=config['model']['predictor'].get('qk_norm', False),
            fusion_activation='swiglu',
            use_rope=config['model']['predictor'].get('use_rope', True),  # Enable RoPE for position-aware attention
            rope_dim=config['model']['decoder'].get('rope_dim', None),  # Use same RoPE dim as decoder
            dim_per_coord_3d=config['model']['decoder'].get('dim_per_coord_3d', None),  # Single-Point RoPE config
            dim_per_coord_multi=config['model']['decoder'].get('dim_per_coord_multi', None),  # Multi-Point RoPE config
            elementwise_gate=self._elementwise_gate,  # Pass elementwise_gate config
            cross_object_attention=config['model']['predictor'].get('cross_object_attention', False),  # Cross-object anchor attention
            cross_object_kv=config['model']['predictor'].get('cross_object_kv', False),  # Cross-object key-value attention
            no_global_state=config['model'].get('no_global_state', False)  # Removes unused state input channels
        )
        # model.predictor.rmsnorm_eps: training_dtype -> the predictor's RMSNorm modules keep the eps they have in
        # bf16-autocast training also when evaluated in fp32 (their default eps follows the input dtype; see
        # pin_rmsnorm_eps). Applied here, so every construction of the model computes the configured function.
        _rms_eps_mode = config['model']['predictor'].get('rmsnorm_eps')
        if _rms_eps_mode == 'training_dtype':
            _n_pinned = self.predictor.pin_rmsnorm_eps()
            logger.info(f"[Predictor] RMSNorm eps pinned to the bf16-training values on {_n_pinned} modules")
        elif _rms_eps_mode not in (None, 'default'):
            raise ValueError("model.predictor.rmsnorm_eps must be 'training_dtype' or 'default'")
        # model.predictor.fp32_output: the acceleration read-out layer in fp32 under bf16 training autocast (evaluation
        # runs it in fp32 already, so evaluation is unchanged)
        self.predictor.fp32_output = bool(config['model']['predictor'].get('fp32_output', False))
        if self.predictor.fp32_output:
            logger.info("[Predictor] acceleration read-out (fusion_out) in fp32 during training")

        # Store attention flags for reference
        self._cross_object_attention = config['model']['predictor'].get('cross_object_attention', False)
        self._cross_object_kv = config['model']['predictor'].get('cross_object_kv', False)

        # Log attention configuration summary
        logger.info("=" * 60)
        logger.info("[Attention Config Summary]")
        logger.info(f"  Decoder Objects Self-Attention: {'ENABLED' if self._objects_self_attention else 'DISABLED'}")
        logger.info(f"  Predictor Cross-Object Anchor Attention: {'ENABLED' if self._cross_object_attention else 'DISABLED'}")
        logger.info(f"  Predictor Cross-Object Key-Value: {'ENABLED' if self._cross_object_kv else 'DISABLED'}")
        logger.info("=" * 60)

        # Store actual dt for video/visualization
        self.dt = config['data'].get('dt', 0.01)  # Actual timestep (0.01 = 100 FPS)
        # Verlet integration always uses dt=1; the network learns accelerations in this unit system.
        self.dt_verlet = 1.0
        


        # Create transform layer (configurable)
        transform_cfg = config.get('model', {}).get('transform', {})
        transform_method = transform_cfg.get('method', 'kabsch')
        self.learnable_transform = (transform_method == 'learnable')

        # RoMa differentiable Kabsch (if enabled, no STE needed)
        self.if_roma = bool(transform_cfg.get('if_roma', False))
        if self.if_roma and not ROMA_AVAILABLE:
            raise ImportError(
                "RoMa is required when if_roma=true. Install with: pip install roma"
            )

        # Build kwargs per method to avoid passing unsupported args
        if self.learnable_transform:
            tl_kwargs = {
                'use_6d_rotation': transform_cfg.get('use_6d_rotation', True),
                'learnable_mode': transform_cfg.get('learnable_mode', 'absolute')
            }
        else:
            tl_kwargs = {
                'regularize_rotation': transform_cfg.get('regularize_rotation', True)
            }
        self.transform_layer = create_transform_layer(
            method=transform_method,
            if_roma=self.if_roma,
            **tl_kwargs
        )
        
        # ============== Multi-object (MOVi) specific modules ==============
        self.enable_multi_object = bool(self.config.get('data', {}).get('multi_object', False))
        if self.enable_multi_object:
            # Cache for mesh faces (dataset_name:obj_idx -> faces array)
            self._faces_cache = {}

            # Shared PointNet encoder for arbitrary number of objects (weights shared)
            enc_cfg = config.get('model', {}).get('encoder', {})
            # Dynamic encoder input_dim calculation:
            # Base: 9D (dist_vec + velocity + rel_pos)
            # + phys_params (if if_phys_param=true): +3D
            # + time_step (if if_time_step=true and no FiLM): +2D (s and s²)
            encoder_input_dim = 9  # Base
            if self._if_phys_param:
                encoder_input_dim += 3  # Physics params
            if self._if_time_step and not self._film_conditioning:
                encoder_input_dim += 2  # Time step (s and s²) — skipped when FiLM handles it
            if enc_cfg.get('height_input', False):
                encoder_input_dim += 1  # Height above the ground plane (absolute z)
            self._encoder_input_dim = encoder_input_dim  # Save for logging

            self.shared_obj_encoder = RigidBodyEncoder(
                input_dim=encoder_input_dim,
                hidden_dims=enc_cfg.get('hidden_dims', [512, 1024]),
                output_dim=enc_cfg.get('backbone_dim', enc_cfg.get('output_dim', 896)),
                backbone_dim=enc_cfg.get('backbone_dim', enc_cfg.get('output_dim', 896)),
                use_stn=enc_cfg.get('use_stn', False),
                use_batch_norm=enc_cfg.get('use_batch_norm', True),
                masked_pooling=enc_cfg.get('masked_pooling', False),
                freeze_bn_stats=enc_cfg.get('freeze_bn_stats', False),
                height_input=enc_cfg.get('height_input', False),
                hierarchical_subsets=enc_cfg.get('hierarchical_subsets', False),
                packed_pooling=enc_cfg.get('packed_pooling', 'scatter'),
                subset_small_objects=enc_cfg.get('subset_small_objects', 'all_below_8'),
                feature_transform=enc_cfg.get('feature_transform', False),
                feature_transform_dim=enc_cfg.get('feature_transform_dim', 64),
                pooling_type=enc_cfg.get('pooling_type', 'max'),  # New: enhanced pooling
                num_hierarchical_levels=enc_cfg.get('num_hierarchical_levels', 2)  # New: hierarchical levels
            )
            # Project per-object latent to model token dim
            token_dim = config['model']['state'].get('token_dim', 768)
            enc_out_dim = enc_cfg.get('backbone_dim', enc_cfg.get('output_dim', 896))
            # Project PointNet encoder output (per-object) to token dimension for Decoder
            self.obj_feature_proj = nn.Linear(enc_out_dim, token_dim)
            # Anchor query dimension is always 9D (dist_vec + vel + rel_pos)
            anchor_query_dim = 9
            # Projection for anchor query features (for Predictor)
            self.anchor_query_proj = nn.Linear(anchor_query_dim, token_dim)

            # FiLM conditioning: modulate encoder output with [s, s²] instead of
            # concatenating step size to the input features
            if self._film_conditioning and self._if_time_step:
                from src.models.encoder import StepSizeFiLM
                self.encoder_film = StepSizeFiLM(time_dim=2, feature_dim=token_dim)
                print(f"[Model] FiLM conditioning ENABLED: step-size modulates encoder output (dim={token_dim})")
                print(f"[Model]   Encoder input: {encoder_input_dim}D (time NOT concatenated, handled by FiLM)")
                print(f"[Model]   Predictor query: {predictor_input_dim}D (time NOT in query)")
            else:
                self.encoder_film = None
                if self._film_conditioning and not self._if_time_step:
                    print(f"[Model] WARNING: film_conditioning=True but if_time_step=False; FiLM disabled")

            # ============== PE Type for ARoPE Ablation ==============
            # pe_type controls what 12D positional descriptor is fed to the decoder's RoPE path.
            # "arope" (default): original 4 anchors × 3D = 12D
            # "obb": OBB center + 3 half-axis vectors = 12D
            # "pca": PCA centroid + √λ·eigenvectors = 12D
            self._pe_type = config['model']['decoder'].get('pe_type', 'arope')
            _VALID_PE_TYPES = ('arope', 'obb', 'pca', 'se3')
            if self._pe_type not in _VALID_PE_TYPES:
                raise ValueError(
                    f"Invalid pe_type='{self._pe_type}'. Must be one of {_VALID_PE_TYPES}."
                )
            _pe_msg = f"[PE Type] Using pe_type='{self._pe_type}' for decoder/predictor positional encoding."
            print(_pe_msg)
            logger.info(_pe_msg)

            self._geometry_encoder = None
            if self._pe_type in ('obb', 'pca'):
                from src.models.geometry_encoders import OBBEncoder, PCAFrameEncoder
                if self._pe_type == 'obb':
                    self._geometry_encoder = OBBEncoder()
                    logger.info("[PE Type] Initialized OBBEncoder (center + 3 half-axis vectors -> 12D)")
                elif self._pe_type == 'pca':
                    self._geometry_encoder = PCAFrameEncoder()
                    logger.info("[PE Type] Initialized PCAFrameEncoder (centroid + scaled eigenvectors -> 12D)")

            # SE(3) pairwise bias module
            self._se3_bias_module = None
            if self._pe_type == 'se3':
                from src.models.se3_attention import SE3PairwiseBias
                _num_heads = config['model']['decoder'].get('num_heads', 6)
                _se3_hidden = config['model']['decoder'].get('se3_hidden_dim', 64)
                self._se3_bias_module = SE3PairwiseBias(
                    num_heads=_num_heads, hidden_dim=_se3_hidden, use_scale_ratio=True
                )
                logger.info(f"[PE Type] Initialized SE3PairwiseBias (7D -> {_se3_hidden} -> {_num_heads} heads, zero-init, scale-norm)")

            # ============== RoPE frequency configuration summary ==============
            _rope_dim = config['model']['decoder'].get('rope_dim', None)
            _dim_per_coord_3d = config['model']['decoder'].get('dim_per_coord_3d', None)
            _dim_per_coord_multi = config['model']['decoder'].get('dim_per_coord_multi', None)
            _hidden_dim = config['model']['decoder'].get('hidden_dim', 768)
            _num_heads = config['model']['decoder'].get('num_heads', 6)
            _head_dim = _hidden_dim // _num_heads if _num_heads > 0 else 0
            _anchor_k = config.get('data', {}).get('anchor_k', 4)
            _n_coords = _anchor_k * 3
            _rope_msg = (
                f"[RoPE Freq] rope_dim={_rope_dim} | "
                f"dim_per_coord_multi={_dim_per_coord_multi} (freq/coord) | "
                f"dim_per_coord_3d={_dim_per_coord_3d} | "
                f"hidden_dim={_hidden_dim} | num_heads={_num_heads} | head_dim={_head_dim} | "
                f"anchor_k={_anchor_k} | n_coords={_n_coords} | "
                f"check: {_n_coords}x{_dim_per_coord_multi}={_n_coords * _dim_per_coord_multi if _dim_per_coord_multi else '?'} "
                f"{'== ' + str(_rope_dim) if _dim_per_coord_multi and _n_coords * _dim_per_coord_multi == _rope_dim else '!= ' + str(_rope_dim) + ' (MISMATCH)'}"
            )
            print(_rope_msg)
            logger.info(_rope_msg)

    def forward(self, batch, return_all=False, teacher_forcing_ratio=1.0):
        """
        Forward pass through the model.

        Must remain consistent with rollout():
        1. Reference pose is frame 0 of the full sequence (not batch[:, 0])
        2. Input: current position only (no velocity to encoder)
        3. Output: acceleration, integrated via Verlet to get the next position
        4. Predictor maps (current_pos, ref_pose_frame0) -> acceleration
        5. Verlet: x_{t+1} = a*dt² + 2*x_t - x_{t-1}

        Args:
            batch: Dictionary with body positions and velocities
            return_all: Whether to return intermediate results
            teacher_forcing_ratio: Probability of using ground truth (1.0 = always teacher, 0.0 = always student)
            
        Returns:
            predictions: Dictionary with predicted positions, velocities, etc.
        """
        # Multi-object path only
        if 'obj_positions' in batch and getattr(self, 'enable_multi_object', False):
            return self._forward_multi_object(batch, return_all=return_all, teacher_forcing_ratio=teacher_forcing_ratio)
        else:
            raise ValueError("Expected multi-object batch with 'obj_positions'.")

    # ================= Multi-object forward (MOVi) =================
    def _forward_multi_object(self, batch, return_all=False, teacher_forcing_ratio=1.0):
        """
        Full multi-object forward path using shared per-object PointNet encoder and
        mask-aware per-vertex queries across all objects.
        """
        time_step_id = batch['time_step_id']  # (B,)
        obj_pos = batch['obj_positions']  # (B,T,Nobj,Nv,3)
        obj_mask = batch['valid_obj_mask']  # (B,Nobj)
        vtx_mask = batch['valid_vtx_mask']  # (B,T,Nobj,Nv)
        ref_obj_positions = batch['ref_obj_positions']  # (B?,Nobj,Nv,3) or (Nobj,Nv,3)

        # Read physics parameters and time step if enabled
        phys_params = None  # (B, Nobj, 3): [mass, friction, restitution]
        skip_step_size_batch = None  # (B,)
        if self._if_phys_param:
            if 'phys_params' not in batch:
                raise KeyError("Batch is missing 'phys_params' required when if_phys_param=true")
            phys_params = batch['phys_params'].to(obj_pos.device)  # (B, Nobj, 3)
        if self._if_time_step:
            if 'skip_step_size' not in batch:
                raise KeyError("Batch is missing 'skip_step_size' required when if_time_step=true")
            skip_step_size_batch = batch['skip_step_size'].to(obj_pos.device)  # (B,)

        B, T, Nobj, Nv, _ = obj_pos.shape
        device = obj_pos.device

        # Normalize reference pose to (B,Nobj,Nv,3)
        if ref_obj_positions.dim() == 3:
            ref_pose = ref_obj_positions.unsqueeze(0).expand(B, -1, -1, -1).to(device)
        else:
            ref_pose = ref_obj_positions.to(device)  # (B,Nobj,Nv,3)

        # ========== Center scene by first frame ==========
        # If enabled, subtract the global center of the first frame from all coordinates;
        # relative distances and object shapes are unaffected.
        if self.config.get('data', {}).get('center_scene_by_t0', False):
            for b in range(B):
                vmask_b0 = vtx_mask[b, 0]  # (Nobj, Nv)
                v_t0 = obj_pos[b, 0][vmask_b0]  # Valid vertices at t=0
                if v_t0.shape[0] > 0:
                    center_t0 = v_t0.mean(dim=0)  # (3,)
                    
                    # Log centering (only once)
                    if not hasattr(self, '_center_t0_logged_train'):
                        logger.info(f"[Center-T0 TRAIN] Enabled. Sample b={b}: center_t0={center_t0.cpu().numpy().tolist()}, "
                                   f"obj_pos range before: [{obj_pos[b].min().item():.4f}, {obj_pos[b].max().item():.4f}]")
                    
                    # Shift positions
                    obj_pos[b] = obj_pos[b] - center_t0
                    ref_pose[b] = ref_pose[b] - center_t0
                    
                    if not hasattr(self, '_center_t0_logged_train'):
                        logger.info(f"[Center-T0 TRAIN] After centering b={b}: "
                                   f"obj_pos range: [{obj_pos[b].min().item():.4f}, {obj_pos[b].max().item():.4f}], "
                                   f"ref_pose range: [{ref_pose[b].min().item():.4f}, {ref_pose[b].max().item():.4f}]")
            if not hasattr(self, '_center_t0_logged_train'):
                self._center_t0_logged_train = True
                logger.info(f"[Center-T0 TRAIN] Scene centering applied to obj_pos and ref_pose for batch of {B} samples")

        # Require anchors for query and supervision (no fallback)
        if 'anchor_idx' not in batch:
            raise KeyError("Batch is missing 'anchor_idx' required for anchor-based querying")
        anchor_idx = batch['anchor_idx'].to(device).long()  # (B,Nobj,Nq)
        if anchor_idx.dim() != 3 or anchor_idx.shape[0] != B or anchor_idx.shape[1] != Nobj:
            raise ValueError(f"anchor_idx must be (B,Nobj,Nq); got {tuple(anchor_idx.shape)} vs (B={B},Nobj={Nobj})")
        Nq = int(anchor_idx.shape[2])

        # Initialize persistent state tokens and 3D positions for RoPE
        # When no_global_state=True, pass None to decoder (skips all state cross-attention)
        if self._no_global_state:
            state_tokens = None
            state_pos = None
        else:
            state_tokens, state_pos = self.state_repr.get_state_and_positions(B, device)

        # Storage across time steps
        all_pred_pos_for_rollout = []  # STE positions for autoregressive input (B, Nobj, Nv, 3)
        all_pred_pos_raw = []          # Raw positions for loss supervision (B, Nobj, Nv, 3)
        all_pred_pos_raw_norm = []     # Raw normalized positions for loss (before Kabsch)
        all_pred_pos_icp_norm = []     # ICP/Kabsch-transformed normalized positions for loss (after Kabsch)
        all_accel_no_icp_norm_anchor_seq = []  # (B, Nobj, Nq, 3)
        all_accel_icp_norm_seq = []
        all_pen_pred_vertices_seq = []  # (B, Nobj, Nv, 1) per step if enabled (ALL vertices)

        # Registration refs per object (initialized with GT at the first step)
        registration_refs = None

        # Previous step's features for penetration prediction (t>1)
        prev_obj_layers = None
        prev_state_layers = None

        # ========== ANCHOR VALIDATION (whole window, vectorized, one host sync) ==========
        # Every anchor of a valid object must sit on a valid vertex at every frame the loop reads
        # (vtx_mask[:, t] for t in 1..T-2). Checked once per batch instead of per (t, object).
        if T > 2:
            if anchor_idx.dim() != 3 or anchor_idx.shape[0] != B or anchor_idx.shape[2] != Nq:
                raise RuntimeError(
                    f"anchor_idx shape mismatch: got {tuple(anchor_idx.shape)}, expected (B={B},Nobj,Nq={Nq})")
            _vmask_win = vtx_mask[:, 1:T - 1]                                # (B,Tw,Nobj,Nv)
            _numv_win = _vmask_win.sum(dim=3)                                # (B,Tw,Nobj)
            _idx_win = anchor_idx.unsqueeze(1).expand(-1, T - 2, -1, -1)     # (B,Tw,Nobj,Nq)
            _at_anchor = torch.gather(_vmask_win, 3, _idx_win)               # (B,Tw,Nobj,Nq)
            _ovalid = obj_mask.unsqueeze(1)                                  # (B,1,Nobj)
            _zero_bad = _ovalid & (_numv_win == 0)
            _short_bad = _ovalid & (_numv_win < Nq)
            _anch_bad = (~_at_anchor) & _ovalid.unsqueeze(-1)
            _flags = torch.stack([_zero_bad.any(), _short_bad.any(), _anch_bad.any()])
            if bool(_flags.any().item()):  # the single sync; everything below is the error path
                zb, sb, ab = [bool(x.item()) for x in _flags]
                _where = (_zero_bad if zb else (_short_bad if sb else _anch_bad.any(dim=-1))).nonzero(as_tuple=False)
                _b0, _tw0, _i0 = [int(v) for v in _where[0][:3]]
                _nv0 = int(_numv_win[_b0, _tw0, _i0].item())
                ds_name = batch.get('dataset_name', None)
                ds_name_str = ds_name if isinstance(ds_name, str) else (ds_name[0] if hasattr(ds_name, '__getitem__') else 'unknown')
                logger.error(
                    f"[Anchor Debug] dataset={ds_name_str} t={_tw0 + 1} obj={_i0} sample={_b0} Nv={Nv} anchor_k={Nq} "
                    f"valid_vertices={_nv0} zero_any={zb} short_any={sb} bad_anchor_any={ab}")
                if zb:
                    raise RuntimeError(f"Zero valid vertices encountered at t={_tw0 + 1}, obj={_i0} for some valid samples; dataset likely corrupted.")
                if sb:
                    raise RuntimeError(f"Not enough valid vertices for anchors at t={_tw0 + 1}, obj={_i0}: valid={_nv0} < anchor_k={Nq}")
                raise RuntimeError(
                    f"Invalid anchors: anchor indices refer to invalid/padded vertices at t={_tw0 + 1}, obj={_i0}. "
                    f"Consider reducing anchor_k or ensuring consistent Nv per frame.")

        # Object slots that hold a valid object in at least one sample (fixed for the batch). The per-object
        # KNN skips the others: they are padding everywhere and their distance vectors are zeroed anyway.
        _knn_active_slots = (torch.nonzero(obj_mask.any(dim=0)).flatten().tolist()
                             if self._knn_num_groups <= 1 else None)
        # Constant scalars of the decoder attention mask and the valid-object index set used by the Kabsch
        # fit: both are fixed for the batch, so they are built once instead of at every step (the per-step
        # versions cost two host-to-device copies and one nonzero sync each step).
        _attn_zero_fwd = torch.tensor(0.0, dtype=torch.float32, device=obj_mask.device)
        _attn_ninf_fwd = torch.tensor(float('-inf'), dtype=torch.float32, device=obj_mask.device)
        _kabsch_valid_idx_fwd = obj_mask.reshape(-1).nonzero(as_tuple=True)[0]
        # Valid-vertex packs (model.knn_packed / model.encoder.packed), fixed for the batch: every frame of a
        # sequence has the same vertex mask, which is checked once here.
        _knn_pack_fwd = _enc_pack_fwd = None
        if self._knn_packed or self._encoder_packed:
            _vm1 = vtx_mask[:, 1]
            if not bool((vtx_mask[:, 1:T - 1] == _vm1.unsqueeze(1)).all()):
                raise ValueError('valid-vertex packing needs the same vertex mask in every frame of a sequence')
            if self._knn_packed:
                _knn_pack_fwd = knn_pack(_vm1, obj_mask)
            if self._encoder_packed:
                _enc_m = self.shared_obj_encoder
                _vr0 = batch.get('vtx_rank')
                _enc_pack_fwd = EncoderPack(_vm1.reshape(B * Nobj, Nv),
                                            _vr0.reshape(B * Nobj, Nv) if _vr0 is not None else None,
                                            len(_enc_m.hierarchical_layers), _enc_m.hierarchical_subsets, 1,
                                            small=_enc_m.subset_small_objects)

        def _net_step(t, curr_pos_bt, prev_pos_bt, B, Nobj, Nv, Nq, obj_mask, vtx_mask_t, anchor_idx, ref_pose,
                      phys_params, skip_step_size_batch, vtx_rank_all, state_tokens, bn_groups, knn_chunks, knn_slots,
                      enc_pack=None, knn_pk=None):
            """Network part of one step (distance features -> encoder -> AVP -> decoder -> predictor):
            per-anchor accelerations for the batch given current/previous positions. Called once per step,
            or once for all teacher-forced steps stacked along the batch (bn_groups = knn_chunks = steps)."""
            # Velocities
            vel_bt = curr_pos_bt - prev_pos_bt  # (B,Nobj,Nv,3)

            # Build per-object features and per-object latents
            per_obj_latents = []  # each (B, D)
            # Build per-object normalized query features and ref poses (padded)
            per_obj_vel_norm = []
            per_obj_relpos_norm = []
            per_obj_distvec_norm = []

            # Prepare far-away filler for invalid vertices in reference lists
            FAR = 1e6

            # (Anchor validation runs once per batch before the time loop: it depends only on the
            #  ground-truth masks and anchor indices, never on predictions.)

            # ========== VECTORIZED KNN FOR ALL OBJECTS AT ONCE ==========
            # curr_pos_bt: (B, Nobj, Nv, 3), vtx_mask_t: (B, Nobj, Nv)
            vmask_t = vtx_mask_t  # (B, Nobj, Nv)

            # Compute KNN distance vectors for ALL objects simultaneously
            # Only computes for valid objects and valid vertices!
            # Returns: (B, Nobj, Nv, 3) - vector from each vertex to nearest point on OTHER objects
            if knn_pk is not None:
                # packed valid vertices; with batched time steps the step-sized chunks share one pack
                _cs = B // knn_chunks
                dist_vec_all = (compute_knn_packed(curr_pos_bt, knn_pk) if knn_chunks == 1 else torch.cat(
                    [compute_knn_packed(curr_pos_bt[_c * _cs:(_c + 1) * _cs], knn_pk) for _c in range(knn_chunks)], dim=0))
            elif knn_chunks == 1:
                dist_vec_all = compute_knn_all_objects_vectorized(
                    curr_pos_bt, vmask_t, FAR, valid_obj_mask=obj_mask,
                    num_groups=self._knn_num_groups, active_slots=knn_slots
                )
            else:
                # batched time steps: one call per step-sized chunk (memory of the per-object cdist stays
                # that of a single step)
                _cs = B // knn_chunks
                dist_vec_all = torch.cat([
                    compute_knn_all_objects_vectorized(
                        curr_pos_bt[_c * _cs:(_c + 1) * _cs], vmask_t[_c * _cs:(_c + 1) * _cs], FAR,
                        valid_obj_mask=obj_mask[_c * _cs:(_c + 1) * _cs],
                        num_groups=self._knn_num_groups, active_slots=knn_slots)
                    for _c in range(knn_chunks)], dim=0)

            # Apply plane constraint adjustment (vectorized)
            # Reshape for distance_2_plane: (B, Nobj*Nv, 3)
            # Use reshape() instead of view() to handle non-contiguous tensors
            curr_pos_flat = curr_pos_bt.reshape(B, Nobj * Nv, 3)
            dist_vec_flat = dist_vec_all.reshape(B, Nobj * Nv, 3)
            dist_vec2_flat = distance_2_plane(curr_pos_flat, [0, 0, 0], "z", dist_vec_flat)
            dist_vec2_stacked = dist_vec2_flat.reshape(B, Nobj, Nv, 3)

            # Relative position (vectorized): curr_pos - ref_pose
            relpos_stacked = curr_pos_bt - ref_pose  # (B, Nobj, Nv, 3)

            # Velocity (already in correct shape)
            vel_stacked = vel_bt  # (B, Nobj, Nv, 3)

            # Noise augmentation (vectorized)
            if self.training:
                aug_config = self.config['data'].get('augmentation', {})
                if aug_config.get('dist_vec_noise', 0) > 0:
                    std = aug_config['dist_vec_noise']
                    dist_vec2_stacked = dist_vec2_stacked + torch.randn_like(dist_vec2_stacked) * std
                if aug_config.get('velocity_noise', 0) > 0:
                    std = aug_config['velocity_noise']
                    vel_stacked = vel_stacked + torch.randn_like(vel_stacked) * std
                if aug_config.get('rel_pos_noise', 0) > 0:
                    std = aug_config['rel_pos_noise']
                    relpos_stacked = relpos_stacked + torch.randn_like(relpos_stacked) * std

            # Apply vertex masks BEFORE normalization (vectorized across all objects)
            vmask_all = vtx_mask_t.unsqueeze(-1)  # (B,Nobj,Nv,1)
            dist_vec2_stacked = torch.where(vmask_all, dist_vec2_stacked, torch.zeros_like(dist_vec2_stacked))
            vel_stacked = torch.where(vmask_all, vel_stacked, torch.zeros_like(vel_stacked))
            relpos_stacked = torch.where(vmask_all, relpos_stacked, torch.zeros_like(relpos_stacked))

            # Apply network-predicted directional sign to ALL vertices' dist_vec (before encoder)
            # Only for t>1 (may have penetration in student forcing with predicted positions)
            # For t=1, GT positions (no penetration, sign=+1 implicit)
            # Use if_penetration as master switch (fallback to if_directional_distance for backwards compatibility)
            penetration_enabled = self.config['model'].get('if_penetration', self.config['model'].get('if_directional_distance', False))
            if penetration_enabled and t > 1:
                # Predict penetration sign using PREVIOUS step's features and CURRENT positions
                # curr_pos_bt was already selected by teacher/student forcing above
                # In student forcing mode: curr_pos_bt = all_pred_pos_for_rollout[-1] = blended_next (after Kabsch transform)
                # GT supervision will use predictions['obj_positions_ste'] to match
                pen_pred_all_vertices = self.predictor.predict_penetration_vertices(
                    prev_obj_layers,  # Use previous step's decoder outputs
                    prev_state_layers,  # Use previous step's state features
                    pos_full=curr_pos_bt  # AFTER Kabsch transform (student forcing) or GT (teacher forcing/t=1)
                )  # (B, Nobj, Nv, 1) in [-1,1]

                # Apply predicted sign to dist_vec for ALL vertices
                pen_sign_all = torch.sign(pen_pred_all_vertices + 1e-6)  # (B, Nobj, Nv, 1) in {-1,+1}
                dist_vec2_stacked = dist_vec2_stacked * pen_sign_all  # (B, Nobj, Nv, 3)

                # Store predicted signs for loss supervision (will compute GT later in validation)
                all_pen_pred_vertices_seq.append(pen_pred_all_vertices)
            elif penetration_enabled and t == 1:
                # First step: GT positions, no penetration prediction
                # Append placeholder (all +1) to maintain time dimension alignment
                # This will NOT be used in loss (masked out), but keeps dimensions consistent
                B, Nobj, Nv = dist_vec2_stacked.shape[0], dist_vec2_stacked.shape[1], dist_vec2_stacked.shape[2]
                placeholder = torch.ones((B, Nobj, Nv, 1), dtype=torch.float32, device=self.device)
                all_pen_pred_vertices_seq.append(placeholder)

            # No normalization — use raw values directly
            dist_vec2_norm_stacked = dist_vec2_stacked
            vel_norm_stacked = vel_stacked
            relpos_norm_stacked = relpos_stacked

            # Broadcast physics parameters and time_step to vertex dimension if enabled
            phys_broadcast = None  # (B, Nobj, Nv, 3)
            time_broadcast = None  # (B, Nobj, Nv, 2) - contains [s, s²]

            if self._if_phys_param:
                # Broadcast per-object physics params (B,Nobj,3) to all vertices (B,Nobj,Nv,3)
                phys_broadcast = phys_params.unsqueeze(2).expand(B, Nobj, Nv, 3)  # (B,Nobj,Nv,3)
                # Mask with vertex mask to ensure padding vertices get zero physics params
                phys_broadcast = torch.where(vmask_all, phys_broadcast, torch.zeros_like(phys_broadcast))

            if self._if_time_step:
                # Broadcast both s and s² to all objects and vertices (B,Nobj,Nv,2)
                # Input both s and s² because:
                #   - s: original step_size for temporal context
                #   - s²: the scaling factor that will be applied to network output
                #   - Network can leverage both for better learning
                step_size_s = skip_step_size_batch.float()  # (B,)
                step_size_sq = (skip_step_size_batch ** 2).float()  # (B,)
                # Stack s and s² -> (B, 2)
                time_features = torch.stack([step_size_s, step_size_sq], dim=-1)  # (B, 2)
                time_broadcast = time_features.view(B, 1, 1, 2).expand(B, Nobj, Nv, 2)  # (B,Nobj,Nv,2)
                # Time step doesn't need masking (it's a scalar property of the batch, not vertex-specific)
                # But for consistency, we can zero it for padding vertices
                # vmask_all is (B,Nobj,Nv,1), expand to match (B,Nobj,Nv,2)
                vmask_time = vmask_all.expand(-1, -1, -1, 2)  # (B,Nobj,Nv,2)
                time_broadcast = torch.where(vmask_time, time_broadcast, torch.zeros_like(time_broadcast))

            # PointNet Encoder: process each object's Nv vertices -> per-object global feature
            # Input dimension depends on flags:
            #   Base: 9D (dist_vec + velocity + rel_pos)
            #   +if_phys_param=true: +3D
            #   +if_time_step=true (no FiLM): +2D (s and s²)
            # Split into components and reshape to (B*Nobj, Nv, D) for encoder
            dist_reshaped = dist_vec2_norm_stacked.view(B * Nobj, Nv, 3)
            vel_reshaped = vel_norm_stacked.view(B * Nobj, Nv, 3)
            relpos_reshaped = relpos_norm_stacked.view(B * Nobj, Nv, 3)

            phys_reshaped = None
            if self._if_phys_param:
                phys_reshaped = phys_broadcast.view(B * Nobj, Nv, 3)

            time_reshaped = None
            if self._if_time_step and not self._film_conditioning:
                time_reshaped = time_broadcast.view(B * Nobj, Nv, 2)  # (s, s²)

            # Prepare vertex mask for encoder pooling
            # vmask_all is (B,Nobj,Nv,1), need (B*Nobj, Nv)
            vtx_mask_for_encoder = vmask_all.squeeze(-1).reshape(B * Nobj, Nv)  # (B*Nobj, Nv)
            height_reshaped = None
            if self.shared_obj_encoder.height_input:
                _h = torch.where(vmask_all, curr_pos_bt[..., 2:3], torch.zeros_like(curr_pos_bt[..., 2:3]))
                height_reshaped = _h.reshape(B * Nobj, Nv, 1)
            rank_reshaped = None
            if self.shared_obj_encoder.hierarchical_subsets:
                if vtx_rank_all is None:
                    raise KeyError("model.encoder.hierarchical_subsets requires data.fps_order: true (batch lacks 'vtx_rank')")
                rank_reshaped = vtx_rank_all.reshape(B * Nobj, Nv)

            # Call encoder with all features (encoder will concatenate them internally)
            # AVP: capture per-point features if AVP is enabled
            obj_features_encoded, _per_vtx_feat, _ = self.shared_obj_encoder(
                dist_reshaped,
                vel_reshaped,
                relpos_reshaped,
                phys_params=phys_reshaped,
                time_step=time_reshaped,
                valid_mask=vtx_mask_for_encoder,
                height=height_reshaped,
                vtx_rank=rank_reshaped,
                bn_groups=bn_groups,
                pack=enc_pack
            )  # obj_features_encoded: (B*Nobj, enc_out_dim), _per_vtx_feat: (B*Nobj, backbone_dim, Nv)

            # AVP: Anchor-Vertex Pooling — enrich anchor queries with collision-aware features
            _avp_anchor_features = None  # Will be (B, Nobj, Nq, avp_output_dim) if AVP enabled
            if self.avp is not None and _per_vtx_feat is not None:
                # _per_vtx_feat is (B*Nobj, backbone_dim, Nv) — Conv1d channels-first format
                # Transpose to (B*Nobj, Nv, backbone_dim) for AVP
                _pvf = _per_vtx_feat.transpose(1, 2)  # (B*Nobj, Nv, backbone_dim)
                # Anchor positions in flat format
                _curr_pos_flat = curr_pos_bt.reshape(B * Nobj, Nv, 3)  # (B*Nobj, Nv, 3)
                _anchor_pos_flat = torch.gather(
                    _curr_pos_flat, dim=1,
                    index=anchor_idx.reshape(B * Nobj, -1).unsqueeze(-1).expand(-1, -1, 3)
                )  # (B*Nobj, Nq, 3)
                _avp_anchor_features = self.avp(
                    _pvf, _curr_pos_flat, _anchor_pos_flat, vtx_mask_for_encoder
                )  # (B*Nobj, Nq, avp_output_dim)
                _avp_anchor_features = _avp_anchor_features.view(B, Nobj, anchor_idx.shape[2], -1)  # (B, Nobj, Nq, avp_output_dim)
                # Zero out padding objects (proj bias makes padding non-zero; consistent with other features)
                _avp_anchor_features = _avp_anchor_features * obj_mask.unsqueeze(-1).unsqueeze(-1).float()

            obj_features_encoded = obj_features_encoded.view(B, Nobj, -1)  # (B, Nobj, enc_out_dim)

            # Project to token dimension for Decoder
            obj_features = self.obj_feature_proj(obj_features_encoded)  # (B, Nobj, D)

            # FiLM: modulate encoder features with step-size [s, s²]
            if self.encoder_film is not None and skip_step_size_batch is not None:
                time_feat_film = torch.stack([
                    skip_step_size_batch.float(),
                    (skip_step_size_batch ** 2).float()
                ], dim=-1)  # (B, 2)
                obj_features = self.encoder_film(obj_features, time_feat_film)  # (B, Nobj, D)

            # Object anchor positions for RoPE (like RenderFormer's triangle approach)
            # RenderFormer: Triangle = 3 vertices (9D)
            # Our approach: Object = Nq anchors (Nq*3 D)
            # Gather anchor positions: (B, Nobj, Nq, 3)
            # Replace padding objects' positions with zeros before the gather:
            # padding objects have anchor_idx=0, which would gather arbitrary values into RoPE inputs.
            obj_mask_pos = obj_mask.unsqueeze(-1).unsqueeze(-1)  # (B, Nobj, 1, 1)
            safe_curr_pos_bt = torch.where(obj_mask_pos, curr_pos_bt, torch.zeros_like(curr_pos_bt))

            idx_exp_3_for_decoder = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, Nobj, Nq, 3)
            obj_anchor_positions = torch.gather(safe_curr_pos_bt, dim=2, index=idx_exp_3_for_decoder)  # (B, Nobj, Nq, 3)
            # Flatten to (B, Nobj, Nq*3) for multi-point RoPE
            Nq = anchor_idx.shape[2]  # Number of anchors per object

            # ============== PE Type Routing (forward) ==============
            if self._pe_type in ('arope', 'se3'):
                # SE(3) uses arope anchors for cross-attention RoPE; SE(3) bias computed separately
                obj_anchor_flat = obj_anchor_positions.view(B, Nobj, Nq * 3)  # (B, Nobj, 12)
            elif self._pe_type in ('obb', 'pca'):
                obj_anchor_flat = self._geometry_encoder(
                    safe_curr_pos_bt, vtx_mask_t, obj_mask=obj_mask
                )  # (B, Nobj, 12)
            else:
                raise ValueError(f"Unknown pe_type='{self._pe_type}' in forward().")

            # ============== SE(3) Pairwise Bias Computation ==============
            _se3_bias = None
            if self._pe_type == 'se3' and self._se3_bias_module is not None:
                from src.models.se3_attention import extract_object_pose_features
                _rotmats, _centers, _scales = extract_object_pose_features(
                    safe_curr_pos_bt, ref_pose, vtx_mask_t, obj_mask
                )
                _se3_bias = self._se3_bias_module(_rotmats, _centers, _scales, obj_mask)

            # Decoder attention mask: SDPA additive format (0=attend, -inf=mask)
            # Prevents padding objects from participating in Decoder self-attention.
            # Combined with feature zeroing to ensure clean inputs.
            obj_attn_mask = torch.where(
                obj_mask.unsqueeze(1).unsqueeze(1),  # (B, 1, 1, Nobj)
                _attn_zero_fwd,
                _attn_ninf_fwd
            )  # (B, 1, 1, Nobj) float32

            # Zero out padding object features and anchor positions BEFORE Decoder
            # to prevent garbage from padding objects from destabilizing attention
            pad_mask_float = obj_mask.unsqueeze(-1).to(dtype=obj_features.dtype)
            obj_features = obj_features * pad_mask_float  # (B, Nobj, D)
            obj_anchor_flat = obj_anchor_flat * obj_mask.unsqueeze(-1).to(dtype=obj_anchor_flat.dtype)

            # Decoder interaction with per-object features
            # Per-layer FiLM: pass time_features for step-size conditioning in each layer
            _decoder_time_feat = time_feat_film if (self._film_conditioning and self._if_time_step) else None
            all_state_layers, all_obj_layers = self.decoder(
                state_tokens,
                obj_features,
                state_pos,
                obj_anchor_flat,
                mask=obj_attn_mask,  # SDPA additive mask for padding objects
                se3_bias=_se3_bias,
                time_features=_decoder_time_feat  # Per-layer FiLM
            )

            # Update state for next step (skip when no_global_state)
            if not self._no_global_state:
                state_tokens = all_state_layers[-1]

            # Build anchor query for Predictor (sample at anchor points Nq)
            # Query dimension depends on flags:
            #   Base: 9D (dist_vec + vel + rel_pos)
            #   +if_phys_param=true: +3D = 12D
            #   +if_time_step=true (no FiLM): +2D (s and s²)
            idx_exp_3 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
            distvec_at_anchors = torch.gather(dist_vec2_norm_stacked, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)
            vel_at_anchors = torch.gather(vel_norm_stacked, dim=2, index=idx_exp_3)            # (B,Nobj,Nq,3)
            relpos_at_anchors = torch.gather(relpos_norm_stacked, dim=2, index=idx_exp_3)      # (B,Nobj,Nq,3)

            # Build query features list
            query_features = [distvec_at_anchors, vel_at_anchors, relpos_at_anchors]

            # Add physics parameters at anchors if enabled
            if self._if_phys_param:
                phys_at_anchors = torch.gather(phys_broadcast, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)
                query_features.append(phys_at_anchors)

            # Add time step at anchors if enabled AND not using FiLM
            if self._if_time_step and not self._film_conditioning:
                idx_exp_2 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,Nobj,Nq,2)
                time_at_anchors = torch.gather(time_broadcast, dim=2, index=idx_exp_2)  # (B,Nobj,Nq,2)
                query_features.append(time_at_anchors)

            # AVP: Append collision-aware anchor features to query
            if _avp_anchor_features is not None:
                query_features.append(_avp_anchor_features)  # (B, Nobj, Nq, avp_output_dim)

            # Concatenate all query features
            anchor_query = torch.cat(query_features, dim=-1)  # (B,Nobj,Nq, 9/12 + avp_dim)

            # Extract anchor positions for RoPE (current frame positions at anchors)
            anchor_pos_at_curr = torch.gather(curr_pos_bt, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)

            # Penetration sign was already applied to all vertices' dist_vec before the encoder;
            # the anchor query therefore carries the sign information as well.

            _ref_src = _object_centered(ref_pose, vmask_all.squeeze(-1).bool()) if self._ref_pose_object_centered else ref_pose
            if self._ref_pose_object_centered and not getattr(self, '_refpose_logged_fwd', False):
                self._refpose_logged_fwd = True
                print(f"[RefPose] object-centred (x/y only, z absolute) reference pose ACTIVE (forward): |shift| max={((ref_pose - _ref_src).abs().max()).item():.3f} m", flush=True)
            ref_pose_at_anchors = torch.gather(_ref_src, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)

            # Check for NaN/Inf before predictor
            if DEBUG_NAN and (torch.isnan(anchor_query).any() or torch.isinf(anchor_query).any()):
                logger.error(f"NaN/Inf detected in anchor_query!")
                logger.error(f"  distvec_at_anchors: nan={torch.isnan(distvec_at_anchors).sum().item()}, inf={torch.isinf(distvec_at_anchors).sum().item()}")
                logger.error(f"  vel_at_anchors: nan={torch.isnan(vel_at_anchors).sum().item()}, inf={torch.isinf(vel_at_anchors).sum().item()}")
                logger.error(f"  relpos_at_anchors: nan={torch.isnan(relpos_at_anchors).sum().item()}, inf={torch.isinf(relpos_at_anchors).sum().item()}")

            if DEBUG_NAN and (torch.isnan(ref_pose_at_anchors).any() or torch.isinf(ref_pose_at_anchors).any()):
                logger.error(f"NaN/Inf detected in ref_pose_at_anchors!")
                logger.error(f"  ref_pose_at_anchors range: [{ref_pose_at_anchors.min().item():.4f}, {ref_pose_at_anchors.max().item():.4f}]")

            for layer_idx, obj_layer in enumerate(all_obj_layers if DEBUG_NAN else ()):
                if torch.isnan(obj_layer).any() or torch.isinf(obj_layer).any():
                    logger.error(f"NaN/Inf detected in all_obj_layers[{layer_idx}]!")
                    logger.error(f"  Shape: {obj_layer.shape}, nan={torch.isnan(obj_layer).sum().item()}, inf={torch.isinf(obj_layer).sum().item()}")

            # Predict accelerations per anchor using grouped format (B, Nobj, Nq, *)
            accel_grouped = self.predictor(
                all_obj_layers,
                anchor_query,           # (B, Nobj, Nq, 9/12)
                ref_pose=ref_pose_at_anchors,  # (B, Nobj, Nq, 3)
                query_pos=anchor_pos_at_curr,  # (B, Nobj, Nq, 3)
                key_pos=obj_anchor_flat,        # (B, Nobj, Nq*3) - multi-point RoPE
                multi_scale_state_features=None if self._no_global_state else all_state_layers,
                state_pos=None if self._no_global_state else state_pos,
                obj_mask=obj_mask  # Object validity mask for padding
            )  # Returns (B, Nobj, Nq, 3)

            # Check for NaN/Inf after predictor
            if DEBUG_NAN and (torch.isnan(accel_grouped).any() or torch.isinf(accel_grouped).any()):
                logger.error(f"NaN/Inf detected in predictor output!")
                logger.error(f"  accel_grouped: nan={torch.isnan(accel_grouped).sum().item()}, inf={torch.isinf(accel_grouped).sum().item()}")
                logger.error(f"  accel_grouped range: min={accel_grouped[~torch.isnan(accel_grouped)].min().item() if (~torch.isnan(accel_grouped)).any() else 'all nan'}, max={accel_grouped[~torch.isnan(accel_grouped)].max().item() if (~torch.isnan(accel_grouped)).any() else 'all nan'}")

            return vmask_t, accel_grouped, all_obj_layers, all_state_layers, state_tokens

        # ========== TIME-BATCHED NETWORK PASS (training.time_batched_forward) ==========
        # With pure teacher forcing every step's network input is built from ground-truth frames, so the network
        # part of all T-2 steps runs once on the steps stacked along the batch (step-major: rows [g*B, (g+1)*B)
        # belong to step g+1). The encoder keeps per-step BatchNorm statistics (grouped masked BN: same
        # statistics, gradients and running-statistics updates as step-by-step calls), and the per-step physics
        # (Verlet, Kabsch, ground handling) below is unchanged. Equal to the step-by-step path up to
        # floating-point rounding and the dropout random stream.
        _tb_accel = None
        if self.training and self._time_batched_forward and T > 2:
            if teacher_forcing_ratio < 1.0:
                raise ValueError("training.time_batched_forward requires teacher_forcing_ratio == 1 (inputs must be ground truth)")
            Tw = T - 2
            rope_memo_reset()

            def _rep(x):
                return x.unsqueeze(0).expand(Tw, *x.shape).reshape(Tw * x.shape[0], *x.shape[1:])
            _ref_b = _rep(ref_pose) if (torch.is_tensor(ref_pose) and ref_pose.dim() == 4 and ref_pose.shape[0] == B) else ref_pose
            _vr = batch.get('vtx_rank')
            _, _acc_s, _, _, _ = _net_step(
                None,
                obj_pos[:, 1:T - 1].transpose(0, 1).reshape(Tw * B, Nobj, Nv, 3),
                obj_pos[:, 0:T - 2].transpose(0, 1).reshape(Tw * B, Nobj, Nv, 3),
                Tw * B, Nobj, Nv, Nq, _rep(obj_mask),
                vtx_mask[:, 1:T - 1].transpose(0, 1).reshape(Tw * B, Nobj, Nv),
                _rep(anchor_idx), _ref_b,
                _rep(phys_params) if phys_params is not None else None,
                _rep(skip_step_size_batch) if skip_step_size_batch is not None else None,
                _rep(_vr) if _vr is not None else None,
                None, Tw, Tw, _knn_active_slots,
                enc_pack=(EncoderPack(vtx_mask[:, 1:T - 1].transpose(0, 1).reshape(Tw * B * Nobj, Nv),
                                      _rep(_vr).reshape(Tw * B * Nobj, Nv) if _vr is not None else None,
                                      len(self.shared_obj_encoder.hierarchical_layers),
                                      self.shared_obj_encoder.hierarchical_subsets, Tw,
                                      small=self.shared_obj_encoder.subset_small_objects)
                          if self._encoder_packed else None),
                knn_pk=_knn_pack_fwd)
            _tb_accel = _acc_s.view(Tw, B, *_acc_s.shape[1:])

        # Time loop predicts for steps t in [1, T-2] -> next positions x_{t+1}
        for t in range(1, T - 1):
            rope_memo_reset()  # rotary angles are memoized within one step only
            # Teacher vs student forcing (no special-case first step)
            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)

            if use_teacher_forcing or len(all_pred_pos_for_rollout) == 0:
                prev_pos_bt = obj_pos[:, t-1]   # (B,Nobj,Nv,3)
                curr_pos_bt = obj_pos[:, t]
            else:
                # Use last predicted (STE blended) as current for autoregressive stability
                curr_pos_bt = all_pred_pos_for_rollout[-1]  # (B,Nobj,Nv,3)
                if len(all_pred_pos_for_rollout) >= 2:
                    prev_pos_bt = all_pred_pos_for_rollout[-2]
                else:
                    prev_pos_bt = obj_pos[:, t-1]

            if _tb_accel is not None:
                # time-batched: this step's accelerations were computed with all teacher-forced steps
                vmask_t = vtx_mask[:, t]
                accel_grouped = _tb_accel[t - 1]
                all_obj_layers = all_state_layers = None
            else:
                vmask_t, accel_grouped, all_obj_layers, all_state_layers, state_tokens = _net_step(
                    t, curr_pos_bt, prev_pos_bt, B, Nobj, Nv, Nq, obj_mask, vtx_mask[:, t], anchor_idx, ref_pose,
                    phys_params, skip_step_size_batch, batch.get('vtx_rank'), state_tokens, 1, 1, _knn_active_slots,
                    enc_pack=_enc_pack_fwd, knn_pk=_knn_pack_fwd)
            # Already in grouped format, no need to reshape
            accel_flat = accel_grouped  # (B, Nobj, Nq, 3)
            accel_base_norm = accel_flat  # Network output: base acceleration (step_size-independent)

            # Verlet integration (dt_verlet=1)
            dt_squared = self.dt_verlet * self.dt_verlet
            with torch.amp.autocast('cuda', enabled=False):
                # No normalizer — network output is raw acceleration
                accel_base_denorm = accel_base_norm

                # ============================================================
                # Explicit step_size² scaling for multi-timescale training
                # ============================================================
                # Physical reasoning:
                #   - Network outputs "base acceleration" a_base (step_size-independent)
                #   - Actual acceleration = a_base × step_size²
                #   - This explicit scaling removes the burden from network to learn s² relationship
                #   - Benefit: Better generalization to unseen step_sizes, easier optimization
                # ============================================================
                if self._if_time_step and skip_step_size_batch is not None:
                    step_size_sq = (skip_step_size_batch ** 2).view(B, 1, 1, 1)  # (B,1,1,1)
                    accel_scaled_denorm = accel_base_denorm * step_size_sq  # (B,Nobj,Nq,3)
                    # Also scale normalized version for loss computation
                    accel_no_icp_norm_anchor = accel_base_norm * step_size_sq
                else:
                    accel_scaled_denorm = accel_base_denorm
                    accel_no_icp_norm_anchor = accel_base_norm

                # Build base next positions for all vertices as zero-accel extrapolation
                # No detach: allow gradients to flow through position history for long-horizon training
                next_pos_bt = 2 * curr_pos_bt - prev_pos_bt  # (B,Nobj,Nv,3)

                # Vectorized scatter using scatter_reduce (faster than per-object loop)
                # Step 1: Gather all anchor positions at once
                anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
                curr_anchors = torch.gather(curr_pos_bt, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)
                prev_anchors = torch.gather(prev_pos_bt, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)

                # Step 2: Compute next positions at anchors using Verlet (with scaled acceleration)
                next_anchors = accel_scaled_denorm * dt_squared + 2 * curr_anchors - prev_anchors  # (B,Nobj,Nq,3)

                # Step 3: Compute updates (delta from base prediction)
                base_anchors = torch.gather(next_pos_bt, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)
                anchor_updates = next_anchors - base_anchors  # (B,Nobj,Nq,3)

                # Mask padding objects' updates to zero
                # Padding objects have anchor_idx=0 which would scatter to wrong positions
                obj_mask_scatter = obj_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B,Nobj,1,1)
                anchor_updates = anchor_updates * obj_mask_scatter  # padding objects -> zero updates

                # Step 4: Vectorized scatter using scatter_reduce
                # Flatten batch and object dimensions for efficient scatter
                # Use reshape() to handle non-contiguous tensors
                B_Nobj_Nv = B * Nobj * Nv
                next_pos_flat = next_pos_bt.reshape(B_Nobj_Nv, 3)  # (B*Nobj*Nv, 3)

                # Create flat indices for scatter_reduce
                # anchor_idx is (B, Nobj, Nq), we need indices in flat (B*Nobj*Nv) space
                obj_offsets = torch.arange(Nobj, device=anchor_idx.device).view(1, Nobj, 1) * Nv  # (1,Nobj,1)
                batch_offsets = torch.arange(B, device=anchor_idx.device).view(B, 1, 1) * (Nobj * Nv)  # (B,1,1)
                flat_anchor_idx = anchor_idx + obj_offsets + batch_offsets  # (B,Nobj,Nq)
                flat_anchor_idx = flat_anchor_idx.reshape(B * Nobj * Nq)  # (B*Nobj*Nq,)

                # Flatten updates
                anchor_updates_flat = anchor_updates.reshape(B * Nobj * Nq, 3)  # (B*Nobj*Nq, 3)

                # Apply updates using scatter_reduce (more efficient than loop)
                next_pos_flat = torch.scatter_reduce(
                    next_pos_flat,
                    dim=0,
                    index=flat_anchor_idx.unsqueeze(-1).expand(-1, 3),
                    src=anchor_updates_flat,
                    reduce='sum',
                    include_self=True
                )

                # Reshape back
                next_pos_bt = next_pos_flat.reshape(B, Nobj, Nv, 3)

                # Per-object rigid projection using anchors (Kabsch on anchors), then apply to full Nv
                if registration_refs is None:
                    registration_refs = curr_pos_bt.detach().clone()  # (B,Nobj,Nv,3)

                # Vectorized Kabsch: Process ONLY valid objects (filter padding BEFORE SVD)
                # Step 1: Gather anchor positions for all objects at once
                anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
                last_anchor_all = torch.gather(registration_refs, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)
                curr_anchor_all = torch.gather(next_pos_bt, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)

                # Reshape to (B*Nobj, Nq, 3) for batch processing
                B_Nobj = B * Nobj
                last_anchor_flat = last_anchor_all.view(B_Nobj, Nq, 3)
                curr_anchor_flat = curr_anchor_all.view(B_Nobj, Nq, 3)

                # Filter padding objects before SVD; they would produce garbage R/t
                obj_mask_flat = obj_mask.view(B_Nobj)  # (B*Nobj,)
                valid_idx = _kabsch_valid_idx_fwd  # indices of valid objects (fixed within the forward)
                n_valid = valid_idx.shape[0]

                # Initialize R/t for ALL objects (identity R, zero t)
                R_flat = torch.eye(3, device=device, dtype=last_anchor_flat.dtype).unsqueeze(0).expand(B_Nobj, -1, -1).clone()
                t_vec_flat = torch.zeros(B_Nobj, 1, 3, device=device, dtype=last_anchor_flat.dtype)

                if n_valid > 0:
                    # Gather only valid objects' anchors
                    last_anchor_valid = last_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)
                    curr_anchor_valid = curr_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)


                    if self.if_roma:
                        # RoMa mode: batch process only valid objects
                        # RoMa requires FP32 for numerical stability
                        with torch.amp.autocast('cuda', enabled=False):
                            R_valid, t_vec_valid = roma.rigid_points_registration(
                                last_anchor_valid.float(), 
                                curr_anchor_valid.float()
                            )
                        R_valid = R_valid.float()  # Ensure FP32 for downstream
                        t_vec_valid = t_vec_valid.float().unsqueeze(1)  # (Nvalid, 1, 3)

                    else:
                        # Standard SVD Kabsch: batch process only valid objects
                        centroid_last = last_anchor_valid.mean(dim=1, keepdim=True)  # (Nvalid,1,3)
                        centroid_curr = curr_anchor_valid.mean(dim=1, keepdim=True)  # (Nvalid,1,3)
                        last_c = last_anchor_valid - centroid_last
                        curr_c = curr_anchor_valid - centroid_curr

                        # Batch SVD for valid objects only
                        H = torch.bmm(last_c.transpose(1, 2), curr_c)  # (Nvalid,3,3)
                        with torch.amp.autocast('cuda', enabled=False):
                            H_float = H.float()
                            U, S, Vt = torch.linalg.svd(H_float)  # Batch SVD!
                        U = U.to(H.dtype); Vt = Vt.to(H.dtype)
                        R_valid = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # (Nvalid,3,3)

                        # Fix reflections
                        with torch.amp.autocast('cuda', enabled=False):
                            det_R = torch.det(R_valid.float()).to(R_valid.dtype)
                        need_flip = det_R < 0
                        if need_flip.any():
                            Fm = torch.eye(3, device=R_valid.device, dtype=R_valid.dtype).unsqueeze(0).expand(n_valid, -1, -1).clone()
                            Fm[need_flip, 2, 2] = -1
                            R_valid = torch.bmm(torch.bmm(Vt.transpose(1, 2), Fm), U.transpose(1, 2))

                        # Translation
                        rotated_centroid = torch.bmm(R_valid, centroid_last.transpose(1, 2)).transpose(1, 2)
                        t_vec_valid = centroid_curr - rotated_centroid  # (Nvalid,1,3)

                    # Scatter valid R/t back to full tensors
                    R_flat[valid_idx] = R_valid
                    t_vec_flat[valid_idx] = t_vec_valid

                # Reshape back to (B,Nobj,3,3) and (B,Nobj,1,3)
                R_all = R_flat.view(B, Nobj, 3, 3)
                t_vec_all = t_vec_flat.view(B, Nobj, 1, 3)

                # ========== VECTORIZED TRANSFORMATION APPLICATION ==========
                # Apply R, t to all objects in a single bmm call (no per-object loop)
                # registration_refs: (B, Nobj, Nv, 3)
                # R_all: (B, Nobj, 3, 3)
                # t_vec_all: (B, Nobj, 1, 3)

                # Reshape for batch bmm: (B, Nobj, Nv, 3) -> (B*Nobj, Nv, 3)
                last_flat = registration_refs.view(B * Nobj, Nv, 3)
                R_flat = R_all.view(B * Nobj, 3, 3)
                t_flat = t_vec_all.view(B * Nobj, 1, 3)

                # Single bmm for ALL objects: y = R @ x + t (RoMa convention)
                # last_flat.transpose(1, 2): (B*Nobj, 3, Nv)
                # bmm result: (B*Nobj, 3, Nv)
                # transpose back + t: (B*Nobj, Nv, 3)
                y_flat = torch.bmm(R_flat, last_flat.transpose(1, 2)).transpose(1, 2) + t_flat
                # Use reshape() because transpose() makes tensor non-contiguous
                yproj_bt = y_flat.reshape(B, Nobj, Nv, 3).float()  # FP32 for numerical stability

                # Raw position BEFORE Kabsch/STE (for position loss supervision)
                next_pos_raw_norm = next_pos_bt

                # Apply gradient strategy based on transform type
                if self.if_roma:
                    # RoMa mode: fully differentiable, gradients flow through Kabsch
                    blended_next = yproj_bt
                else:
                    # Standard mode: straight-through estimator (STE)
                    # Use Kabsch value but only pass gradient to original prediction
                    blended_next = next_pos_bt + (yproj_bt - next_pos_bt).detach()

                # Recompute accelerations from blended positions (for acceleration_after_icp loss)
                accel_icp_denorm = (blended_next - 2 * curr_pos_bt + prev_pos_bt) / dt_squared
                accel_icp_norm = accel_icp_denorm

            # ========== INTERSECTION HANDLING ==========
            # Resolve object-ground and object-object penetrations (STE-differentiable)
            intersection_cfg = self.config.get('model', {}).get('intersection_handling', {})
            rigid_next = blended_next  # Kabsch output before the ground/object correction
            blended_next = handle_intersections(
                blended_next, obj_mask, vmask_t, intersection_cfg
            )

            # Update registration refs to the post-correction position (consistent with rollout); with
            # model.kabsch_reference 'fixed' every step fits the same reference mesh (the first current frame)
            if not self._kabsch_reference_fixed:
                registration_refs = blended_next.detach()

            # Store positions:
            # - STE blended positions for autoregressive rollout (more stable)
            # - Raw positions for loss supervision (direct gradient)
            # - ICP-transformed positions for position_after_icp loss
            all_pred_pos_for_rollout.append(blended_next)  # For next step input
            all_pred_pos_raw.append(next_pos_bt)  # For position_before_icp loss
            all_pred_pos_raw_norm.append(next_pos_raw_norm)  # For position_before_icp loss (normalized)
            # ICP/Kabsch-transformed positions for position_after_icp loss. after_kabsch_loss 'rigid': the Kabsch
            # output itself, the state the acceleration_after_icp loss uses (the stated after-Kabsch losses);
            # 'handled': the state after the intersection correction.
            blended_next_norm = rigid_next if self._after_kabsch_loss_rigid else blended_next
            all_pred_pos_icp_norm.append(blended_next_norm)  # For position_after_icp loss
            all_accel_no_icp_norm_anchor_seq.append(accel_no_icp_norm_anchor)

            # Save current features for next step's penetration prediction
            prev_obj_layers = all_obj_layers
            prev_state_layers = all_state_layers
            all_accel_icp_norm_seq.append(accel_icp_norm)

        # Stack time dimension
        # Design principle:
        # - Position outputs: RAW predictions (before Kabsch) AND ICP-transformed (after Kabsch)
        # - Acceleration outputs: before and after Kabsch transform
        # - Autoregressive rollout: uses ICP-transformed positions internally for stability
        output = {
            'obj_positions_denorm': torch.stack(all_pred_pos_raw, dim=1),      # (B,T-2,Nobj,Nv,3) - raw (before Kabsch)
            'obj_positions_norm': torch.stack(all_pred_pos_raw_norm, dim=1),    # (B,T-2,Nobj,Nv,3) - raw normalized (before Kabsch)
            'obj_positions_icp_norm': torch.stack(all_pred_pos_icp_norm, dim=1),  # (B,T-2,Nobj,Nv,3) - ICP-transformed normalized (after Kabsch)
            'obj_positions_ste': torch.stack(all_pred_pos_for_rollout, dim=1),  # (B,T-2,Nobj,Nv,3) - ICP blended (matches rollout)
            'obj_accelerations_no_icp_norm_anchor': torch.stack(all_accel_no_icp_norm_anchor_seq, dim=1),  # before Kabsch
            'obj_accelerations_norm': torch.stack(all_accel_icp_norm_seq, dim=1),  # after Kabsch
        }

        # Include penetration predictions for ALL vertices if collected
        penetration_enabled = self.config['model'].get('if_penetration', self.config['model'].get('if_directional_distance', False))
        if penetration_enabled and len(all_pen_pred_vertices_seq) > 0:
            try:
                output['obj_pen_pred_vertices'] = torch.stack(all_pen_pred_vertices_seq, dim=1)  # (B,T-2,Nobj,Nv,1)
            except Exception as _e:
                logger.warning(f"Failed to stack pen predictions: {_e}")

        if return_all:
            # Not storing states for multi-object yet
            pass

        return output
    
    def rollout(self, initial_batch, num_steps, dt=0.01, return_extras=True, warmup_frames=2):
        """
        Rollout predictions from initial conditions (multi-object).

        Args:
            initial_batch: dict from dataloader containing at least
                - obj_positions (B, T0, Nobj, Nv, 3)
                - valid_obj_mask (B, Nobj)
                - valid_vtx_mask (B, T0, Nobj, Nv)
                - ref_obj_positions (B or 1, Nobj, Nv, 3) [optional]
            num_steps: total frames to output (including provided GT warmup frames)
            dt: timestep (unused; integration handled in model forward)
            warmup_frames: number of GT warmup frames (default=2 for our model, HopNet uses 4)
        Returns:
            dict with 'obj_positions' of shape (B, num_steps, Nobj, Nv, 3)
                and optionally 'obj_body_dist_vec2' and 'obj_accelerations_anchor' if return_extras=True
        """
        self.eval()
        with torch.no_grad():
            if 'obj_positions' not in initial_batch:
                raise ValueError("rollout expects multi-object 'obj_positions' in initial_batch")

            obj_pos_all = initial_batch['obj_positions']  # (B,T0,Nobj,Nv,3)
            valid_obj_mask = initial_batch['valid_obj_mask']  # (B,Nobj)
            valid_vtx_mask_all = initial_batch['valid_vtx_mask']  # (B,T0,Nobj,Nv)

            # Read physics parameters and time step if enabled
            phys_params = None  # (B, Nobj, 3): [mass, friction, restitution]
            skip_step_size_batch = None  # (B,)
            if self._if_phys_param:
                if 'phys_params' not in initial_batch:
                    raise KeyError("rollout requires 'phys_params' in initial_batch when if_phys_param=true")
                phys_params = initial_batch['phys_params'].to(obj_pos_all.device)  # (B, Nobj, 3)
            if self._if_time_step:
                if 'skip_step_size' not in initial_batch:
                    raise KeyError("rollout requires 'skip_step_size' in initial_batch when if_time_step=true")
                skip_step_size_batch = initial_batch['skip_step_size'].to(obj_pos_all.device)  # (B,)

                # Optional debug logging of the rollout step size
                if_debug = self.config.get('data', {}).get('random_step_size', {}).get('if_debug', False)
                if if_debug:
                    logger.info(f"[DEBUG Rollout] step_size for rollout: {skip_step_size_batch.cpu().numpy().tolist()}, num_steps={num_steps}")

                # Validate step_size is within training range
                # Check if multi-timescale training is enabled and validate step_size
                random_step_config = self.config.get('data', {}).get('random_step_size', {})
                if random_step_config.get('enable', False):
                    trained_step_sizes = random_step_config.get('step_sizes', [1])  # e.g., [1, 2, 5, 10]
                    for b_idx, step_size_val in enumerate(skip_step_size_batch.cpu().numpy()):
                        if step_size_val not in trained_step_sizes:
                            logger.warning(
                                f"[Rollout Warning] Sample {b_idx}: step_size={step_size_val} is NOT in trained range {trained_step_sizes}!\n"
                                f"Model was only trained on step_sizes {trained_step_sizes}. Prediction may be inaccurate due to extrapolation."
                            )
                        elif if_debug:
                            logger.info(f"[Rollout Validation] Sample {b_idx}: step_size={step_size_val} is within trained range {trained_step_sizes}")

            B, T0, Nobj, Nv, _ = obj_pos_all.shape
            device = obj_pos_all.device

            # Require at least warmup_frames GT frames (minimum 2 for Verlet initialization)
            min_required = max(2, warmup_frames)
            if T0 < min_required:
                raise ValueError(f"rollout requires at least {min_required} GT frames (warmup_frames={warmup_frames}), got {T0}")

            # Reference pose: MUST be global frame 0 (same as training)
            if 'ref_obj_positions' not in initial_batch:
                raise ValueError("rollout requires 'ref_obj_positions' (global frame 0) in initial_batch")
            ref_pose = initial_batch['ref_obj_positions']  # (B?,Nobj,Nv,3)
            if ref_pose.dim() == 3:
                ref_pose = ref_pose.unsqueeze(0).expand(B, -1, -1, -1).to(device)
            else:
                ref_pose = ref_pose.to(device)

            # ========== Center scene by first frame (eval/rollout alignment) ==========
            # If enabled, subtract the global center of the first frame from all coordinates
            # so the rollout coordinate system matches the training setup.
            # center_t0 is stored per sample so visualization can apply the same shift to disk-loaded GT.
            _center_t0_per_sample = []  # Store per-sample center for visualization GT alignment
            if self.config.get('data', {}).get('center_scene_by_t0', False):
                for b in range(B):
                    vmask_b0 = valid_vtx_mask_all[b, 0]  # (Nobj, Nv)
                    v_t0 = obj_pos_all[b, 0][vmask_b0]  # Valid vertices at t=0
                    if v_t0.shape[0] > 0:
                        center_t0 = v_t0.mean(dim=0)  # (3,)
                        _center_t0_per_sample.append(center_t0.cpu().numpy())
                        
                        logger.info(f"[Center-T0 ROLLOUT] b={b}: center_t0={center_t0.cpu().numpy().tolist()}, "
                                   f"obj_pos range before: [{obj_pos_all[b].min().item():.4f}, {obj_pos_all[b].max().item():.4f}]")
                        
                        # Shift positions
                        obj_pos_all[b] = obj_pos_all[b] - center_t0
                        ref_pose[b] = ref_pose[b] - center_t0
                        
                        logger.info(f"[Center-T0 ROLLOUT] b={b} after: "
                                   f"obj_pos range: [{obj_pos_all[b].min().item():.4f}, {obj_pos_all[b].max().item():.4f}]")
                    else:
                        _center_t0_per_sample.append(None)
                logger.info(f"[Center-T0 ROLLOUT] Applied centering to {len(_center_t0_per_sample)} samples")

            # Require anchors for query and supervision (no fallback)
            if 'anchor_idx' not in initial_batch:
                raise KeyError("rollout requires 'anchor_idx' in initial_batch for anchor-based querying")
            anchor_idx = initial_batch['anchor_idx'].to(device).long()  # (B,Nobj,Nq)
            if anchor_idx.dim() != 3 or anchor_idx.shape[0] != B or anchor_idx.shape[1] != Nobj:
                raise ValueError(f"anchor_idx must be (B,Nobj,Nq); got {tuple(anchor_idx.shape)} vs (B={B},Nobj={Nobj})")
            Nq = int(anchor_idx.shape[2])

            # Initialize previous/current positions (use last 2 frames of warmup for Verlet)
            # For warmup_frames=2: use frame 0, 1
            # For warmup_frames=4: use frame 2, 3 (GT frames 0,1,2,3 then predict from frame 4)
            prev_idx = warmup_frames - 2
            curr_idx = warmup_frames - 1
            prev_pos_bt = obj_pos_all[:, prev_idx]  # (B,Nobj,Nv,3)
            curr_pos_bt = obj_pos_all[:, curr_idx]
            registration_refs = curr_pos_bt.detach().clone()

            # Initialize persistent state tokens (carry across steps like training)
            # When no_global_state=True, pass None to decoder
            if self._no_global_state:
                state_tokens = None
                state_pos = None
            else:
                state_tokens, state_pos = self.state_repr.get_state_and_positions(B, device)

            # Collect frames: include ALL GT warmup frames first
            out_seq = [obj_pos_all[:, i] for i in range(warmup_frames)]

            # Steps to predict beyond the provided GT frames in the output horizon
            remaining = max(0, num_steps - warmup_frames)

            # Collect per-step input body_dist_vec2 for visualization (curr frame at each step)
            # Only collect if return_extras=True (video gen) to save memory during RMSE eval
            all_dist_vec2_seq = [] if return_extras else None  # list of (B, Nobj, Nv, 3)
            # Collect predicted anchor accelerations (denormalized) for visualization
            all_accel_anchor_denorm_seq = [] if return_extras else None  # list of (B, Nobj, Nq, 3)

            # Previous step's features for penetration prediction (step>0)
            prev_obj_layers_rollout = None
            prev_state_layers_rollout = None

            # Optional CUDA-graph fast path (RIGIDFORMER_CUDA_GRAPH=1): after the
            # early steps whose vertex-mask index is still advancing, every step
            # executes an identical op stream on fixed shapes, so one step is
            # captured as a CUDA graph and replayed for the remainder. Cross-step
            # tensors are kept in fixed-address buffers that the graph itself
            # rotates, which makes each replay self-contained.
            use_cuda_graph = (os.environ.get('RIGIDFORMER_CUDA_GRAPH', '0') == '1'
                              and not return_extras and torch.cuda.is_available())
            # Constant scalars used inside the step; created once so the step body
            # performs no host-to-device transfers (required for graph capture).
            _attn_zero = torch.tensor(0.0, dtype=torch.float32, device=device)
            _attn_ninf = torch.tensor(float('-inf'), dtype=torch.float32, device=device)
            # Kabsch padding filter: valid_obj_mask is fixed for the whole rollout,
            # so the valid-object index set is computed once (nonzero would force a
            # host sync every step and cannot run under graph capture).
            _kabsch_mask_flat = valid_obj_mask.reshape(B * Nobj)
            _kabsch_valid_idx = _kabsch_mask_flat.nonzero(as_tuple=True)[0]
            _kabsch_n_valid = int(_kabsch_valid_idx.shape[0])
            use_cuda_graph = use_cuda_graph and bool(getattr(self, 'if_roma', False))
            graph_obj = None
            graph_obj2 = None
            graph_capture_step = T0 - 1  # first step index with the saturated mask
            if use_cuda_graph and remaining <= graph_capture_step + 1:
                use_cuda_graph = False

            # Progress bar for rollout (only show on rank 0)
            import torch.distributed as dist
            from tqdm import tqdm
            show_pbar = True
            if dist.is_available() and dist.is_initialized():
                show_pbar = (dist.get_rank() == 0)

            # Object slots holding a valid object in at least one sample; fixed for the rollout, so
            # computed once here (a host sync outside the step loop and any graph capture).
            _knn_active_slots_r = (torch.nonzero(valid_obj_mask.any(dim=0)).flatten().tolist()
                                   if self._knn_num_groups <= 1 else None)
            # Valid-vertex packs, one per mask index the loop uses (min(warmup_frames - 1 + step, T0 - 1)); built here, outside
            # the step loop and any graph capture.
            _knn_packs_r, _enc_packs_r = {}, {}
            if self._knn_packed or self._encoder_packed:
                _vr_r = initial_batch.get('vtx_rank')
                for _mi in range(1, valid_vtx_mask_all.shape[1]):
                    _vm_r = valid_vtx_mask_all[:, _mi]
                    if self._knn_packed:
                        _knn_packs_r[_mi] = knn_pack(_vm_r, valid_obj_mask)
                    if self._encoder_packed:
                        _enc_packs_r[_mi] = EncoderPack(
                            _vm_r.reshape(B * Nobj, Nv), _vr_r.reshape(B * Nobj, Nv) if _vr_r is not None else None,
                            len(self.shared_obj_encoder.hierarchical_layers), self.shared_obj_encoder.hierarchical_subsets, 1,
                            small=self.shared_obj_encoder.subset_small_objects)

            # Anchor validation. Anchors and vertex masks are fixed for the whole
            # rollout, so every per-step check is equivalent to checking each mask
            # index once up front; this keeps the step loop free of host syncs.
            for _mask_idx in range(1, T0):
                for i in range(Nobj):
                    vmask_curr = valid_vtx_mask_all[:, _mask_idx, i]      # (B,Nv)
                    idx_i_chk = anchor_idx[:, i]                          # (B,Nq)
                    mask_at_anchors = torch.gather(vmask_curr, dim=1, index=idx_i_chk)
                    obj_valid_b = valid_obj_mask[:, i]                    # (B,)
                    num_valid = vmask_curr.sum(dim=1)                     # (B,)
                    if int(((num_valid == 0) & obj_valid_b).sum().item()) > 0:
                        raise RuntimeError(f"Zero valid vertices for rollout at mask_idx={_mask_idx}, obj={i} for valid samples")
                    need = (num_valid < Nq) & obj_valid_b
                    if bool(need.any()):
                        min_valid = int(num_valid[obj_valid_b].min().item()) if bool(obj_valid_b.any()) else 0
                        raise RuntimeError(f"Not enough valid vertices for rollout at mask_idx={_mask_idx}, obj={i}: min_valid={min_valid} < anchor_k={Nq}")
                    bad_mask = (~mask_at_anchors) & obj_valid_b.unsqueeze(1)
                    if bool(bad_mask.any()):
                        bad = bad_mask.sum().item()
                        raise RuntimeError(f"Invalid anchors for rollout: {bad} indices invalid at mask_idx={_mask_idx}, obj={i}.")

            pbar = tqdm(range(remaining), desc='Rollout', disable=not show_pbar, leave=False)
            for step in pbar:
                rope_memo_reset()  # rotary angles are memoized within one step only (keeps graph capture self-contained)
                _capturing = use_cuda_graph and step == graph_capture_step
                if _capturing:
                    # Move cross-step carries into fixed-address buffers, then
                    # record this step. Kernels are only recorded during capture,
                    # so the step is materialized by an immediate replay below.
                    prev_pos_bt = prev_pos_bt.clone()
                    curr_pos_bt = curr_pos_bt.clone()
                    registration_refs = registration_refs.clone()
                    if prev_obj_layers_rollout is not None:
                        prev_obj_layers_rollout = [x.clone() for x in prev_obj_layers_rollout]
                    if prev_state_layers_rollout is not None:
                        prev_state_layers_rollout = [x.clone() for x in prev_state_layers_rollout]
                    torch.cuda.synchronize()
                    graph_obj = torch.cuda.CUDAGraph()
                    _graph_stream = torch.cuda.Stream()
                    _prev_stream = torch.cuda.current_stream()
                    torch.cuda.set_stream(_graph_stream)
                    # thread_local: NCCL watchdog / DataLoader threads must not
                    # poison the capture under DDP evaluation.
                    graph_obj.capture_begin(capture_error_mode='thread_local')
                # ========== STATE RESET (optional, to avoid train-test mismatch) ==========
                # When rollout_state_reset=True, reset state tokens at each step to match
                # training distribution (training only sees ~2 state updates per sequence).
                # This prevents state drift during long rollouts (e.g., 400 frames).
                if self._rollout_state_reset:
                    if self._no_global_state:
                        state_tokens = None
                        state_pos = None
                    else:
                        state_tokens, state_pos = self.state_repr.get_state_and_positions(B, device)

                # Compute velocities from last two frames
                vel_bt = curr_pos_bt - prev_pos_bt  # (B,Nobj,Nv,3)

                # Prepare far-away filler for invalid vertices in reference lists
                FAR = 1e6

                mask_idx_curr = min(warmup_frames - 1 + step, T0 - 1)  # current frame's mask

                # ========== VECTORIZED KNN FOR ALL OBJECTS AT ONCE ==========
                # curr_pos_bt: (B, Nobj, Nv, 3)
                # valid_vtx_mask_all[:, mask_idx_curr]: (B, Nobj, Nv)
                vmask_step = valid_vtx_mask_all[:, mask_idx_curr]  # (B, Nobj, Nv)

                # Compute KNN distance vectors for ALL objects simultaneously
                # Only computes for valid objects and valid vertices!
                if self._knn_packed:
                    dist_vec_all = compute_knn_packed(curr_pos_bt, _knn_packs_r[mask_idx_curr])
                else:
                    dist_vec_all = compute_knn_all_objects_vectorized(
                        curr_pos_bt, vmask_step, FAR, valid_obj_mask=valid_obj_mask,
                        num_groups=self._knn_num_groups, active_slots=_knn_active_slots_r
                    )

                # Apply plane constraint adjustment (vectorized)
                # Use reshape() instead of view() to handle non-contiguous tensors
                curr_pos_flat = curr_pos_bt.reshape(B, Nobj * Nv, 3)
                dist_vec_flat = dist_vec_all.reshape(B, Nobj * Nv, 3)
                dist_vec2_flat = distance_2_plane(curr_pos_flat, [0, 0, 0], "z", dist_vec_flat)
                dist_vec2_bt = dist_vec2_flat.reshape(B, Nobj, Nv, 3)

                # Store raw dist_vec2 for visualization (only if return_extras=True)
                if return_extras:
                    all_dist_vec2_seq.append(dist_vec2_bt)

                # Relative position (vectorized): curr_pos - ref_pose
                relpos_cat = curr_pos_bt - ref_pose  # (B, Nobj, Nv, 3)

                # Velocity (already in correct shape)
                vel_cat = vel_bt  # (B, Nobj, Nv, 3)

                # Dist vectors
                distvec_cat = dist_vec2_bt  # (B, Nobj, Nv, 3)

                # Get current step's vertex mask for all objects (vectorized)
                mask_idx_curr_all = mask_idx_curr
                vmask_curr_all = valid_vtx_mask_all[:, mask_idx_curr_all]  # (B,Nobj,Nv)
                vmask_curr_all_3d = vmask_curr_all.unsqueeze(-1)  # (B,Nobj,Nv,1)

                # Apply masks BEFORE normalization (vectorized across all objects)
                distvec_cat = torch.where(vmask_curr_all_3d, distvec_cat, torch.zeros_like(distvec_cat))
                vel_cat = torch.where(vmask_curr_all_3d, vel_cat, torch.zeros_like(vel_cat))
                relpos_cat = torch.where(vmask_curr_all_3d, relpos_cat, torch.zeros_like(relpos_cat))

                # Apply network-predicted directional sign to ALL vertices' dist_vec (before normalization)
                # Only for step>0 (student forcing steps with predicted positions, may have penetration)
                # For step=0, use GT positions from warmup (no penetration, sign=+1)
                penetration_enabled = self.config['model'].get('if_penetration', self.config['model'].get('if_directional_distance', False))
                if penetration_enabled and step > 0:
                    # Predict penetration sign using PREVIOUS step's features and CURRENT predicted positions
                    # curr_pos_bt = blended_next (AFTER Kabsch + intersection handling)
                    pen_pred_all_vertices = self.predictor.predict_penetration_vertices(
                        prev_obj_layers_rollout,  # Use previous step's decoder outputs
                        prev_state_layers_rollout,  # Use previous step's state features
                        pos_full=curr_pos_bt  # AFTER Kabsch transform (matches training forward)
                    )  # (B, Nobj, Nv, 1) in [-1,1]

                    # Apply predicted sign to dist_vec for ALL vertices
                    pen_sign_all = torch.sign(pen_pred_all_vertices + 1e-6)  # (B, Nobj, Nv, 1) in {-1,+1}
                    distvec_cat = distvec_cat * pen_sign_all  # (B, Nobj, Nv, 3)

                # No normalization — raw values, matching the forward path
                # (features are masked once, then the penetration sign is applied)

                # Broadcast physics parameters and time_step to vertex dimension if enabled (rollout)
                phys_broadcast = None  # (B, Nobj, Nv, 3)
                time_broadcast = None  # (B, Nobj, Nv, 2)

                if self._if_phys_param:
                    # Broadcast per-object physics params (B,Nobj,3) to all vertices (B,Nobj,Nv,3)
                    phys_broadcast = phys_params.unsqueeze(2).expand(B, Nobj, Nv, 3)  # (B,Nobj,Nv,3)
                    # Mask with vertex mask to ensure padding vertices get zero physics params
                    phys_broadcast = torch.where(vmask_curr_all_3d, phys_broadcast, torch.zeros_like(phys_broadcast))

                if self._if_time_step:
                    # Broadcast both s and s² to all objects and vertices (B,Nobj,Nv,2)
                    # Must match the forward path
                    step_size_s = skip_step_size_batch.float()  # (B,)
                    step_size_sq = (skip_step_size_batch ** 2).float()  # (B,)
                    time_features = torch.stack([step_size_s, step_size_sq], dim=-1)  # (B, 2)
                    time_broadcast = time_features.view(B, 1, 1, 2).expand(B, Nobj, Nv, 2)  # (B,Nobj,Nv,2)
                    # Mask for consistency
                    # vmask_curr_all_3d is (B,Nobj,Nv,1), expand to match (B,Nobj,Nv,2)
                    vmask_time = vmask_curr_all_3d.expand(-1, -1, -1, 2)  # (B,Nobj,Nv,2)
                    time_broadcast = torch.where(vmask_time, time_broadcast, torch.zeros_like(time_broadcast))

                # PointNet Encoder: process each object's Nv vertices -> per-object global feature
                # Input dimension depends on flags:
                #   Base: 9D (dist_vec + velocity + rel_pos)
                #   +if_phys_param=true: +3D
                #   +if_time_step=true (no FiLM): +2D (s and s²)
                # Split into components and reshape to (B*Nobj, Nv, D) for encoder
                dist_reshaped = distvec_cat.view(B * Nobj, Nv, 3)
                vel_reshaped = vel_cat.view(B * Nobj, Nv, 3)
                relpos_reshaped = relpos_cat.view(B * Nobj, Nv, 3)

                # Optional features

                phys_reshaped = None
                if self._if_phys_param:
                    phys_reshaped = phys_broadcast.view(B * Nobj, Nv, 3)

                time_reshaped = None
                if self._if_time_step and not self._film_conditioning:
                    time_reshaped = time_broadcast.view(B * Nobj, Nv, 2)  # (s, s²)

                # Prepare vertex mask for encoder pooling
                # vmask_curr_all_3d is (B,Nobj,Nv,1), need (B*Nobj, Nv)
                vtx_mask_for_encoder = vmask_curr_all_3d.squeeze(-1).reshape(B * Nobj, Nv)  # (B*Nobj, Nv)
                height_reshaped = None
                if self.shared_obj_encoder.height_input:
                    _h = torch.where(vmask_curr_all_3d, curr_pos_bt[..., 2:3], torch.zeros_like(curr_pos_bt[..., 2:3]))
                    height_reshaped = _h.reshape(B * Nobj, Nv, 1)
                rank_reshaped = None
                if self.shared_obj_encoder.hierarchical_subsets:
                    if 'vtx_rank' not in initial_batch:
                        raise KeyError("model.encoder.hierarchical_subsets requires data.fps_order: true (batch lacks 'vtx_rank')")
                    rank_reshaped = initial_batch['vtx_rank'].reshape(B * Nobj, Nv)

                # Call encoder with all features (encoder will concatenate them internally)
                # AVP: capture per-point features if AVP is enabled
                obj_features_encoded, _per_vtx_feat, _ = self.shared_obj_encoder(
                    dist_reshaped,
                    vel_reshaped,
                    relpos_reshaped,
                        phys_params=phys_reshaped,
                    time_step=time_reshaped,
                    valid_mask=vtx_mask_for_encoder,
                    height=height_reshaped,
                    vtx_rank=rank_reshaped,
                    pack=_enc_packs_r.get(mask_idx_curr_all) if self._encoder_packed else None
                )  # obj_features_encoded: (B*Nobj, enc_out_dim), _per_vtx_feat: (B*Nobj, backbone_dim, Nv)

                # AVP: Anchor-Vertex Pooling — enrich anchor queries with collision-aware features
                _avp_anchor_features = None  # Will be (B, Nobj, Nq, avp_output_dim) if AVP enabled
                if self.avp is not None and _per_vtx_feat is not None:
                    # _per_vtx_feat is (B*Nobj, backbone_dim, Nv) — Conv1d channels-first format
                    # Transpose to (B*Nobj, Nv, backbone_dim) for AVP
                    _pvf = _per_vtx_feat.transpose(1, 2)  # (B*Nobj, Nv, backbone_dim)
                    # Anchor positions in flat format
                    _curr_pos_flat = curr_pos_bt.reshape(B * Nobj, Nv, 3)  # (B*Nobj, Nv, 3)
                    _anchor_pos_flat = torch.gather(
                        _curr_pos_flat, dim=1,
                        index=anchor_idx.reshape(B * Nobj, -1).unsqueeze(-1).expand(-1, -1, 3)
                    )  # (B*Nobj, Nq, 3)
                    _avp_anchor_features = self.avp(
                        _pvf, _curr_pos_flat, _anchor_pos_flat, vtx_mask_for_encoder
                    )  # (B*Nobj, Nq, avp_output_dim)
                    _avp_anchor_features = _avp_anchor_features.view(B, Nobj, anchor_idx.shape[2], -1)  # (B, Nobj, Nq, avp_output_dim)
                    # Zero out padding objects (proj bias makes padding non-zero; consistent with other features)
                    _avp_anchor_features = _avp_anchor_features * valid_obj_mask.unsqueeze(-1).unsqueeze(-1).float()

                obj_features_encoded = obj_features_encoded.view(B, Nobj, -1)  # (B, Nobj, enc_out_dim)

                # Project to token dimension for Decoder
                obj_features = self.obj_feature_proj(obj_features_encoded)  # (B, Nobj, D)

                # FiLM: modulate encoder features with step-size [s, s²]
                if self.encoder_film is not None and skip_step_size_batch is not None:
                    time_feat_film = torch.stack([
                        skip_step_size_batch.float(),
                        (skip_step_size_batch ** 2).float()
                    ], dim=-1)  # (B, 2)
                    obj_features = self.encoder_film(obj_features, time_feat_film)  # (B, Nobj, D)

                # Object anchor positions for RoPE
                # Same approach as forward(): gather anchors, flatten to multi-point for RoPE.
                # Replace padding objects' positions with zeros before the gather:
                # padding objects have anchor_idx=0, which would gather arbitrary values into RoPE inputs.
                obj_mask_pos = valid_obj_mask.unsqueeze(-1).unsqueeze(-1)  # (B, Nobj, 1, 1)
                safe_curr_pos_bt = torch.where(obj_mask_pos, curr_pos_bt, torch.zeros_like(curr_pos_bt))

                idx_exp_3_for_decoder = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, Nobj, Nq, 3)
                obj_anchor_positions = torch.gather(safe_curr_pos_bt, dim=2, index=idx_exp_3_for_decoder)  # (B, Nobj, Nq, 3)
                # Flatten to (B, Nobj, Nq*3) for multi-point RoPE
                Nq = anchor_idx.shape[2]  # Number of anchors per object

                # ============== PE Type Routing (rollout) ==============
                # valid_vtx_mask_t is needed for geometry encoder
                valid_vtx_mask_t = valid_vtx_mask_all[:, mask_idx_curr]  # (B, Nobj, Nv)
                if self._pe_type in ('arope', 'se3'):
                    # SE(3) uses arope anchors for cross-attention RoPE; SE(3) bias computed separately
                    obj_anchor_flat = obj_anchor_positions.view(B, Nobj, Nq * 3)  # (B, Nobj, 12)
                elif self._pe_type in ('obb', 'pca'):
                    obj_anchor_flat = self._geometry_encoder(
                        safe_curr_pos_bt, valid_vtx_mask_t, obj_mask=valid_obj_mask
                    )  # (B, Nobj, 12)
                else:
                    raise ValueError(f"Unknown pe_type='{self._pe_type}' in rollout().")

                # ============== SE(3) Pairwise Bias Computation (rollout) ==============
                _se3_bias = None
                if self._pe_type == 'se3' and self._se3_bias_module is not None:
                    from src.models.se3_attention import extract_object_pose_features
                    _rotmats, _centers, _scales = extract_object_pose_features(
                        safe_curr_pos_bt, ref_pose, valid_vtx_mask_t, valid_obj_mask
                    )
                    _se3_bias = self._se3_bias_module(_rotmats, _centers, _scales, valid_obj_mask)

                # Decoder: SDPA additive mask + feature zeroing (same as forward)
                obj_attn_mask = torch.where(
                    valid_obj_mask.unsqueeze(1).unsqueeze(1),
                    _attn_zero,
                    _attn_ninf
                )
                pad_mask_float = valid_obj_mask.unsqueeze(-1).to(dtype=obj_features.dtype)
                obj_features = obj_features * pad_mask_float
                obj_anchor_flat = obj_anchor_flat * valid_obj_mask.unsqueeze(-1).to(dtype=obj_anchor_flat.dtype)

                # Per-layer FiLM: pass time_features for step-size conditioning in each layer
                _decoder_time_feat = time_feat_film if (self._film_conditioning and self._if_time_step) else None
                all_state_layers, all_obj_layers = self.decoder(
                    state_tokens,
                    obj_features,
                    state_pos,
                    obj_anchor_flat,
                    mask=obj_attn_mask,
                    se3_bias=_se3_bias,
                    time_features=_decoder_time_feat  # Per-layer FiLM
                )
                # No need to remove any tokens - all_obj_layers directly contains object features

                # Update state (skip when no_global_state)
                if not self._no_global_state:
                    state_tokens = all_state_layers[-1]

                # Build anchor query for Predictor (sample at anchor points Nq)
                # Query dimension depends on flags:
                #   Base: 9D (dist_vec + vel + rel_pos)
                #   +if_phys_param=true: +3D = 12D
                #   +if_time_step=true (no FiLM): +2D (s and s²)
                idx_exp_3 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
                distvec_at_anchors = torch.gather(distvec_cat, dim=2, index=idx_exp_3)
                vel_at_anchors = torch.gather(vel_cat, dim=2, index=idx_exp_3)
                relpos_at_anchors = torch.gather(relpos_cat, dim=2, index=idx_exp_3)

                # Build query features list
                query_features = [distvec_at_anchors, vel_at_anchors, relpos_at_anchors]

                # Add physics parameters at anchors if enabled
                if self._if_phys_param:
                    phys_at_anchors = torch.gather(phys_broadcast, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)
                    query_features.append(phys_at_anchors)

                # Add time step at anchors if enabled AND not using FiLM
                if self._if_time_step and not self._film_conditioning:
                    idx_exp_2 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,Nobj,Nq,2)
                    time_at_anchors = torch.gather(time_broadcast, dim=2, index=idx_exp_2)  # (B,Nobj,Nq,2)
                    query_features.append(time_at_anchors)

                # AVP: Append collision-aware anchor features to query
                if _avp_anchor_features is not None:
                    query_features.append(_avp_anchor_features)  # (B, Nobj, Nq, avp_output_dim)

                # Concatenate all query features
                anchor_query = torch.cat(query_features, dim=-1)  # (B,Nobj,Nq, 9/12 + avp_dim)

                # Extract anchor positions for RoPE (current frame positions at anchors)
                anchor_pos_at_curr = torch.gather(curr_pos_bt, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)

                # Penetration sign was already applied to all vertices' dist_vec;
                # the anchor query therefore carries the sign information as well.

                _ref_src = _object_centered(ref_pose, vmask_step.bool()) if self._ref_pose_object_centered else ref_pose
                if self._ref_pose_object_centered and not getattr(self, '_refpose_logged_roll', False):
                    self._refpose_logged_roll = True
                    print(f"[RefPose] object-centred (x/y only, z absolute) reference pose ACTIVE (rollout): |shift| max={((ref_pose - _ref_src).abs().max()).item():.3f} m", flush=True)
                ref_pose_at_anchors = torch.gather(_ref_src, dim=2, index=idx_exp_3)  # (B,Nobj,Nq,3)

                # Predict accelerations at anchors using grouped format (B, Nobj, Nq, *)
                accel_norm_anchor = self.predictor(
                    all_obj_layers,
                    anchor_query,           # (B, Nobj, Nq, 9/12)
                    ref_pose=ref_pose_at_anchors,  # (B, Nobj, Nq, 3)
                    query_pos=anchor_pos_at_curr,  # (B, Nobj, Nq, 3)
                    key_pos=obj_anchor_flat,        # (B, Nobj, Nq*3) - multi-point RoPE
                    multi_scale_state_features=None if self._no_global_state else all_state_layers,
                    state_pos=None if self._no_global_state else state_pos,
                    obj_mask=valid_obj_mask  # Object validity mask for padding
                )  # Returns (B, Nobj, Nq, 3)

                # Force FP32 for Verlet integration and position computation
                # (must match the training forward path for numerical stability)
                with torch.amp.autocast('cuda', enabled=False):
                    accel_base_denorm = accel_norm_anchor.float()

                    # Explicit step_size² scaling for multi-timescale rollout
                    # (must match the forward path)
                    if self._if_time_step and skip_step_size_batch is not None:
                        step_size_sq = (skip_step_size_batch.float() ** 2).view(B, 1, 1, 1)  # (B,1,1,1) FP32
                        accel_scaled_denorm = accel_base_denorm * step_size_sq  # FP32 * FP32 = FP32
                    else:
                        accel_scaled_denorm = accel_base_denorm

                    # Store scaled acceleration for visualization (matches forward behavior)
                    if return_extras:
                        all_accel_anchor_denorm_seq.append(accel_scaled_denorm)

                    # Verlet integration on anchors; initialize full next positions as zero-accel extrapolation
                    dt_squared = self.dt_verlet * self.dt_verlet
                    # Positions must be FP32 for numerical stability
                    curr_pos_bt_f = curr_pos_bt.float()
                    prev_pos_bt_f = prev_pos_bt.float()
                    next_pos_bt = 2 * curr_pos_bt_f - prev_pos_bt_f  # (B,Nobj,Nv,3) - FP32

                    # Vectorized scatter using scatter_reduce (same optimization as forward)
                    # Step 1: Gather all anchor positions at once (use FP32 positions)
                    anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
                    curr_anchors = torch.gather(curr_pos_bt_f, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)
                    prev_anchors = torch.gather(prev_pos_bt_f, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)

                    # Step 2: Compute next positions at anchors using Verlet (with scaled acceleration)
                    next_anchors = accel_scaled_denorm * dt_squared + 2 * curr_anchors - prev_anchors  # (B,Nobj,Nq,3)

                    # Step 3: Compute updates (delta from base prediction)
                    base_anchors = torch.gather(next_pos_bt, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)
                    anchor_updates = next_anchors - base_anchors  # (B,Nobj,Nq,3)

                    # Mask padding objects' updates to zero
                    # Padding objects have anchor_idx=0 which would scatter to wrong positions
                    obj_mask_scatter = valid_obj_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B,Nobj,1,1)
                    anchor_updates = anchor_updates * obj_mask_scatter  # padding objects -> zero updates

                    # Step 4: Vectorized scatter using scatter_reduce
                    # Flatten batch and object dimensions for efficient scatter
                    # Use reshape() to handle non-contiguous tensors
                    B_Nobj_Nv = B * Nobj * Nv
                    next_pos_flat = next_pos_bt.reshape(B_Nobj_Nv, 3)  # (B*Nobj*Nv, 3)

                    # Create flat indices for scatter_reduce
                    obj_offsets = torch.arange(Nobj, device=anchor_idx.device).view(1, Nobj, 1) * Nv  # (1,Nobj,1)
                    batch_offsets = torch.arange(B, device=anchor_idx.device).view(B, 1, 1) * (Nobj * Nv)  # (B,1,1)
                    flat_anchor_idx = anchor_idx + obj_offsets + batch_offsets  # (B,Nobj,Nq)
                    flat_anchor_idx = flat_anchor_idx.reshape(B * Nobj * Nq)  # (B*Nobj*Nq,)

                    # Flatten updates
                    anchor_updates_flat = anchor_updates.reshape(B * Nobj * Nq, 3)  # (B*Nobj*Nq, 3)

                    # Apply updates using scatter_reduce (more efficient than loop)
                    next_pos_flat = torch.scatter_reduce(
                        next_pos_flat,
                        dim=0,
                        index=flat_anchor_idx.unsqueeze(-1).expand(-1, 3),
                        src=anchor_updates_flat,
                        reduce='sum',
                        include_self=True
                    )

                    # Reshape back
                    next_pos_bt = next_pos_flat.reshape(B, Nobj, Nv, 3)

                # Per-object rigid projection using anchors; apply to full Nv
                # Vectorized Kabsch: Process ONLY valid objects (filter padding BEFORE SVD)
                # Step 1: Gather anchor positions for all objects at once
                anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
                last_anchor_all = torch.gather(registration_refs, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)
                curr_anchor_all = torch.gather(next_pos_bt, dim=2, index=anchor_idx_exp)  # (B,Nobj,Nq,3)

                # Reshape to (B*Nobj, Nq, 3) for batch processing
                B_Nobj = B * Nobj
                last_anchor_flat = last_anchor_all.view(B_Nobj, Nq, 3)
                curr_anchor_flat = curr_anchor_all.view(B_Nobj, Nq, 3)

                # Filter padding objects before SVD; they would produce garbage R/t
                # (indices precomputed before the loop; the mask never changes)
                valid_idx = _kabsch_valid_idx
                n_valid = _kabsch_n_valid

                # Initialize R/t for ALL objects (identity R, zero t)
                # FP32 so the RoMa FP32 output is not downcast during scatter
                R_flat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B_Nobj, -1, -1).clone()
                t_vec_flat = torch.zeros(B_Nobj, 1, 3, device=device, dtype=torch.float32)

                if n_valid > 0:
                    # Gather only valid objects' anchors
                    last_anchor_valid = last_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)
                    curr_anchor_valid = curr_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)

                    if _capturing:
                        # The SVD inside the rigid solve is not graph-capturable, so
                        # the step is recorded as two graphs around an eager solve.
                        # Close graph 1 here and execute it to materialize its outputs.
                        graph_obj.capture_end()
                        graph_obj.replay()
                        torch.cuda.current_stream().synchronize()

                    if self.if_roma:
                        # RoMa mode: batch process only valid objects
                        # RoMa requires FP32 for numerical stability
                        with torch.amp.autocast('cuda', enabled=False):
                            R_valid, t_vec_valid = roma.rigid_points_registration(
                                last_anchor_valid.float(), 
                                curr_anchor_valid.float()
                            )
                        R_valid = R_valid.float()  # Ensure FP32 for downstream
                        t_vec_valid = t_vec_valid.float().unsqueeze(1)  # (Nvalid, 1, 3)

                    else:
                        # Standard SVD Kabsch: batch process only valid objects
                        centroid_last = last_anchor_valid.mean(dim=1, keepdim=True)  # (Nvalid,1,3)
                        centroid_curr = curr_anchor_valid.mean(dim=1, keepdim=True)  # (Nvalid,1,3)
                        last_c = last_anchor_valid - centroid_last
                        curr_c = curr_anchor_valid - centroid_curr

                        # Batch SVD for valid objects only
                        H = torch.bmm(last_c.transpose(1, 2), curr_c)  # (Nvalid,3,3)
                        with torch.amp.autocast('cuda', enabled=False):
                            H_float = H.float()
                            U, S, Vt = torch.linalg.svd(H_float)  # Batch SVD!
                        U = U.to(H.dtype); Vt = Vt.to(H.dtype)
                        R_valid = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # (Nvalid,3,3)

                        # Fix reflections
                        with torch.amp.autocast('cuda', enabled=False):
                            det_R = torch.det(R_valid.float()).to(R_valid.dtype)
                        need_flip = det_R < 0
                        if need_flip.any():
                            Fm = torch.eye(3, device=R_valid.device, dtype=R_valid.dtype).unsqueeze(0).expand(n_valid, -1, -1).clone()
                            Fm[need_flip, 2, 2] = -1
                            R_valid = torch.bmm(torch.bmm(Vt.transpose(1, 2), Fm), U.transpose(1, 2))

                        # Translation
                        rotated_centroid = torch.bmm(R_valid, centroid_last.transpose(1, 2)).transpose(1, 2)
                        t_vec_valid = centroid_curr - rotated_centroid  # (Nvalid,1,3)

                    if _capturing:
                        # Fixed-address buffers carry the solve result into graph 2.
                        _rt_R_buf = R_valid.clone()
                        _rt_t_buf = t_vec_valid.clone()
                        R_valid = _rt_R_buf
                        t_vec_valid = _rt_t_buf
                        graph_obj2 = torch.cuda.CUDAGraph()
                        graph_obj2.capture_begin(pool=graph_obj.pool(), capture_error_mode='thread_local')
                    elif graph_obj2 is not None:
                        _rt_R_buf.copy_(R_valid)
                        _rt_t_buf.copy_(t_vec_valid)
                        R_valid = _rt_R_buf
                        t_vec_valid = _rt_t_buf

                    # Scatter valid R/t back to full tensors
                    R_flat[valid_idx] = R_valid
                    t_vec_flat[valid_idx] = t_vec_valid

                # Reshape back to (B,Nobj,3,3) and (B,Nobj,1,3)
                R_all = R_flat.view(B, Nobj, 3, 3)
                t_vec_all = t_vec_flat.view(B, Nobj, 1, 3)

                # ========== VECTORIZED TRANSFORMATION APPLICATION ==========
                # Apply R, t to all objects in a single bmm call (no per-object loop)
                # registration_refs: (B, Nobj, Nv, 3)
                # R_all: (B, Nobj, 3, 3)
                # t_vec_all: (B, Nobj, 1, 3)

                # Reshape for batch bmm: (B, Nobj, Nv, 3) -> (B*Nobj, Nv, 3)
                last_flat = registration_refs.view(B * Nobj, Nv, 3)
                R_flat = R_all.view(B * Nobj, 3, 3)
                t_flat = t_vec_all.view(B * Nobj, 1, 3)

                # Single bmm for ALL objects: y = R @ x + t (RoMa convention)
                # Convert to FP32 to prevent BF16 precision loss during autoregressive rollout
                y_flat = torch.bmm(R_flat, last_flat.transpose(1, 2)).transpose(1, 2) + t_flat
                # Use reshape() because transpose() makes tensor non-contiguous
                yproj_bt = y_flat.reshape(B, Nobj, Nv, 3).float()  # FP32 for numerical stability

                # Apply gradient strategy based on transform type
                if self.if_roma:
                    # RoMa mode: fully differentiable, gradients flow through Kabsch
                    blended_next = yproj_bt
                else:
                    # Standard mode: straight-through estimator (STE)
                    blended_next = next_pos_bt + (yproj_bt - next_pos_bt).detach()

                # ========== INTERSECTION HANDLING (EVAL) ==========
                intersection_cfg = self.config.get('model', {}).get('intersection_handling', {})
                blended_next = handle_intersections(
                    blended_next, valid_obj_mask, vmask_step, intersection_cfg
                )

                if _capturing:
                    # Rotate the carry buffers in place inside the captured region
                    # so every replay advances the state by one step.
                    prev_pos_bt.copy_(curr_pos_bt)
                    curr_pos_bt.copy_(blended_next)
                    if not self._kabsch_reference_fixed:
                        registration_refs.copy_(blended_next)
                    if prev_obj_layers_rollout is not None:
                        for _buf, _new in zip(prev_obj_layers_rollout, all_obj_layers):
                            _buf.copy_(_new)
                    if prev_state_layers_rollout is not None:
                        for _buf, _new in zip(prev_state_layers_rollout, all_state_layers):
                            _buf.copy_(_new)
                    graph_obj2.capture_end()
                    torch.cuda.set_stream(_prev_stream)
                    torch.cuda.synchronize()
                    graph_obj2.replay()  # capture only records; execute part 2 now
                    torch.cuda.synchronize()
                    out_seq.append(curr_pos_bt.clone())
                    pbar.close()
                    break
                # Registration refs use the post-collision position, matching the forward path ('advancing');
                # model.kabsch_reference 'fixed' keeps the last warm-up frame as the reference for every step
                if not self._kabsch_reference_fixed:
                    registration_refs = blended_next.detach()

                out_seq.append(blended_next)

                # Save current features for next step's penetration prediction
                prev_obj_layers_rollout = all_obj_layers
                prev_state_layers_rollout = all_state_layers

                prev_pos_bt = curr_pos_bt
                curr_pos_bt = blended_next

            if graph_obj is not None:
                target_len = warmup_frames + remaining
                while len(out_seq) < target_len:
                    graph_obj.replay()
                    with torch.amp.autocast('cuda', enabled=False):
                        _R_new, _t_new = roma.rigid_points_registration(
                            last_anchor_valid.float(),
                            curr_anchor_valid.float()
                        )
                    _rt_R_buf.copy_(_R_new.float())
                    _rt_t_buf.copy_(_t_new.float().unsqueeze(1))
                    graph_obj2.replay()
                    out_seq.append(curr_pos_bt.clone())

            if graph_obj is not None:
                # Tear the graphs down and return their private pools to the
                # allocator: the evaluator calls rollout once per batch and per
                # step size, and retired pools would otherwise accumulate.
                graph_obj.reset()
                if graph_obj2 is not None:
                    graph_obj2.reset()
                graph_obj = None
                graph_obj2 = None
                if os.environ.get('RIGIDFORMER_GRAPH_KEEPCACHE', '0') != '1':
                    torch.cuda.empty_cache()

            full = torch.stack(out_seq, dim=1)  # (B, 2+remaining, Nobj, Nv, 3)

            # Single numerical-health check per rollout (one host sync total, not
            # one per step): any non-finite prediction aborts loudly here.
            if not torch.isfinite(full).all():
                raise RuntimeError("Non-finite values in rollout output; "
                                   "inspect the checkpoint and input data.")

            # ========== Map back to original absolute space ==========
            # If center_scene_by_t0 was applied, add center_t0 back to predictions
            # so that the output is in the original world coordinate frame;
            # centering remains a purely internal operation of the model.
            if self.config.get('data', {}).get('center_scene_by_t0', False) and len(_center_t0_per_sample) > 0:
                for b in range(B):
                    if _center_t0_per_sample[b] is not None:
                        center_t0_tensor = torch.from_numpy(_center_t0_per_sample[b]).to(full.device)
                        full[b] = full[b] + center_t0_tensor
                logger.info(f"[Center-T0 ROLLOUT] Mapped predictions back to original absolute space")

            output = {
                'obj_positions': full
            }

            # Return body_dist_vec2 and accelerations only if requested (for visualization)
            # Skip during RMSE eval to save ~4GB memory on long rollouts
            if return_extras and remaining > 0:
                output['obj_body_dist_vec2'] = torch.stack(all_dist_vec2_seq, dim=1)
                output['obj_accelerations_anchor'] = torch.stack(all_accel_anchor_denorm_seq, dim=1)

            return output

            


class Trainer:
    """Training manager for the model."""
    
    def __init__(self, config, model, train_loader, val_loader, device='cuda', viz_params=None):
        self.config = config
        self.viz_params = viz_params or {}

        # Fast fail: start the video renderer if video saving is enabled.
        # Must run before any CUDA context is created (e.g. model.to(device))
        # to avoid EGL/CUDA memory conflicts.
        if self.config['training']['checkpoint'].get('save_video', False):
            # Only main process needs visualization
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                if self.config['training']['checkpoint'].get('video_renderer', 'filament') == 'polyscope':
                    self._polyscope_self_test()
                else:
                    self._filament_renderer()
        # Avoid moving DDP-wrapped model again
        if isinstance(model, DDP):
            self.model = model
        else:
            # Try to move model to device with OOM handling (only in inference mode)
            try:
                self.model = model.to(device)
                logger.info(f"Model moved to {device}")
            except torch.cuda.OutOfMemoryError as e:
                # Check if we're in inference-only mode (will be set later, so check train_loader)
                is_inference = len(train_loader) == 0
                if is_inference:
                    logger.error(f"CUDA OOM when moving model to GPU: {e}")
                    logger.warning("Inference-only mode: falling back to CPU execution")
                    device = 'cpu'
                    torch.cuda.empty_cache()
                    self.model = model.to(device)
                    logger.info("Model moved to CPU (execution will be slower)")
                else:
                    logger.error("CUDA OOM during training initialization")
                    logger.error("Reduce batch size or model size, or use a GPU with more memory")
                    raise
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    is_inference = len(train_loader) == 0
                    if is_inference:
                        logger.error(f"GPU memory error: {e}")
                        logger.warning("Inference-only mode: falling back to CPU execution")
                        device = 'cpu'
                        torch.cuda.empty_cache()
                        self.model = model.to(device)
                        logger.info("Model moved to CPU (execution will be slower)")
                    else:
                        logger.error("GPU OOM during training initialization")
                        logger.error("Reduce batch size or model size, or use a GPU with more memory")
                        raise
                else:
                    raise
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Cache for dataset mesh faces to avoid repeated disk IO
        self._faces_cache = {}
        # Cache for video datasets (per step_size) to avoid repeated indexing
        # Key: step_size, Value: MultiObjectDynamics dataset
        self._test_video_datasets = {}   # Test split: {step_size: dataset}
        self._train_video_datasets = {}  # Train split: {step_size: dataset}
        self._val_video_datasets = {}    # Val split: {step_size: dataset}

        # Read validation enable flag from config
        self.enable_validation = self.config['training'].get('enable_validation', True)
        if not self.enable_validation:
            logger.info("Validation disabled by config (training.enable_validation=false)")

        # Detect inference-only mode (empty train_loader)
        self.inference_only = len(train_loader) == 0
        if self.inference_only:
            logger.info("Inference-only mode detected (empty train_loader), skipping optimizer/scheduler setup")

        # Setup optimizer based on config type
        if not self.inference_only:
            optimizer_type = config['training']['optimizer'].get('type', None)
            if optimizer_type is None:
                raise ValueError(
                    "Optimizer type must be specified in training.optimizer.type\n"
                    "Supported: 'AdamW'"
                )
        else:
            optimizer_type = None

        if optimizer_type is None:
            # Inference-only mode, optimizer setup will be skipped
            pass
        elif optimizer_type.lower() == 'adamw':
            # Standard AdamW optimizer (matching baselines)
            opt_cfg = config['training']['optimizer']
            lr = float(opt_cfg.get('lr', 1e-4))
            betas = tuple(opt_cfg.get('betas', [0.9, 0.999]))
            weight_decay = float(opt_cfg.get('weight_decay', 0.01))
            # Adam's eps (torch default 1e-8). The anchor-space losses are in metres per frame^2, so gradients can
            # be small enough for eps to damp the step; training.optimizer.eps makes it a recorded per-run choice.
            adam_eps = float(opt_cfg.get('eps', 1e-8))

            # Split params: fusion layers may use a higher learning rate
            fusion_params = []
            other_params = []
            fusion_param_names = []
            # encoder_lr_mult > 1 gives the point encoder its own learning rate; used when
            # fine-tuning a checkpoint after changing the encoder's pooling semantics.
            enc_params = []
            enc_lr_mult = float(opt_cfg.get('encoder_lr_mult', 1.0))
            # decoder_lr_mult > 1 does the same for the Transformer decoder and the anchor predictor
            # (the modules that consume the rotary embedding).
            dec_params = []
            dec_lr_mult = float(opt_cfg.get('decoder_lr_mult', 1.0))
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'predictor.fusion_norm1' in name or 'predictor.fusion_w1' in name or 'predictor.fusion_g1' in name:
                    fusion_params.append(param)
                    fusion_param_names.append(name)
                elif enc_lr_mult != 1.0 and 'shared_obj_encoder' in name:
                    enc_params.append(param)
                elif dec_lr_mult != 1.0 and (name.startswith('decoder.') or name.startswith('predictor.')
                                             or name.startswith('module.decoder.') or name.startswith('module.predictor.')):
                    dec_params.append(param)
                else:
                    other_params.append(param)

            fusion_lr_mult = float(opt_cfg.get('fusion_lr_mult', 10.0))
            fusion_lr = lr * fusion_lr_mult

            if fusion_params or enc_params or dec_params:
                param_groups = [
                    {'params': other_params, 'lr': lr},
                ]
                if fusion_params:
                    param_groups.append({'params': fusion_params, 'lr': fusion_lr})
                if enc_params:
                    param_groups.append({'params': enc_params, 'lr': lr * enc_lr_mult})
                    logger.info(f"Encoder parameter group: {len(enc_params)} tensors at lr={lr * enc_lr_mult:.2e} (x{enc_lr_mult})")
                if dec_params:
                    param_groups.append({'params': dec_params, 'lr': lr * dec_lr_mult})
                    logger.info(f"Decoder/predictor parameter group: {len(dec_params)} tensors at lr={lr * dec_lr_mult:.2e} (x{dec_lr_mult})")
                # training.optimizer.decay_matrices_only: weight decay on weight matrices / convolution kernels only; norm
                # gains, biases, scalars (e.g. the anchor-vertex pooling log-radius) and the learned register-token
                # embeddings are not decayed. Every group is split in two with the same learning rate.
                if bool(opt_cfg.get('decay_matrices_only', False)):
                    _split = []
                    _embed = {id(q) for n_, q in model.named_parameters() if 'register_tokens' in n_}
                    for _g in param_groups:
                        _mat = [q for q in _g['params'] if q.ndim >= 2 and id(q) not in _embed]
                        _vec = [q for q in _g['params'] if q.ndim < 2 or id(q) in _embed]
                        _split.append({'params': _mat, 'lr': _g['lr']})
                        if _vec:
                            _split.append({'params': _vec, 'lr': _g['lr'], 'weight_decay': 0.0})
                    param_groups = _split
                    logger.info(f"[Optimizer] weight decay {weight_decay} on {sum(len(g['params']) for g in param_groups if 'weight_decay' not in g)} "
                                f"weight matrices only; {sum(len(g['params']) for g in param_groups if 'weight_decay' in g)} "
                                f"norm / bias / scalar tensors undecayed")
                self.optimizer = optim.AdamW(
                    param_groups,
                    betas=betas,
                    eps=adam_eps,
                    weight_decay=weight_decay
                )
                logger.info(f"Optimizer: AdamW with 2 groups (eps={adam_eps:g}):")
                logger.info(f"  - Base params ({len(other_params)}): lr={lr}")
                logger.info(f"  - Fusion params ({len(fusion_params)}): lr={fusion_lr} ({fusion_lr_mult}x)")
                for fn in fusion_param_names:
                    logger.info(f"    - {fn}")
            else:
                if bool(opt_cfg.get('decay_matrices_only', False)):
                    raise ValueError('training.optimizer.decay_matrices_only needs the grouped optimizer path')
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    betas=betas,
                    eps=adam_eps,
                    weight_decay=weight_decay
                )
                logger.info(f"Optimizer: AdamW (lr={lr}, betas={betas}, wd={weight_decay}, eps={adam_eps:g})")
        else:
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}\n"
                "Supported: 'AdamW'"
            )
        
        # Setup scheduler based on config (skip in inference-only mode)
        if not self.inference_only:
            scheduler_type = config['training']['scheduler'].get('type', 'cosine')
            if scheduler_type == 'constant':
                # No learning rate scheduling - keep constant
                self.scheduler = optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: 1.0  # Keep learning rate constant
                )
            elif scheduler_type == 'cosine':
                T_max = config['training']['max_epochs']
                # Support both keys 'min_lr' and 'eta_min'
                eta_min = float(config['training']['scheduler'].get('eta_min', config['training']['scheduler'].get('min_lr', 1e-6)))
                warmup_epochs = int(config['training']['scheduler'].get('warmup_epochs', 0))
                
                if warmup_epochs > 0:
                    # Use start_factor directly (recommended: 0.1 = start from 10% of base_lr)
                    start_factor = float(config['training']['scheduler'].get('start_factor', 0.1))

                    logger.info(f"Initializing Scheduler with Warmup: {warmup_epochs} epochs, start_factor={start_factor}")
                    
                    try:
                        from torch.optim.lr_scheduler import LinearLR, SequentialLR
                        
                        # Warmup phase
                        warmup_scheduler = LinearLR(
                            self.optimizer, 
                            start_factor=start_factor, 
                            end_factor=1.0, 
                            total_iters=warmup_epochs
                        )
                        
                        # Cosine phase (starts after warmup)
                        # T_max for cosine is the remaining epochs
                        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer,
                            T_max=max(1, T_max - warmup_epochs),
                            eta_min=eta_min
                        )
                        
                        self.scheduler = SequentialLR(
                            self.optimizer, 
                            schedulers=[warmup_scheduler, cosine_scheduler], 
                            milestones=[warmup_epochs]
                        )
                    except ImportError:
                        logger.warning("SequentialLR/LinearLR not found (PyTorch < 1.10?). Fallback to simple CosineAnnealingLR without warmup.")
                        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer,
                            T_max=T_max,
                            eta_min=eta_min
                        )
                else:
                    # Standard Cosine without warmup
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=T_max,
                        eta_min=eta_min
                    )
            else:
                # Default to cosine
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config['training']['max_epochs'],
                    eta_min=float(config['training']['scheduler'].get('eta_min', 1e-6))
                )
        else:
            # Inference-only mode: no optimizer/scheduler needed
            self.optimizer = None
            self.scheduler = None
        
        # Setup loss functions
        # Use SmoothL1Loss (Huber loss) for more stable gradients
        self.position_loss = nn.SmoothL1Loss()  # Default beta=1.0
        self.velocity_loss = nn.SmoothL1Loss()
        self.acceleration_loss = nn.SmoothL1Loss()
        
        # Loss weights
        self.loss_weights = config['training']['loss_weights']

        # Learnable transform checks
        self.is_learnable_transform = (
            self.config.get('model', {}).get('transform', {}).get('method', 'kabsch') == 'learnable'
        )
        self.require_gt_motion = bool(self.config.get('data', {}).get('load_gt_motion', False))
        self._did_learnable_assert = False
        
        # Setup logging and checkpointing with same timestamp (allow override via ENV)
        timestamp = os.environ.get('EXP_TIMESTAMP', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.timestamp = timestamp  # Store for later use
        
        # Logging directory with optional experiment tag
        exp_tag = self.config.get('experiment_tag', None)
        log_base = Path(config['paths']['log_dir']) / exp_tag if exp_tag else Path(config['paths']['log_dir'])
        self.log_dir = log_base / f'run_{timestamp}'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Only rank0 writes TensorBoard to avoid 8x event files in DDP
        def _is_main_proc():
            return (not (dist.is_available() and dist.is_initialized())) or dist.get_rank() == 0

        self.is_main = _is_main_proc()

        class _NoopWriter:
            def add_scalar(self, *args, **kwargs):
                return
            def add_text(self, *args, **kwargs):
                return
            def close(self):
                return

        if self.is_main and self.config.get('logging', {}).get('use_tensorboard', True):
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = _NoopWriter()
        
        # Optional file logging (disabled by default); only rank0 writes
        if self.is_main and self.config.get('logging', {}).get('file_logging', False):
            # Put file log under the run directory next to TensorBoard
            log_file_path = self.log_dir / 'training.log'
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            # Add to root logger so ALL modules (evaluator, dataloader, etc.) write to file
            logging.getLogger().addHandler(file_handler)
            logger.info(f"  - Log file: {log_file_path}")
        
        # Checkpoint directory with same timestamp and optional tag
        base_checkpoint_dir = Path(config['paths']['checkpoint_dir']) / exp_tag if exp_tag else Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir = base_checkpoint_dir / f'exp_{timestamp}'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Log the experiment directories
        logger.info(f"Experiment {timestamp} directories:")
        logger.info(f"  - TensorBoard: {self.log_dir}")
        # (log file path printed only when file_logging enabled)
        logger.info(f"  - Checkpoints: {self.checkpoint_dir}")

        # Save config to log directory (run_XXXXXX) for reproducibility
        config_save_path = self.log_dir / 'config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"  - Config saved to: {config_save_path}")
        
        self.best_val_loss = float('inf')
        self.epoch = 0
        # First epoch of the training loop. Stays 0 for a fresh run or a weights-only warm start;
        # set to (completed epoch + 1) when a full training checkpoint is resumed.
        self.start_epoch = 0

        # Optional exponential moving average of the weights (training.ema: {enable, decay}).
        # Rank 0 only: after DDP's all-reduce every rank holds identical weights. Off by default,
        # in which case training is unchanged.
        self.ema = None
        _ema_cfg = config['training'].get('ema', {}) or {}
        if bool(_ema_cfg.get('enable', False)) and not self.inference_only and self.is_main:
            _ema_base = self.model.module if isinstance(self.model, DDP) else self.model
            self.ema = WeightEMA(_ema_base, float(_ema_cfg.get('decay', 0.9999)))
            logger.info(f"[EMA] weight averaging ACTIVE on rank 0: decay={self.ema.decay} "
                        f"({len(self.ema.shadow)} state entries, fp32 shadow)")

        # GPU Augmentation (rotation) - only for training
        if not self.inference_only:
            from src.data.gpu_augmentation import GPUAugmentation
            self.gpu_augmentation = GPUAugmentation(config, self.device)
            logger.info(f"GPU augmentation initialized (rotation={self.gpu_augmentation.rotation_enabled})")
        else:
            self.gpu_augmentation = None

        # Scheduled Sampling configuration
        self.scheduled_sampling_config = config['training'].get('scheduled_sampling', {})
        self.use_scheduled_sampling = self.scheduled_sampling_config.get('enabled', False)
        
        if self.use_scheduled_sampling:
            logger.info(f"Scheduled Sampling enabled with mode: {self.scheduled_sampling_config.get('mode', 'linear')}")

        # ========== COMPREHENSIVE CONFIG LOGGING ==========
        loss_w = config['training'].get('loss_weights', {})
        ss_cfg = self.scheduled_sampling_config
        logger.info("=" * 70)
        logger.info("[CONFIG SUMMARY] Training Configuration at Startup:")
        logger.info(f"  Loss Weights:")
        logger.info(f"    position_before_icp:     {loss_w.get('position_before_icp', 'N/A')}")
        logger.info(f"    position_after_icp:      {loss_w.get('position_after_icp', 'N/A')}")
        logger.info(f"    acceleration_before_icp: {loss_w.get('acceleration_before_icp', 'N/A')}")
        logger.info(f"    acceleration_after_icp:  {loss_w.get('acceleration_after_icp', 'N/A')}")
        logger.info(f"    penetration_sign:        {loss_w.get('penetration_sign', 'N/A')}")
        logger.info(f"    if_time_step_scale:      {loss_w.get('if_time_step_scale', False)}")
        logger.info(f"  Scheduled Sampling:")
        logger.info(f"    enabled:          {ss_cfg.get('enabled', False)}")
        logger.info(f"    mode:             {ss_cfg.get('mode', 'N/A')}")
        logger.info(f"    teacher_epochs:   {ss_cfg.get('teacher_epochs', 'N/A')}")
        logger.info(f"    ramping_epochs:   {ss_cfg.get('ramping_epochs', 'N/A')}")
        logger.info(f"  Model Features:")
        # Access model attrs (model_ref = unwrapped model, self.model may be DDP-wrapped)
        model_ref = model if not hasattr(model, 'module') else model.module
        logger.info(f"    if_phys_param:    {getattr(model_ref, '_if_phys_param', 'N/A')}")
        logger.info(f"    if_time_step:     {getattr(model_ref, '_if_time_step', 'N/A')}")
        logger.info(f"    film_conditioning:{getattr(model_ref, '_film_conditioning', 'N/A')}")
        logger.info(f"    encoder_film:     {getattr(model_ref, 'encoder_film', None) is not None}")
        logger.info(f"    encoder_input_dim:{getattr(model_ref, '_encoder_input_dim', 'N/A')}")
        logger.info(f"    pe_type:          {getattr(model_ref, '_pe_type', 'N/A')}")
        logger.info(f"    if_roma:          {getattr(model_ref, 'if_roma', 'N/A')}")
        logger.info(f"  Training:")
        logger.info(f"    sequence_length:  {config['data'].get('sequence_length', 'N/A')}")
        logger.info(f"    eval_warmup:      {config.get('evaluation', {}).get('eval_warmup_frames', 'N/A')}")
        logger.info(f"    optimizer:        {config['training']['optimizer'].get('type', 'N/A')}")
        logger.info(f"    lr:               {config['training']['optimizer'].get('lr', 'N/A')}")
        logger.info("=" * 70)



        # Initialize evaluator for RMSE metrics
        from src.training.evaluator import Evaluator
        self.evaluator = Evaluator(self)
        logger.info(f"Evaluator initialized: metrics_enabled={self.evaluator.metrics_enabled}, "
                   f"print_every_n_epochs={self.evaluator.print_every_n_epochs}")

        # Ensure DDP sampler epoch set before first epoch
        try:
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.epoch)
            if hasattr(self.val_loader, 'sampler') and hasattr(self.val_loader.sampler, 'set_epoch'):
                self.val_loader.sampler.set_epoch(self.epoch)
        except Exception:
            pass

    def _filament_renderer(self):
        """The process's offscreen renderer for videos (src/utils/rollout_render.py), created on first use."""
        from src.utils import rollout_render
        ck = self.config['training']['checkpoint']
        width, height = (int(v) for v in ck.get('video_size', [1280, 720]))
        return rollout_render.get_renderer(width, height, int(ck.get('video_supersample', 2)),
                                           floor=ck.get('video_floor', 'light'))

    def _polyscope_self_test(self):
        """Attempt to initialize Polyscope headless EGL once at startup; raise on failure."""
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault('POLYSCOPE_BACKEND', 'openGL3_egl')
        os.environ.setdefault('POLYSCOPE_ALLOW_HEADLESS', '1')
        import polyscope as ps
        # Allow headless backends and select egl before init
        if hasattr(ps, 'set_allow_headless'):
            ps.set_allow_headless(True)
        # Set EGL device index to match CUDA_VISIBLE_DEVICES (for multi-GPU systems)
        egl_device_id = os.environ.get('EGL_DEVICE_ID', None)
        if egl_device_id is not None and hasattr(ps, 'set_egl_device_index'):
            ps.set_egl_device_index(int(egl_device_id))
            logger.info(f"Polyscope EGL device index set to: {egl_device_id}")
        # Initialize the EGL backend directly
        ps.init('openGL3_egl')
        # Shutdown immediately; we only test availability
        if hasattr(ps, 'shutdown'):
            ps.shutdown()
        logger.info("Polyscope headless (EGL) initialization test passed")
        
    def get_teacher_forcing_ratio(self, epoch):
        """Calculate teacher forcing ratio based on scheduled sampling configuration."""
        if not self.use_scheduled_sampling:
            return 1.0  # Always use teacher forcing if disabled
        
        config = self.scheduled_sampling_config
        mode = config.get('mode', 'linear')
        
        if mode == 'three_stage':
            # Three-stage schedule like character-motion-vaes
            teacher_epochs = config.get('teacher_epochs', 100)
            ramping_epochs = config.get('ramping_epochs', 200)
            
            if epoch < teacher_epochs:
                # Pure teacher forcing
                return 1.0
            elif epoch < teacher_epochs + ramping_epochs:
                # Linear ramping
                progress = (epoch - teacher_epochs) / ramping_epochs
                return 1.0 - progress  # From 1.0 to 0.0
            else:
                # Pure student forcing
                return 0.0
                
        elif mode == 'linear':
            # Linear decay
            start_epoch = config.get('start_epoch', 100)
            end_epoch = config.get('end_epoch', 1000)
            min_ratio = config.get('min_teacher_ratio', 0.1)
            max_ratio = config.get('max_teacher_ratio', 1.0)
            
            if epoch < start_epoch:
                return max_ratio
            elif epoch >= end_epoch:
                return min_ratio
            else:
                progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                return max_ratio - progress * (max_ratio - min_ratio)
                
        elif mode == 'exponential':
            # Exponential decay
            start_epoch = config.get('start_epoch', 100)
            decay_rate = config.get('decay_rate', 0.99)
            min_ratio = config.get('min_teacher_ratio', 0.1)
            
            if epoch < start_epoch:
                return 1.0
            else:
                epochs_since_start = epoch - start_epoch
                ratio = decay_rate ** epochs_since_start
                return max(ratio, min_ratio)
                
        elif mode == 'sigmoid':
            # Inverse sigmoid decay
            start_epoch = config.get('start_epoch', 100)
            k = config.get('k', 5.0)
            min_ratio = config.get('min_teacher_ratio', 0.1)
            
            if epoch < start_epoch:
                return 1.0
            else:
                epochs_since_start = epoch - start_epoch
                ratio = k / (k + np.exp(epochs_since_start / k))
                return max(ratio, min_ratio)
        else:
            logger.warning(f"Unknown scheduled sampling mode: {mode}, using teacher forcing")
            return 1.0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        # AMP is required for Flash Attention (fp16 or bf16)
        use_amp = True
        # Check for BF16 support (preferred on Ampere and newer)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16)) # Scaler needed only for fp16
        total_loss = 0
        num_batches = 0
        # Per-iteration linear warmup from zero (training.scheduler.warmup_iter_epochs, in epochs).
        # A fresh Adam state takes uniform +-lr steps on every parameter for its first ~1000
        # iterations; ramping the rate from zero keeps a converged warm start intact.
        _wu_frac = float(self.config['training'].get('scheduler', {}).get('warmup_iter_epochs', 0.0))
        _iters_per_epoch = len(self.train_loader)
        _wu_iters = int(round(_wu_frac * _iters_per_epoch))
        _epoch_base_lrs = [g['lr'] for g in self.optimizer.param_groups]
        # Component accumulators (averaged per epoch)
        comp_sum_pos_before = 0.0
        comp_sum_pos_after = 0.0
        comp_sum_acc_after = 0.0
        comp_sum_acc_before = 0.0
        comp_sum_edge = 0.0
        comp_sum_pen = 0.0
        comp_sum_pen_acc = 0.0


        # Compute teacher forcing ratio from scheduled sampling config
        if self.use_scheduled_sampling:
            teacher_forcing_ratio = self.get_teacher_forcing_ratio(self.epoch)
        else:
            teacher_forcing_ratio = 1.0  # Scheduled sampling off -> pure teacher forcing

        if self.is_main:
            logger.info(f"[Train Epoch {self.epoch}] teacher_forcing_ratio={teacher_forcing_ratio:.3f} "
                       f"(scheduled_sampling={'ON' if self.use_scheduled_sampling else 'OFF'})")

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.epoch}', disable=not self.is_main)
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Apply GPU augmentation (rotation) - only for training
            if self.gpu_augmentation is not None:
                batch = self.gpu_augmentation(batch, split='train')

            # Forward pass (Mixed Precision)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                predictions = self.model(batch, teacher_forcing_ratio=teacher_forcing_ratio)

            # Access underlying module when using DDP
            _model_ref = self.model.module if isinstance(self.model, DDP) else self.model
            # For multi-object batches, we will compute GT inside the multi-object loss branch below
            
            # Loss aggregation
            if 'obj_positions_denorm' in predictions:
                # Multi-object path: aggregate across all valid vertices using masks
                import torch.nn.functional as F
                T0 = batch['obj_positions'].shape[1]
                # obj_positions: (B, T, Nobj, Nv, 3)
                obj_mask = batch['valid_obj_mask'].unsqueeze(1).unsqueeze(-1)  # (B,1,Nobj,1)
                # Position predictions align to frames 2..T0-1
                vmask_pos = batch['valid_vtx_mask'][:, 2:T0]  # (B,T-2,Nobj,Nv)
                full_mask_pos = (vmask_pos & obj_mask).to(predictions['obj_positions_norm'].dtype)
                # Acceleration predictions align to centers 1..T0-2 (curr index in forward)
                vmask_acc = batch['valid_vtx_mask'][:, 1:T0-1]  # (B,T-2,Nobj,Nv)
                full_mask_acc = (vmask_acc & obj_mask).to(predictions['obj_positions_norm'].dtype)

                # ============================================================
                # Anchor-only losses: all losses are computed at the Nq anchor points.
                # Rationale: the network predicts accelerations at anchors, Kabsch R,t
                # come from anchors, and all Nv vertices are a deterministic function
                # of the anchors (rigid body). A full-vertex loss would be redundant
                # and dilute the gradient signal.
                # ============================================================

                # Prepare anchor index for gathering (shared across all losses)
                if 'anchor_idx' not in batch:
                    raise KeyError("Missing 'anchor_idx' in batch for anchor-based supervision")
                anchor_idx_b = batch['anchor_idx']  # (B,Nobj,Nq)
                Bsz = batch['obj_positions'].shape[0]
                Tm2_pos = T0 - 2  # Number of prediction steps
                Nq_loss = anchor_idx_b.shape[2]

                # Anchor gather index: (B,Nobj,Nq) -> (B,T-2,Nobj,Nq,3) for 5D tensors
                anchor_idx_pos = anchor_idx_b.unsqueeze(1).unsqueeze(-1).expand(
                    -1, Tm2_pos, -1, -1, 3
                )  # (B,T-2,Nobj,Nq,3)

                # Anchor mask: obj_mask only (all anchors valid by construction for valid objects)
                # obj_mask is (B,1,Nobj,1) -> (B,T-2,Nobj,Nq)
                anchor_mask_pos = obj_mask.expand(-1, Tm2_pos, -1, Nq_loss).to(predictions['obj_positions_norm'].dtype)
                # Divisor of the anchor losses: this rank's valid-anchor count, or with training.loss_weights.
                # global_anchor_mean the global count / world size, so that DDP's gradient average is the gradient of
                # the mean over every valid anchor of the global batch (ranks hold different numbers of objects)
                _anchor_count = anchor_mask_pos.float().sum()
                if (self.config['training']['loss_weights'].get('global_anchor_mean', False)
                        and dist.is_available() and dist.is_initialized()):
                    _anchor_count = _anchor_count.detach().clone()
                    dist.all_reduce(_anchor_count, op=dist.ReduceOp.SUM)
                    _anchor_count = _anchor_count / dist.get_world_size()

                # 1) Position loss BEFORE Kabsch (anchor-only)
                gt_pos = batch['obj_positions'][:, 2:T0]  # (B,T-2,Nobj,Nv,3)
                gt_pos_norm = gt_pos

                # Gather at anchors
                gt_pos_anchor = torch.gather(gt_pos_norm, dim=3, index=anchor_idx_pos)  # (B,T-2,Nobj,Nq,3)
                pred_pos_anchor = torch.gather(predictions['obj_positions_norm'], dim=3, index=anchor_idx_pos)  # (B,T-2,Nobj,Nq,3)

                # Compute position error at anchors
                pos_error = pred_pos_anchor - gt_pos_anchor  # (B,T-2,Nobj,Nq,3)

                # Apply step_size² normalization if enabled
                if_time_step_scale_pos = self.config['training']['loss_weights'].get('if_time_step_scale', True)
                if if_time_step_scale_pos:
                    if 'skip_step_size' not in batch:
                        raise KeyError(
                            "Missing 'skip_step_size' in batch for position loss normalization!\n"
                            "Multi-timescale training with if_time_step_scale=true requires step_size."
                        )
                    step_sizes_pos = batch['skip_step_size'].float()  # (B,)
                    step_size_scale_sq_pos = step_sizes_pos ** 2  # (B,)
                    scale_pos = step_size_scale_sq_pos.view(Bsz, 1, 1, 1, 1)  # (B,1,1,1,1)
                    pos_error_normalized = pos_error / scale_pos  # (B,T-2,Nobj,Nq,3)
                else:
                    pos_error_normalized = pos_error

                pos_l1 = F.smooth_l1_loss(
                    pos_error_normalized, torch.zeros_like(pos_error_normalized), reduction='none'
                ).sum(dim=-1)  # (B,T-2,Nobj,Nq)
                position_loss_before_icp = (pos_l1.float() * anchor_mask_pos.float()).sum() / (_anchor_count + 1e-8)

                # 1b) Position loss AFTER Kabsch (anchor-only)
                if 'obj_positions_icp_norm' in predictions:
                    pred_pos_icp_anchor = torch.gather(predictions['obj_positions_icp_norm'], dim=3, index=anchor_idx_pos)  # (B,T-2,Nobj,Nq,3)
                    pos_error_icp = pred_pos_icp_anchor - gt_pos_anchor  # (B,T-2,Nobj,Nq,3)
                    if if_time_step_scale_pos:
                        pos_error_icp_normalized = pos_error_icp / scale_pos
                    else:
                        pos_error_icp_normalized = pos_error_icp
                    pos_icp_l1 = F.smooth_l1_loss(
                        pos_error_icp_normalized, torch.zeros_like(pos_error_icp_normalized), reduction='none'
                    ).sum(dim=-1)  # (B,T-2,Nobj,Nq)
                    position_loss_after_icp = (pos_icp_l1.float() * anchor_mask_pos.float()).sum() / (_anchor_count + 1e-8)
                else:
                    position_loss_after_icp = torch.tensor(0.0, device=self.device)

                # 2) Acceleration losses in normalized space
                gt_prev = batch['obj_positions'][:, 0:T0-2]
                gt_curr = batch['obj_positions'][:, 1:T0-1]
                gt_next = batch['obj_positions'][:, 2:T0]
                gt_accel_den = gt_next - 2 * gt_curr + gt_prev  # (B,T-2,Nobj,Nv,3)
                gt_accel_norm = gt_accel_den

                # 2) Acceleration losses (anchor-only for both before and after Kabsch)
                a_icp_norm = predictions['obj_accelerations_norm']  # (B,T-2,Nobj,Nv,3) — after Kabsch
                if 'obj_accelerations_no_icp_norm_anchor' not in predictions:
                    raise KeyError("Missing 'obj_accelerations_no_icp_norm_anchor' in predictions")
                a_no_icp_norm_anchor = predictions['obj_accelerations_no_icp_norm_anchor']  # (B,T-2,Nobj,Nq,3)
                Bsz_acc, Tm2, Nobj_acc, Nv_acc, _ = a_icp_norm.shape

                # Gather GT accel and predicted accel_after_icp at anchor points
                # Reuse anchor_idx_b from position loss section above
                idx_exp_acc = anchor_idx_b.unsqueeze(1).unsqueeze(-1).expand(-1, Tm2, -1, -1, 3)  # (B,T-2,Nobj,Nq,3)
                gt_accel_anchor = torch.gather(gt_accel_norm, dim=3, index=idx_exp_acc)  # (B,T-2,Nobj,Nq,3)
                a_icp_anchor = torch.gather(a_icp_norm, dim=3, index=idx_exp_acc)  # (B,T-2,Nobj,Nq,3)

                # Step_size² normalization
                if_time_step_scale = self.config['training']['loss_weights'].get('if_time_step_scale', True)
                if_debug = self.config['training']['loss_weights'].get('if_debug', False)

                # Compute acceleration errors at anchors
                error_icp = a_icp_anchor - gt_accel_anchor  # (B,T-2,Nobj,Nq,3)
                error_no_icp = a_no_icp_norm_anchor - gt_accel_anchor  # (B,T-2,Nobj,Nq,3)

                if if_time_step_scale:
                    if 'skip_step_size' not in batch:
                        raise KeyError(
                            "Missing 'skip_step_size' in batch!\n"
                            "Multi-timescale training with if_time_step_scale=true requires step_size.\n"
                            "Please ensure dataloader provides 'skip_step_size' field."
                        )
                    step_sizes = batch['skip_step_size'].float()  # (B,)
                    step_size_scale_sq = step_sizes ** 2  # (B,)
                    scale_acc = step_size_scale_sq.view(Bsz_acc, 1, 1, 1, 1)  # (B,1,1,1,1)
                    error_icp_normalized = error_icp / scale_acc  # (B,T-2,Nobj,Nq,3)
                    error_no_icp_normalized = error_no_icp / scale_acc  # (B,T-2,Nobj,Nq,3)
                else:
                    error_icp_normalized = error_icp
                    error_no_icp_normalized = error_no_icp

                # Anchor-only L1 losses
                a_ste_l1 = F.smooth_l1_loss(error_icp_normalized, torch.zeros_like(error_icp_normalized), reduction='none').sum(dim=-1)  # (B,T-2,Nobj,Nq)
                a_no_icp_l1 = F.smooth_l1_loss(error_no_icp_normalized, torch.zeros_like(error_no_icp_normalized), reduction='none').sum(dim=-1)  # (B,T-2,Nobj,Nq)

                # Anchor mask: obj_mask only (all anchors valid for valid objects); the same anchors as the position
                # losses (same objects, T-2 steps), so the same divisor
                anchor_full_mask = obj_mask.expand(-1, Tm2, -1, Nq_loss).to(a_icp_norm.dtype)  # (B,T-2,Nobj,Nq)
                if anchor_full_mask.shape != anchor_mask_pos.shape:
                    raise ValueError(f"acceleration and position anchor masks differ: {tuple(anchor_full_mask.shape)} vs "
                                     f"{tuple(anchor_mask_pos.shape)}")
                accel_loss_after_icp_ste = (a_ste_l1.float() * anchor_full_mask.float()).sum() / (_anchor_count + 1e-8)
                accel_loss_before_icp = (a_no_icp_l1.float() * anchor_full_mask.float()).sum() / (_anchor_count + 1e-8)

                # 3) Penetration sign supervision (optional) - ALL VERTICES
                pen_loss = torch.tensor(0.0, device=self.device)
                pen_acc = torch.tensor(0.0, device=self.device)
                penetration_enabled = self.config['model'].get('if_penetration', self.config['model'].get('if_directional_distance', False))
                do_pen = penetration_enabled and ('obj_pen_pred_vertices' in predictions)
                if do_pen:
                    try:
                        # Prepare GT signs for ALL vertices for steps 1..T0-2 (aligned with predictions time)
                        # For each step t (forward uses prev=t-1, curr=t), use current frame vertices to test inside others
                        with torch.no_grad():
                            import numpy as _np
                            import trimesh as _trimesh  # type: ignore
                            Bsz = batch['obj_positions'].shape[0]
                            Tm2 = predictions['obj_pen_pred_vertices'].shape[1]
                            Nobj_ = batch['obj_positions'].shape[2]
                            Nv = batch['obj_positions'].shape[3]
                            gt_sign_vertices = torch.ones((Bsz, Tm2, Nobj_, Nv, 1), dtype=torch.float32, device=self.device)
                            valid_obj = batch['valid_obj_mask']  # (B,Nobj)
                            for b in range(Bsz):
                                # dataset name per-sample is a string (from dataloader list)
                                if 'dataset_name' not in batch or not isinstance(batch['dataset_name'], (list, tuple)):
                                    raise RuntimeError("Batch missing per-sample 'dataset_name' list for penetration GT")
                                ds_name = batch['dataset_name'][b]
                                if not isinstance(ds_name, str) or not ds_name:
                                    raise RuntimeError(f"Invalid dataset_name for sample {b}: {ds_name}")
                                for tstep in range(Tm2):
                                    # Skip tstep=0 (t=1, placeholder, GT positions, no penetration)
                                    if tstep == 0:
                                        continue

                                    # For tstep>=1: use predicted mesh positions to construct GT penetration
                                    # tstep=1 -> t=2 -> uses predictions[0]; tstep=2 -> t=3 -> uses predictions[1]
                                    # Use STE blended positions (after Kabsch projection), which match
                                    # forward's curr_pos_bt in student forcing mode:
                                    #   obj_positions_denorm = next_pos_bt (before Kabsch)
                                    #   obj_positions_ste = blended_next (after Kabsch, STE)
                                    V_all = predictions['obj_positions_ste'][b, tstep-1]  # (Nobj,Nv,3), after transform
                                    # Vertex mask for the corresponding frame: tstep=1 -> frame2, tstep=2 -> frame3
                                    vmask_all = batch['valid_vtx_mask'][b, tstep+1]  # (Nobj,Nv) - mask for frame tstep+1
                                    for i in range(Nobj_):
                                        if not valid_obj[b, i]:
                                            continue
                                        Vi = V_all[i][vmask_all[i]]  # (Nvi,3)
                                        if Vi.shape[0] == 0:
                                            continue
                                        inside_any = _np.zeros((Vi.shape[0],), dtype=_np.bool_)
                                        # Test against all other objects' meshes (triangle faces from dataset directory)
                                        for j in range(Nobj_):
                                            if j == i or not valid_obj[b, j]:
                                                continue
                                            Vj = V_all[j][vmask_all[j]].detach().cpu().numpy()
                                            # Load faces from batch (must exist; no fallback)
                                            try:
                                                # Use permuted faces from batch - this ensures faces match the permuted object order
                                                if 'faces_list' not in batch or batch['faces_list'] is None:
                                                    raise RuntimeError(
                                                        f"Missing 'faces_list' in batch! DataLoader must provide faces. "
                                                        f"Check that faces_list is included in sample dict."
                                                    )
                                                faces_np = batch['faces_list'][b][j].astype(_np.int64, copy=False)
                                                # Filter faces by current Nv mask (drop faces referencing padded verts)
                                                idx_valid = _np.where(vmask_all[j].detach().cpu().numpy())[0]
                                                if idx_valid.size == 0:
                                                    continue
                                                # Build mapping original->compact indices
                                                rev = -_np.ones((int(vmask_all[j].shape[0]),), dtype=_np.int64)
                                                rev[idx_valid] = _np.arange(idx_valid.size, dtype=_np.int64)
                                                faces_compact = rev[faces_np]
                                                faces_compact = faces_compact[(faces_compact >= 0).all(axis=1)]
                                                if faces_compact.size == 0 or faces_compact.shape[0] == 0:
                                                    continue
                                                mesh_j = _trimesh.Trimesh(vertices=Vj, faces=faces_compact, process=False)
                                                inside_j = mesh_j.contains(Vi.detach().cpu().numpy())
                                                inside_any |= inside_j
                                            except Exception as __e:
                                                logger.warning(f"Pen GT faces/contains failed (ds={ds_name}, obj={j}): {__e}")
                                        # Map vertex-level GT to full Nv space (ALL vertices)
                                        # inside_any is compact (Vi.shape[0]); need to expand to full Nv space
                                        idx_valid_i_val = _np.where(vmask_all[i].detach().cpu().numpy())[0]
                                        Nv_full = int(vmask_all[i].shape[0])
                                        inside_full = _np.zeros((Nv_full,), dtype=_np.bool_)
                                        if idx_valid_i_val.size > 0 and inside_any.shape[0] > 0:
                                            inside_full[idx_valid_i_val] = inside_any
                                        # Convert to sign for ALL vertices
                                        gt_sign = _np.where(inside_full, -1.0, 1.0).astype(_np.float32)
                                        gt_sign_vertices[b, tstep, i, :, 0] = torch.from_numpy(gt_sign).to(self.device)

                        # Pred scores and loss/acc (ALL VERTICES)
                        pen_pred = predictions['obj_pen_pred_vertices']  # (B,T-2,Nobj,Nv,1) in [-1,1]
                        pen_l2 = (pen_pred - gt_sign_vertices).pow(2).squeeze(-1)  # (B,T-2,Nobj,Nv)

                        # Vertex-level mask for loss aggregation
                        vtx_mask_for_loss = batch['valid_vtx_mask'][:, 1:Tm2+1, :, :]  # (B,T-2,Nobj,Nv)

                        # Time-level mask: Skip t=1 (tstep=0) where no penetration prediction was made
                        # Only supervise t>=2 (tstep>=1) where actual prediction occurred
                        time_mask = torch.ones((Tm2,), dtype=torch.float32, device=self.device)
                        time_mask[0] = 0.0  # Mask out t=1 (first step, GT positions, placeholder prediction)
                        time_mask = time_mask.view(1, Tm2, 1, 1)  # (1, T-2, 1, 1)

                        # Combined mask: vertex validity AND time validity
                        vertex_full_mask = vtx_mask_for_loss.to(pen_l2.dtype) * time_mask  # (B,T-2,Nobj,Nv)
                        pen_loss = (pen_l2.float() * vertex_full_mask.float()).sum() / (vertex_full_mask.float().sum() + 1e-8)

                        # Accuracy: sign match
                        pred_sign = torch.sign(pen_pred)
                        acc_bool = (pred_sign == torch.sign(gt_sign_vertices)).squeeze(-1)
                        pen_acc = (acc_bool * vertex_full_mask.bool()).sum().float() / (vertex_full_mask.bool().sum().float() + 1e-8)
                    except Exception as _e:
                        logger.warning(f"Penetration GT computation failed: {_e}")
            else:
                raise ValueError("Expected multi-object predictions with 'obj_positions_denorm'.")
                

            ##############################
            # Visualization of predictions vs ground truth
            # Same frequency as video saving
            if self.config['training']['checkpoint'].get('save_video', False):
                video_every = self.config['training']['checkpoint'].get('video_every_n_epochs', 5)
                if self.epoch % video_every == 0 and num_batches == 1:  # Only visualize first batch
                    self._visualize_predictions_vs_gt(
                        predictions, batch, batch_idx=num_batches-1
                    )
            
            # Compute edge length loss to preserve rigid body structure
            edge_loss = 0.0
            if self.loss_weights.get('edge_length', 0.0) > 0:
                if 'obj_positions_denorm' in predictions:
                    # Proxy edge-length loss, computed per object relative to t=0
                    init_obj = batch['obj_positions'][:, 0]  # (B,Nobj,Nv,3)
                    # Sum edge loss over objects using mask
                    vmask0 = batch['valid_vtx_mask'][:, 0]  # (B,Nobj,Nv)
                    obj_mask_b = batch['valid_obj_mask'].unsqueeze(-1)  # (B,Nobj,1)
                    mask0 = (vmask0 & obj_mask_b)
                    # Flatten objects and apply edge length loss per object then average over valid
                    pred_den = predictions['obj_positions_denorm']  # (B,T-2,Nobj,Nv,3)
                    # Use t-0 anchor per object
                    # Compute mean over time first, then per-object edge len towards init
                    pred_mean = pred_den.mean(dim=1)  # (B,Nobj,Nv,3)
                    # Edge loss proxy: mean squared displacement from initial pose
                    ed = (pred_mean - init_obj).pow(2).sum(dim=-1)
                    edge_loss = (ed.float() * mask0.float()).sum() / (mask0.float().sum() + 1e-8)
            
            # Combine losses
            pos_before_icp_w = self.loss_weights.get('position_before_icp', 0.0)  # Before Kabsch
            pos_after_icp_w = self.loss_weights.get('position_after_icp', 0.0)  # After Kabsch
            accel_after_icp_w = self.loss_weights.get('acceleration_after_icp', 0.0)  # After Kabsch
            accel_before_icp_w = self.loss_weights.get('acceleration_before_icp', 0.0)  # Before Kabsch

            loss = (
                pos_before_icp_w * position_loss_before_icp
                + pos_after_icp_w * position_loss_after_icp
                + accel_after_icp_w * accel_loss_after_icp_ste
                + accel_before_icp_w * accel_loss_before_icp
            )

            # Add penetration loss with its own weight if enabled
            pen_w = self.loss_weights.get('penetration_sign', 0.0)
            if pen_w > 0 and 'obj_pen_pred_vertices' in predictions:
                loss = loss + pen_w * pen_loss

            if self.loss_weights.get('edge_length', 0.0) > 0:
                loss = loss + self.loss_weights['edge_length'] * edge_loss

            # Per-step-size loss weighting: weight = 1/step_size
            # s=1: x1.0, s=5: x0.2, s=10: x0.1 -> prioritizes fine-grained (s=1) accuracy
            loss_weight_per_ss = self.config['training']['loss_weights'].get('loss_weight_per_step_size', False)
            if loss_weight_per_ss and 'skip_step_size' in batch:
                ss_val = batch['skip_step_size'].float().mean().item()  # All samples in batch share same step_size
                ss_weight = 1.0 / max(ss_val, 1.0)
                loss = loss * ss_weight

            # Diagnostic: log a detailed breakdown before backward when the loss is suspiciously high
            loss_val_pre = loss.item()
            if loss_val_pre > 100 or torch.isnan(loss) or torch.isinf(loss):
                ss_info = batch['skip_step_size'].cpu().tolist() if 'skip_step_size' in batch else 'N/A'
                logger.error(f"[Loss Explosion] Loss={loss_val_pre:.4e} at batch {num_batches}, step_size={ss_info}")
                logger.error(f"  pos_before_icp={position_loss_before_icp.item():.6e} (w={pos_before_icp_w})")
                logger.error(f"  pos_after_icp ={position_loss_after_icp.item():.6e} (w={pos_after_icp_w})")
                logger.error(f"  acc_before_icp={accel_loss_before_icp.item():.6e} (w={accel_before_icp_w})")
                logger.error(f"  acc_after_icp ={accel_loss_after_icp_ste.item():.6e} (w={accel_after_icp_w})")
                # Log which component caused the explosion
                weighted = {
                    'pos_before': pos_before_icp_w * position_loss_before_icp.item(),
                    'pos_after': pos_after_icp_w * position_loss_after_icp.item(),
                    'acc_before': accel_before_icp_w * accel_loss_before_icp.item(),
                    'acc_after': accel_after_icp_w * accel_loss_after_icp_ste.item(),
                }
                culprit = max(weighted, key=weighted.get)
                logger.error(f"  Weighted: {weighted} -> culprit={culprit}")
                logger.error(f"  time_step_id range: {batch['time_step_id'].min().item()}-{batch['time_step_id'].max().item()}")

            # Non-finite loss: abort on every rank together before anything reaches the
            # parameters or the optimizer state (a rank raising alone would leave the others
            # blocked in the gradient all-reduce). No fallback: the [Loss Explosion] breakdown
            # above is the diagnostic.
            _bad = (~torch.isfinite(loss.detach())).to(torch.float32).reshape(1)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(_bad, op=dist.ReduceOp.MAX)
            if _bad.item() > 0:
                raise RuntimeError(
                    f"Non-finite loss ({loss_val_pre}) at epoch {self.epoch}, batch {num_batches}; "
                    f"aborting before the optimizer step")

            # Backward pass
            self.optimizer.zero_grad()
            loss = loss.float()
            scaler.scale(loss).backward()

            # Gradient clipping with AMP requires unscale first
            clip_val = self.config['training'].get('gradient_clip', 0.0)
            if clip_val and clip_val > 0:
                scaler.unscale_(self.optimizer)
                # Gradients are identical on all ranks after DDP's all-reduce, so a non-finite
                # norm raises on every rank together instead of being written into the weights.
                # (With an fp16 GradScaler, inf gradients are expected occasionally and the
                # scaler skips that step itself, so the check is left to the scaler there.)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    clip_val,
                    norm_type=self.config['training'].get('gradient_clip_norm_type', 2),
                    error_if_nonfinite=not scaler.is_enabled(),
                )
                # Log when grad_norm was clipped significantly
                if grad_norm > clip_val * 10:
                    logger.warning(f"[GRAD CLIP] grad_norm={grad_norm:.2f} >> clip_val={clip_val}, loss={loss_val_pre:.4e}")
            if _wu_iters > 0:
                _gstep = self.epoch * _iters_per_epoch + num_batches
                if _gstep < _wu_iters:
                    _f = (_gstep + 1) / _wu_iters
                    for _g, _b in zip(self.optimizer.param_groups, _epoch_base_lrs):
                        _g['lr'] = _b * _f
                    if _gstep in (0, _wu_iters // 2, _wu_iters - 1) and self.is_main:
                        logger.info("[Warmup-iter] step %d/%d: lr = %s" % (_gstep + 1, _wu_iters, ['%.2e' % _g['lr'] for _g in self.optimizer.param_groups]))
            scaler.step(self.optimizer)
            scaler.update()
            if self.ema is not None:
                self.ema.update()
            
            # Update progress bar (only on rank 0 when DDP)
            loss_value = loss.item()

            # Check for inf/nan and log warning
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"[Batch {num_batches}] Loss is inf/nan: {loss_value}")
                logger.warning(f"  Accel after ICP: {accel_loss_after_icp_ste.item()}, Accel before ICP: {accel_loss_before_icp.item()}")
                logger.warning(f"  Position before ICP: {position_loss_before_icp.item()}, Position after ICP: {position_loss_after_icp.item()}")
                if self.loss_weights.get('edge_length', 0.0) > 0:
                    logger.warning(f"  Edge loss: {edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss}")
                # Skip this batch for average calculation
            else:
                total_loss += loss_value
                num_batches += 1

            if self.is_main:
                pbar.set_postfix({'loss': loss_value})

            # Synchronize batch losses across all ranks for accurate logging
            if dist.is_available() and dist.is_initialized():
                # Prepare all loss components for sync
                edge_loss_val = edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss
                # Pack pen metrics too (fallback to zeros)
                pen_loss_val = pen_loss.item() if isinstance(pen_loss, torch.Tensor) else float(pen_loss)
                pen_acc_val = pen_acc.item() if isinstance(pen_acc, torch.Tensor) else float(pen_acc)
                pos_after_icp_val = position_loss_after_icp.item() if isinstance(position_loss_after_icp, torch.Tensor) else float(position_loss_after_icp)
                loss_components = torch.tensor([
                    loss.item(),
                    accel_loss_after_icp_ste.item(),
                    accel_loss_before_icp.item(),
                    position_loss_before_icp.item(),
                    pos_after_icp_val,
                    edge_loss_val,
                    pen_loss_val,
                    pen_acc_val,
                ], device=self.device)
                # Sum across all ranks (Gloo backend does not support AVG), then divide by world size
                dist.all_reduce(loss_components, op=dist.ReduceOp.SUM)
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    loss_components = loss_components / max(world_size, 1)
                loss_sync, accel_after_icp_sync, accel_before_icp_sync, pos_before_icp_sync, pos_after_icp_sync, edge_sync, pen_loss_sync, pen_acc_sync = loss_components.tolist()
            else:
                # Single GPU: use local values
                loss_sync = loss.item()
                accel_after_icp_sync = accel_loss_after_icp_ste.item()
                accel_before_icp_sync = accel_loss_before_icp.item()
                pos_before_icp_sync = position_loss_before_icp.item()
                pos_after_icp_sync = position_loss_after_icp.item() if isinstance(position_loss_after_icp, torch.Tensor) else float(position_loss_after_icp)
                edge_sync = edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss
                pen_loss_sync = pen_loss.item() if isinstance(pen_loss, torch.Tensor) else float(pen_loss)
                pen_acc_sync = pen_acc.item() if isinstance(pen_acc, torch.Tensor) else float(pen_acc)


            # Log to tensorboard (only rank 0 in DDP)
            if self.is_main:
                global_step = self.epoch * len(self.train_loader) + num_batches
                # Rigid alpha is fixed at 1.0
                self.writer.add_scalar('train/rigid_alpha', 1.0, global_step)
                self.writer.add_scalar('train/loss', loss_sync, global_step)
                self.writer.add_scalar('train/accel_loss_after_icp', accel_after_icp_sync, global_step)
                self.writer.add_scalar('train/accel_loss_before_icp', accel_before_icp_sync, global_step)
                self.writer.add_scalar('train/position_loss_before_icp', pos_before_icp_sync, global_step)
                self.writer.add_scalar('train/position_loss_after_icp', pos_after_icp_sync, global_step)
                if self.loss_weights.get('edge_length', 0.0) > 0:
                    self.writer.add_scalar('train/edge_loss', edge_sync, global_step)
                if pen_w > 0 and 'obj_pen_pred_vertices' in predictions:
                    self.writer.add_scalar('train/pen_sign_loss', pen_loss_sync, global_step)
                    self.writer.add_scalar('train/pen_sign_acc', pen_acc_sync, global_step)


            # Accumulate epoch sums (use synced values)
            comp_sum_pos_before += float(pos_before_icp_sync)
            comp_sum_pos_after += float(pos_after_icp_sync)
            comp_sum_acc_after += float(accel_after_icp_sync)
            comp_sum_acc_before += float(accel_before_icp_sync)
            comp_sum_edge += float(edge_sync)
            comp_sum_pen += float(pen_loss_sync)
            comp_sum_pen_acc += float(pen_acc_sync)

        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_pos_before = comp_sum_pos_before / num_batches
            avg_pos_after = comp_sum_pos_after / num_batches
            avg_acc_after = comp_sum_acc_after / num_batches
            avg_acc_before = comp_sum_acc_before / num_batches
            avg_edge = comp_sum_edge / num_batches
            avg_pen = comp_sum_pen / max(num_batches, 1)
            avg_pen_acc = comp_sum_pen_acc / max(num_batches, 1)

        else:
            logger.warning("No valid batches in training epoch (all losses were inf/nan)")
            avg_loss = float('inf')
            avg_pos_before = float('nan')
            avg_pos_after = float('nan')
            avg_acc_after = float('nan')
            avg_acc_before = float('nan')
            avg_edge = float('nan')
            avg_pen = float('nan')
            avg_pen_acc = float('nan')


        # Synchronize loss across all ranks in DDP mode
        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            avg_loss = (loss_tensor / max(world_size, 1)).item()

        self.last_train_loss = avg_loss  # Store for overfitting mode
        if _wu_iters > 0:
            # undo the per-iteration warm-up scaling before the epoch scheduler advances from these rates
            for _g, _b in zip(self.optimizer.param_groups, _epoch_base_lrs):
                _g['lr'] = _b
        return {
            'loss': float(avg_loss),
            'pos_before_icp': float(avg_pos_before),
            'pos_after_icp': float(avg_pos_after),
            'accel_after_icp': float(avg_acc_after),
            'accel_before_icp': float(avg_acc_before),
            'edge': float(avg_edge),
            'pen_sign_loss': float(avg_pen),
            'pen_sign_acc': float(avg_pen_acc),

        }
    


    def validate(self):
        """Validate the model."""
        # Skip validation if no validation data
        if self.val_loader is None or len(self.val_loader) == 0:
            logger.info("No validation data, using training loss as validation metric")
            return self.last_train_loss if hasattr(self, 'last_train_loss') else float('inf')

        try:
            return self._validate_impl()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' in str(e).lower():
                if self.inference_only:
                    logger.error(f"CUDA OOM during validation: {e}")
                    logger.warning("Inference-only mode: moving model to CPU and retrying validation")

                    # Move model to CPU
                    self.device = 'cpu'
                    self.model = self.model.to('cpu')
                    torch.cuda.empty_cache()
                    logger.info("Model moved to CPU (execution will be slower)")

                    # Retry validation on CPU
                    return self._validate_impl()
                else:
                    logger.error("CUDA OOM during training validation - stopping training")
                    logger.error("Reduce batch size or model size, or use a GPU with more memory")
                    raise
            else:
                raise

    def _validate_impl(self):
        """Internal validation implementation (can be called with GPU or CPU)."""
        self.model.eval()
        _model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        if not hasattr(_model_ref, '_pe_type'):
            raise RuntimeError("Model missing _pe_type attribute; cannot validate without PE type.")
        total_loss = 0
        total_pos_loss = 0
        total_accel_loss = 0
        total_edge_loss = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', disable=not self.is_main)
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass (Flash Attention handles dtype internally in decoder.py)
                predictions = self.model(batch, teacher_forcing_ratio=0.0)  # No teacher forcing during validation
                
                # Compute losses (multi-object path)
                if 'obj_positions_denorm' in predictions and 'obj_positions' in batch:
                    # Sequence length from multi-object tensor
                    seq_len = batch['obj_positions'].shape[1]
                    if seq_len <= 2:
                        logger.warning("Single/short sequence in validation - skipping")
                        continue
                    # Ground truth positions for steps 1..T-2
                    gt_obj_pos = batch['obj_positions'][:, 2:seq_len]  # (B,T-2,Nobj,Nv,3)
                    pred_obj_pos = predictions['obj_positions_denorm']    # (B,T-2,Nobj,Nv,3)
                    # Ensure vertex dimension matches between pred and gt (defensive alignment)
                    Nv_pred = pred_obj_pos.shape[3]
                    Nv_gt = gt_obj_pos.shape[3]
                    if Nv_pred != Nv_gt:
                        Nv_common = min(Nv_pred, Nv_gt)
                        pred_obj_pos = pred_obj_pos[:, :, :, :Nv_common, :]
                        gt_obj_pos = gt_obj_pos[:, :, :, :Nv_common, :]
                    # Masks at those steps (align vertex dim accordingly)
                    vmask = batch['valid_vtx_mask'][:, 1:seq_len-1]
                    if vmask.shape[3] != gt_obj_pos.shape[3]:
                        vmask = vmask[:, :, :, :gt_obj_pos.shape[3]]
                    obj_mask = batch['valid_obj_mask'].unsqueeze(1).unsqueeze(-1)  # (B,1,Nobj,1)
                    full_mask = vmask & obj_mask  # (B,T-2,Nobj,Nv)
                    # Position loss (denorm space)
                    pos_err = (pred_obj_pos - gt_obj_pos).pow(2).sum(dim=-1)  # (B,T-2,Nobj,Nv)
                    pos_loss_before_icp = (pos_err * full_mask).sum() / (full_mask.sum() + 1e-8)

                    # Acceleration losses (must match training computation!)
                    _model_ref = self.model.module if isinstance(self.model, DDP) else self.model
                    accel_loss_after_icp_ste = 0.0
                    accel_loss_before_icp = 0.0

                    if 'obj_accelerations_norm' in predictions:
                        a_ste = predictions['obj_accelerations_norm']
                        gt_prev = batch['obj_positions'][:, 0:seq_len-2]
                        gt_curr = batch['obj_positions'][:, 1:seq_len-1]
                        gt_next = batch['obj_positions'][:, 2:seq_len]
                        gt_accel = gt_next - 2 * gt_curr + gt_prev

                        # Multi-timescale training: optionally normalize acceleration error by step_size²
                        # (must match the training computation for consistent validation)
                        accel_err_ste = a_ste - gt_accel  # (B,T-2,Nobj,Nv,3)

                        # Apply step_size² normalization if enabled (match training)
                        if_time_step_scale = self.config['training']['loss_weights'].get('if_time_step_scale', True)
                        if if_time_step_scale:
                            if 'skip_step_size' not in batch:
                                raise KeyError(
                                    "Missing 'skip_step_size' in validation batch!\n"
                                    "Multi-timescale training with if_time_step_scale=true requires step_size."
                                )
                            step_sizes = batch['skip_step_size'].float()  # (B,)
                            Bsz_val = step_sizes.shape[0]
                            scale_full = (step_sizes ** 2).view(Bsz_val, 1, 1, 1, 1)  # (B,1,1,1,1)
                            accel_err_ste = accel_err_ste / scale_full  # Normalize by step_size²
                        # else: no normalization, use raw error

                        accel_err_ste_sq = accel_err_ste.pow(2).sum(dim=-1)  # (B,T-2,Nobj,Nv)
                        accel_loss_after_icp_ste = (accel_err_ste_sq * full_mask).sum() / (full_mask.sum() + 1e-8)

                    if 'obj_accelerations_no_icp_norm_anchor' in predictions and 'anchor_idx' in batch:
                        # Predicted anchor accelerations (raw): (B,T-2,Nobj,Nq,3)
                        a_no_icp_anchor = predictions['obj_accelerations_no_icp_norm_anchor']
                        Tm2 = a_no_icp_anchor.shape[1]
                        # Ground truth accelerations (full Nv)
                        gt_prev = batch['obj_positions'][:, 0:seq_len-2]
                        gt_curr = batch['obj_positions'][:, 1:seq_len-1]
                        gt_next = batch['obj_positions'][:, 2:seq_len]
                        gt_accel = gt_next - 2 * gt_curr + gt_prev  # (B,T-2,Nobj,Nv,3)
                        # Downsample GT accel to anchors for fair comparison
                        idx_exp = batch['anchor_idx'].unsqueeze(1).unsqueeze(-1).expand(-1, Tm2, -1, -1, 3)  # (B,T-2,Nobj,Nq,3)
                        gt_accel_anchor = torch.gather(gt_accel, dim=3, index=idx_exp)  # (B,T-2,Nobj,Nq,3)

                        # Multi-timescale training: optionally normalize acceleration error by step_size²
                        # (must match the training computation for consistent validation)
                        accel_err_no_icp = a_no_icp_anchor - gt_accel_anchor  # (B,T-2,Nobj,Nq,3)

                        # Apply step_size² normalization if enabled (match training)
                        if_time_step_scale = self.config['training']['loss_weights'].get('if_time_step_scale', True)
                        if if_time_step_scale:
                            if 'skip_step_size' not in batch:
                                raise KeyError(
                                    "Missing 'skip_step_size' in validation batch!\n"
                                    "Multi-timescale training with if_time_step_scale=true requires step_size."
                                )
                            step_sizes = batch['skip_step_size'].float()  # (B,)
                            Bsz_val = step_sizes.shape[0]
                            scale_anchor = (step_sizes ** 2).view(Bsz_val, 1, 1, 1, 1)  # (B,1,1,1,1)
                            accel_err_no_icp = accel_err_no_icp / scale_anchor  # Normalize by step_size²
                        # else: no normalization, use raw error

                        accel_err_no_icp_sq = accel_err_no_icp.pow(2).sum(dim=-1)  # (B,T-2,Nobj,Nq)
                        # Anchor mask derived from object validity only (anchors are prevalidated)
                        anchor_full_mask = batch['valid_obj_mask'].unsqueeze(1).unsqueeze(-1).expand(-1, Tm2, -1, a_no_icp_anchor.shape[3])
                        accel_loss_before_icp = (accel_err_no_icp_sq * anchor_full_mask).sum() / (anchor_full_mask.sum() + 1e-8)
                    # Edge length proxy (optional)
                    edge_loss = 0.0
                    if self.loss_weights.get('edge_length', 0.0) > 0:
                        init_obj = batch['obj_positions'][:, 0]  # (B,Nobj,Nv,3)
                        pred_mean = pred_obj_pos.mean(dim=1)      # (B,Nobj,Nv,3)
                        vmask0 = batch['valid_vtx_mask'][:, 0]    # (B,Nobj,Nv)
                        obj_mask_b = batch['valid_obj_mask'].unsqueeze(-1)  # (B,Nobj,1)
                        mask0 = vmask0 & obj_mask_b
                        ed = (pred_mean - init_obj).pow(2).sum(dim=-1)
                        edge_loss = (ed * mask0).sum() / (mask0.sum() + 1e-8)
                else:
                    logger.warning("Validation batch is missing multi-object tensors; skipping batch.")
                    continue
                    
                # Total loss (must match training weights!)
                loss = self.loss_weights['position_before_icp'] * pos_loss_before_icp
                if self.loss_weights.get('acceleration_after_icp', 0.0) > 0 and accel_loss_after_icp_ste > 0:
                    loss = loss + self.loss_weights['acceleration_after_icp'] * accel_loss_after_icp_ste
                if self.loss_weights.get('acceleration_before_icp', 0.0) > 0 and accel_loss_before_icp > 0:
                    loss = loss + self.loss_weights['acceleration_before_icp'] * accel_loss_before_icp
                if self.loss_weights.get('edge_length', 0.0) > 0:
                    loss = loss + self.loss_weights['edge_length'] * edge_loss

                total_loss += loss.item()
                total_pos_loss += pos_loss_before_icp.item() if isinstance(pos_loss_before_icp, torch.Tensor) else pos_loss_before_icp
                total_accel_loss += accel_loss_after_icp_ste.item() if isinstance(accel_loss_after_icp_ste, torch.Tensor) else accel_loss_after_icp_ste
                total_edge_loss += edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss
                num_batches += 1
                
                pbar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_pos_loss = total_pos_loss / num_batches
        avg_accel_loss = total_accel_loss / num_batches
        avg_edge_loss = total_edge_loss / num_batches

        # Synchronize validation metrics across all ranks in DDP mode
        if dist.is_available() and dist.is_initialized():
            metrics = torch.tensor([avg_loss, avg_pos_loss, avg_accel_loss, avg_edge_loss], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            metrics = metrics / max(world_size, 1)
            avg_loss, avg_pos_loss, avg_accel_loss, avg_edge_loss = metrics.tolist()

        # Log to tensorboard (only rank 0 in DDP)
        if self.is_main:
            # Log STE rigid alpha (always enabled)
            self.writer.add_scalar('val/rigid_alpha', 1.0, self.epoch)
            self.writer.add_scalar('val/loss', avg_loss, self.epoch)
            self.writer.add_scalar('val/position_loss_before_icp', avg_pos_loss, self.epoch)
            self.writer.add_scalar('val/accel_loss_after_icp_ste', avg_accel_loss, self.epoch)
            if self.loss_weights.get('edge_length', 0.0) > 0:
                self.writer.add_scalar('val/edge_loss', avg_edge_loss, self.epoch)

        return avg_loss

    def check_forward_rollout_consistency(self):
        """
        One-time check to verify forward and rollout produce consistent results.
        Called once after epoch 1 to ensure training and inference paths are aligned.
        """
        logger.info("=" * 80)
        logger.info("Running forward-rollout consistency check...")
        logger.info("=" * 80)

        # Get underlying model (unwrap DDP if needed)
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        model_ref.eval()

        # Check if validation loader exists
        if self.val_loader is None:
            logger.info("Validation disabled - skipping consistency check")
            return

        # Get a batch from validation set
        val_iter = iter(self.val_loader)
        try:
            batch = next(val_iter)
        except StopIteration:
            logger.warning("No validation data available for consistency check")
            return

        # Move to device
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Flash Attention handles dtype internally in decoder.py - rest of model runs in FP32
        with torch.no_grad():
            # Set fixed seed for reproducibility
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            # Run forward with student forcing to align with rollout (teacher_forcing_ratio=0.0)
            forward_output = model_ref(batch, teacher_forcing_ratio=0.0)

            # Prepare batch for rollout (needs only first 2 frames)
            B, T, Nobj, Nv, _ = batch['obj_positions'].shape
            logger.info(f"Consistency check: T={T}, predicting {T-2} step(s)")

            # Reset seed before rollout for exact match
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            rollout_batch = {
                'obj_positions': batch['obj_positions'][:, :2],  # First 2 frames for warmup
                'valid_obj_mask': batch['valid_obj_mask'],
                'valid_vtx_mask': batch['valid_vtx_mask'][:, :2],  # First 2 frames
                'ref_obj_positions': batch['ref_obj_positions'],
                'anchor_idx': batch['anchor_idx'],
            }
            # Per-sample inputs the rollout reads besides positions and masks (physical parameters, step size, FPS
            # ranks), and dataset_name (penetration sign prediction)
            for _k in ('phys_params', 'skip_step_size', 'vtx_rank', 'dataset_name'):
                if _k in batch:
                    rollout_batch[_k] = batch[_k]

            # Run rollout (inference path) - predict same number of steps as forward
            num_steps_to_predict = T  # Total frames (includes 2 warmup)
            rollout_output = model_ref.rollout(rollout_batch, num_steps=num_steps_to_predict)

            # Extract predictions for comparison
            # Compare STE-blended positions (used by rollout autoregression) to ensure path alignment
            # Forward produces predictions for frames [2, T-1] -> shape (B, T-2, Nobj, Nv, 3)
            forward_pred = forward_output.get('obj_positions_ste', forward_output['obj_positions_denorm'])

            # Rollout produces full sequence including warmup -> shape (B, T, Nobj, Nv, 3)
            # Extract predicted frames [2, T-1] to match forward
            rollout_pred = rollout_output['obj_positions'][:, 2:T]  # (B, T-2, Nobj, Nv, 3)

            logger.info("=" * 80)
            logger.info("Step-by-step comparison:")
            logger.info("=" * 80)

            # Compare input frames (should be identical)
            input_frame0_fw = batch['obj_positions'][:, 0]
            input_frame1_fw = batch['obj_positions'][:, 1]
            input_frame0_ro = rollout_batch['obj_positions'][:, 0]
            input_frame1_ro = rollout_batch['obj_positions'][:, 1]

            diff_frame0 = torch.abs(input_frame0_fw - input_frame0_ro).max().item()
            diff_frame1 = torch.abs(input_frame1_fw - input_frame1_ro).max().item()
            logger.info(f"Input frame 0 difference: {diff_frame0:.6e}")
            logger.info(f"Input frame 1 difference: {diff_frame1:.6e}")

            # Compare ref_pose
            ref_diff = torch.abs(batch['ref_obj_positions'] - rollout_batch['ref_obj_positions']).max().item()
            logger.info(f"Reference pose difference: {ref_diff:.6e}")

            # Compare outputs for each predicted frame
            num_pred_frames = forward_pred.shape[1]
            logger.info(f"Number of predicted frames: {num_pred_frames}")

            for frame_idx in range(num_pred_frames):
                fw_frame = forward_pred[:, frame_idx]  # (B, Nobj, Nv, 3)
                ro_frame = rollout_pred[:, frame_idx]  # (B, Nobj, Nv, 3)
                frame_diff = torch.abs(fw_frame - ro_frame)

                # Get mask for this frame (frame_idx corresponds to frame 2+frame_idx in original batch)
                actual_frame_idx = 2 + frame_idx
                if actual_frame_idx < batch['valid_vtx_mask'].shape[1]:
                    vmask = batch['valid_vtx_mask'][:, actual_frame_idx]  # (B, Nobj, Nv)
                    obj_mask = batch['valid_obj_mask'].unsqueeze(-1)  # (B, Nobj, 1)
                    full_mask = vmask & obj_mask
                    masked_diff = frame_diff * full_mask.unsqueeze(-1)
                    masked_max = masked_diff.max().item()
                    masked_mean = masked_diff.sum().item() / (full_mask.sum().item() * 3 + 1e-8)
                else:
                    masked_max = frame_diff.max().item()
                    masked_mean = frame_diff.mean().item()

                logger.info(f"  Frame {actual_frame_idx}: max_diff={frame_diff.max().item():.6e}, "
                           f"masked_max={masked_max:.6e}, masked_mean={masked_mean:.6e}")

                # Print per-object statistics for first predicted frame
                if frame_idx == 0:
                    logger.info(f"    Per-object breakdown for frame {actual_frame_idx}:")
                    for obj_idx in range(min(Nobj, 5)):  # Only first 5 objects
                        obj_diff = frame_diff[:, obj_idx].max().item()
                        logger.info(f"      Object {obj_idx}: max_diff={obj_diff:.6e}")

            logger.info("=" * 80)

            # Compute difference
            diff = torch.abs(forward_pred - rollout_pred)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Get masks for valid comparison
            obj_mask = batch['valid_obj_mask'].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (B,1,Nobj,1,1)
            vtx_mask = batch['valid_vtx_mask'][:, 2:T].unsqueeze(-1)  # (B,T-2,Nobj,Nv,1)
            full_mask = (obj_mask & vtx_mask).float()

            # Masked statistics
            masked_diff = diff * full_mask
            num_valid = full_mask.sum().item()
            if num_valid > 0:
                masked_mean_diff = masked_diff.sum().item() / num_valid
                masked_max_diff = masked_diff.max().item()
            else:
                masked_mean_diff = 0.0
                masked_max_diff = 0.0

            # Log results
            logger.info(f"Consistency Check Results:")
            logger.info(f"  Forward output shape: {forward_pred.shape}")
            logger.info(f"  Rollout output shape: {rollout_pred.shape}")
            logger.info(f"  Max absolute difference: {max_diff:.6e}")
            logger.info(f"  Mean absolute difference: {mean_diff:.6e}")
            logger.info(f"  Masked max difference: {masked_max_diff:.6e}")
            logger.info(f"  Masked mean difference: {masked_mean_diff:.6e}")
            logger.info(f"  Number of valid elements: {int(num_valid)}")

            # Tolerance check (allow small numerical errors from GPU operations and accumulation)
            # Relaxed from 1e-4 to 1e-3 to account for autoregressive error accumulation
            tolerance = 1e-3
            if masked_max_diff > tolerance:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"Consistency check FAILED.\n"
                    f"Forward and rollout outputs differ by {masked_max_diff:.6e} (tolerance: {tolerance:.6e})\n"
                    f"This indicates train and inference paths are not aligned.\n"
                    f"{'='*80}\n"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.info("=" * 80)
                logger.info("Consistency check passed - forward and rollout are aligned")
                logger.info("=" * 80)

        # Restore training mode
        model_ref.train()

    def generate_visualization_video(self):
        """Generate visualization videos for multiple datasets (configurable per split)."""
        logger.info(f"generate_visualization_video called for epoch {self.epoch}")
        # Only rank 0 should generate videos in DDP
        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_rank() != 0:
                    return
            except Exception:
                pass

        if not self.config['training']['checkpoint'].get('save_video', False):
            logger.info("Video generation disabled in config")
            return

        video_every = self.config['training']['checkpoint'].get('video_every_n_epochs', 10)
        logger.info(f"video_every_n_epochs = {video_every}, epoch = {self.epoch}, should generate = {self.epoch % video_every == 0}")

        if self.epoch % video_every != 0:
            return

        logger.info(f"Generating visualization videos for epoch {self.epoch}")
        _video_failures = []

        # Get number of videos per split from config (default to 3 for backwards compatibility)
        num_videos_cfg = self.config['training']['checkpoint'].get('num_videos_per_split', {})
        num_train_videos = num_videos_cfg.get('train', 3)
        num_val_videos = num_videos_cfg.get('val', 0)
        num_test_videos = num_videos_cfg.get('test', 0)

        logger.info(f"Video config: {num_train_videos} train videos, {num_val_videos} val videos, {num_test_videos} test videos")

        # Load dataset split info to know which datasets are train/val/test
        split_file = self.config.get('data', {}).get('split_file', 'dataset_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)

            # Read all dataset lists
            train_datasets_all = split_info.get('train_datasets', [])
            val_datasets_all = split_info.get('val_datasets', [])
            test_datasets_all = split_info.get('test_datasets', [])

            # Select datasets for visualization based on config
            train_datasets = []
            val_datasets = []
            test_datasets = []

            # Train videos
            if num_train_videos > 0 and len(train_datasets_all) > 0:
                train_datasets = train_datasets_all[:num_train_videos]
                logger.info(f"Train videos: {len(train_datasets)} datasets selected")

            # Val videos (only if validation is enabled)
            if num_val_videos > 0 and len(val_datasets_all) > 0:
                if self.config['training'].get('enable_validation', False):
                    val_datasets = val_datasets_all[:num_val_videos]
                    logger.info(f"Val videos: {len(val_datasets)} datasets selected (validation enabled)")
                else:
                    logger.info(f"Val videos: skipped (validation disabled, enable_validation=false)")

            # Test videos (always available if datasets exist)
            if num_test_videos > 0 and len(test_datasets_all) > 0:
                test_datasets = test_datasets_all[:num_test_videos]
                logger.info(f"Test videos: {len(test_datasets)} datasets selected")

            # Create list of (dataset_name, split_type) tuples
            datasets_to_visualize = (
                [(d, 'train') for d in train_datasets] +
                [(d, 'val') for d in val_datasets] +
                [(d, 'test') for d in test_datasets]
            )
            logger.info(f"Visualizing datasets: {[f'{d[0]}({d[1]})' for d in datasets_to_visualize]}")
        else:
            raise RuntimeError(f"Split file not found: {split_file}")
        
        # Multi-timescale video generation: loop over eval_step_sizes
        # (falls back to data.skip_step_size if not set)
        eval_step_sizes = self.config['data'].get('eval_step_sizes', [self.config['data'].get('skip_step_size', 1)])

        logger.info(f"Video generation: evaluating at step_sizes {eval_step_sizes}")

        # Optional debug logging (data.random_step_size.if_debug)
        if_debug = self.config.get('data', {}).get('random_step_size', {}).get('if_debug', False)
        if if_debug:
            logger.info(f"[DEBUG Video Gen] eval_step_sizes={eval_step_sizes}, total datasets={len(datasets_to_visualize)}")

        # Put model in eval mode
        self.model.eval()

        # Outer loop: iterate over each step_size
        for step_size in eval_step_sizes:
            logger.info(f"\n{'='*80}")
            logger.info(f"[Video Generation] step_size = {step_size}")
            logger.info(f"{'='*80}")

            # Optional debug logging
            if if_debug:
                logger.info(f"[DEBUG Video Gen] Starting step_size={step_size} iteration ({eval_step_sizes.index(step_size)+1}/{len(eval_step_sizes)})")

            # Number of frames to generate for this step_size
            base_num_frames = self.config['training']['checkpoint'].get('video_num_frames', 50)
            num_frames = base_num_frames // step_size + 1

            # Inner loop: generate video for each dataset at this step_size
            for dataset_name, split_type in datasets_to_visualize:
                try:
                    logger.info(f"Generating video for {dataset_name} ({split_type}) with step_size={step_size}")

                    # Extract dataset number for video naming (e.g., "dataset_aug_100" -> "100")
                    dataset_num = dataset_name.split('_')[-1]

                    # One video per step size; visualization.if_overlay in the
                    # config adds a transparent ground-truth overlay when true.
                    video_path = self.log_dir / f'inference_epoch_{self.epoch:04d}_dataset_{dataset_num}_{split_type}_size_{step_size}.mp4'
                    inference_fps = self._create_video_multiobj_pred(dataset_name, video_path, split=split_type, step_size=step_size)
                    if video_path.exists():
                        file_size_kb = video_path.stat().st_size / 1024
                        logger.info(f"Video saved to {video_path} ({file_size_kb:.1f} KB, Inference: {inference_fps:.2f} FPS)")
                        self.writer.add_text('visualization', f'[Video](file://{video_path})', self.epoch)

                except Exception as e:
                    logger.error(f"Visualization error for {dataset_name} (step_size={step_size}): {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    _video_failures.append(f"{dataset_name} s={step_size}: {e}")
                    continue  # Continue with next dataset

        # Put model back in train mode after all videos
        self.model.train()
        logger.info(f"\nCompleted video generation for epoch {self.epoch} (all step_sizes)")
        if _video_failures and (self.config.get('inference', {}) or {}).get('mode', False):
            # a requested video that failed is an error of the run (scripts/render_videos.sh relies on the exit status);
            # during training the videos are diagnostics and the errors above are logged
            raise RuntimeError(f"{len(_video_failures)} video(s) failed: {_video_failures[:3]}")

    def _create_video_multiobj_gt(self, obj_positions_seq, faces_list, output_path, fps=100):
        """Render multi-object GT meshes per frame using provided faces (indices per object)."""
        try:
            os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
            os.environ.setdefault('POLYSCOPE_BACKEND', 'openGL3_egl')
            os.environ.setdefault('POLYSCOPE_ALLOW_HEADLESS', '1')
            import polyscope as ps
            import numpy as np
            import subprocess
        except ImportError:
            raise RuntimeError("Polyscope not available; install polyscope or disable save_video")

        frames_dir = self.log_dir / f'temp_frames_epoch_{self.epoch}_mobj'
        frames_dir.mkdir(exist_ok=True)
        try:
            if hasattr(ps, 'set_allow_headless'):
                ps.set_allow_headless(True)
            if hasattr(ps, 'set_backend'):
                ps.set_backend('openGL3_egl')
            ps.init('openGL3_egl')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Polyscope headless backend: {e}")
        # Env overrides for visualization
        try:
            _w = int(os.environ.get('WIDTH', '0'))
            _h = int(os.environ.get('HEIGHT', '0'))
        except Exception:
            _w, _h = 0, 0
        if _w > 0 and _h > 0 and hasattr(ps, 'set_window_size'):
            try:
                ps.set_window_size(_w, _h)
            except Exception:
                pass
        _fps_env = os.environ.get('FPS', '').strip()
        if _fps_env:
            try:
                fps = float(_fps_env)
            except Exception:
                pass
        try:
            _edge_w = float(os.environ.get('EDGE_WIDTH', 'nan'))
        except Exception:
            _edge_w = float('nan')
        ps.set_autocenter_structures(False)
        ps.set_ground_plane_mode("none")
        ps.set_ground_plane_height(0.0)
        ps.set_navigation_style('turntable')
        ps.set_up_dir('z_up')

        # Camera settings from config
        checkpoint_config = self.config['training']['checkpoint']
        camera_config = checkpoint_config.get('camera', {})
        camera_mode = camera_config.get('mode', 'auto')

        first = obj_positions_seq[0]
        all_pts = np.concatenate(first, axis=0)

        if camera_mode == 'fixed':
            # Use fixed camera from config
            center = np.array(camera_config.get('center', [0.0, 0.0, 0.5]), dtype=np.float64)
            eye = np.array(camera_config.get('eye', [3.0, -3.0, 3.0]), dtype=np.float64)
        else:
            # Auto camera: compute from scene bounds
            center = np.mean(all_pts, axis=0)
            bbox_min = np.min(all_pts, axis=0)
            bbox_max = np.max(all_pts, axis=0)
            radius = 0.5 * np.linalg.norm(bbox_max - bbox_min)
            auto_direction = np.array(camera_config.get('auto_direction', [2.0, -2.0, 2.0]), dtype=np.float64)
            auto_scale = camera_config.get('auto_scale', 3.0)
            eye = center + auto_direction * max(radius, 1e-3) * auto_scale

        # Ground plane settings from config
        ground_config = checkpoint_config.get('ground', {})
        ground_size = ground_config.get('size', 10.0)
        ground_color = ground_config.get('color', [0.8, 0.8, 0.8])
        ground_vertices = np.array([
            [-ground_size, -ground_size, 0.0],
            [ ground_size, -ground_size, 0.0],
            [ ground_size,  ground_size, 0.0],
            [-ground_size,  ground_size, 0.0]
        ])
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]])

        for t, objs in enumerate(obj_positions_seq):
            # Register ground first
            ps.register_surface_mesh("ground", ground_vertices, ground_faces, enabled=True)
            ps.get_surface_mesh("ground").set_color(ground_color)
            try:
                if _edge_w == _edge_w:
                    ps.get_surface_mesh("ground").set_edge_width(_edge_w)
            except Exception:
                pass
            for i, V in enumerate(objs):
                F = faces_list[i]
                ps.register_surface_mesh(f"obj_{i}", V, F, enabled=True, smooth_shade=True)
                try:
                    if _edge_w == _edge_w:
                        ps.get_surface_mesh(f"obj_{i}").set_edge_width(_edge_w)
                        ps.get_surface_mesh(f"obj_{i}").set_edge_color([0.0, 0.0, 0.0])
                except Exception:
                    pass
            ps.look_at(eye, center)  # ps.look_at(camera_location, target)
            ps.screenshot(str(frames_dir / f'frame_{t:05d}.png'))
            ps.remove_all_structures()
            if t % 10 == 0:
                logger.info(f"  [multiobj] Generated frame {t}/{len(obj_positions_seq)}")

        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-r', str(fps),
            '-i', str(frames_dir / 'frame_%05d.png'),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'mpeg4', '-q:v', '2', '-pix_fmt', 'yuv420p', '-y',
            str(output_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Clean up temporary frames
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        logger.info(f"  [multiobj] Cleaned up temp frames: {frames_dir}")

    def _render_video_filament(self, ds_dir, batch, pred_frames, gt_frames, vmask_all, output_path, fps, times):
        """Render a rollout with src/utils/rollout_render.py. Every object is drawn with its visual mesh, moved by the
        rigid motion fitted to its collision-mesh vertices (predicted: opaque; ground truth, when given: translucent).
        `times` (s) of the key frames: the video plays them in real time at `fps`, interpolating in between."""
        import json
        from src.utils import rollout_render
        mesh_dir = self.config['training']['checkpoint'].get('video_meshes', 'data_generation/objects')
        info = json.loads((Path(ds_dir) / 'physics_params.json').read_text())
        objects = [info[f'obj_{i}'] for i in range(sum(k.startswith('obj_') for k in info))]
        if 'permutation' in batch and batch['permutation'] is not None:
            perm = batch['permutation'][0]
            perm = perm.cpu().numpy() if isinstance(perm, torch.Tensor) else np.asarray(perm)
            objects = [objects[int(k)] for k in perm]
        if pred_frames.shape[1] != len(objects):
            raise RuntimeError(f'{ds_dir}: {pred_frames.shape[1]} predicted objects, physics_params.json lists '
                               f'{len(objects)}')
        shapes = [o['shape'] for o in objects]
        vm = None if vmask_all is None else np.asarray(vmask_all[0], bool)            # (Nobj, Nv)

        def poses(frames):
            out = []
            for i, o in enumerate(objects):
                ref = float(o['size']) * rollout_render.collision_vertices(mesh_dir, o['shape'])
                nv = len(ref)
                if vm is not None and not (vm[i, :nv].all() and not vm[i, nv:].any()):
                    raise RuntimeError(f"{ds_dir}: object {i} has {int(vm[i].sum())} valid vertices, its "
                                       f"{o['shape']} collision mesh {nv}")
                P, resid = rollout_render.fit_poses(ref, np.asarray(frames[:, i, :nv], np.float64))
                if resid > 1e-3:
                    raise RuntimeError(f"{ds_dir}: object {i} is not a rigid motion of its {o['shape']} collision "
                                       f"mesh at size {o['size']:.4f} (largest residual {resid:.2e} m)")
                out.append(P)
            return out

        visual = rollout_render.load_render_meshes(mesh_dir, shapes)
        self._filament_renderer().render(
            output_path, [visual[s] for s in shapes], [o['size'] for o in objects],
            [o.get('material') == 'metal' for o in objects], poses(pred_frames), fps,
            ghost=None if gt_frames is None else poses(np.asarray(gt_frames)), times=np.asarray(times, np.float64))
        logger.info(f"  [video] {output_path} ({len(pred_frames)} predicted frames, real time at {fps:g} fps)")

    def _create_video_multiobj_pred(self, dataset_name: str, output_path, fps=100, split='val', step_size: int = None, override_if_overlay: int = None):
        """Render predicted per-object vertices if batch/masks are available for first sample.
        This function will run one batch from the dataloader for the given dataset name and render
        the model predictions with simple point rendering per object (different colors).

        Args:
            dataset_name: Name of dataset to visualize
            output_path: Path to save video
            fps: Frames per second for video
            split: 'train', 'val', or 'test'
            step_size: Frame skip step size for multi-timescale (defaults to config skip_step_size)
            override_if_overlay: If provided (0 or 1), override config's if_overlay setting

        Returns:
            float: Inference FPS (frames predicted per second)
        """
        # Multi-Timescale: Determine skip_step_size early (needed for dataset creation)
        if step_size is None:
            skip_step_size = int(self.config['data'].get('skip_step_size', 1))
        else:
            skip_step_size = int(step_size)

        # Optional debug logging (data.random_step_size.if_debug)
        if_debug = self.config.get('data', {}).get('random_step_size', {}).get('if_debug', False)
        if if_debug:
            logger.info(f"[DEBUG Video Gen] Starting video generation for dataset='{dataset_name}', split='{split}', step_size={skip_step_size}")

        # training.checkpoint.video_renderer: 'filament' (default; src/utils/rollout_render.py, physically based, the objects'
        # visual meshes) or 'polyscope' (the earlier diagnostic view of the collision meshes)
        video_renderer = self.config['training']['checkpoint'].get('video_renderer', 'filament')
        if video_renderer not in ('filament', 'polyscope'):
            raise ValueError(f"training.checkpoint.video_renderer must be 'filament' or 'polyscope', got {video_renderer!r}")
        import numpy as np
        import random
        from pathlib import Path
        if video_renderer == 'polyscope':
            try:
                os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
                os.environ.setdefault('POLYSCOPE_BACKEND', 'openGL3_egl')
                os.environ.setdefault('POLYSCOPE_ALLOW_HEADLESS', '1')
                import polyscope as ps
            except ImportError:
                raise RuntimeError("Polyscope not available; install polyscope or disable save_video")

        # ============================================================
        # Create a separate dataset per step_size.
        # Training datasets sample a random step_size per sample (multi-timescale),
        # while video generation requires a fixed step_size to match the network
        # conditioning, so an independent dataset is created and cached per
        # (split, step_size) pair. Cache key: {step_size: dataset} per split.
        # ============================================================

        from src.data.multi_dataloader import MultiObjectDynamics

        # Select cache dictionary based on split
        if split == 'train':
            cache_dict = self._train_video_datasets
        elif split == 'val':
            if self.val_loader is None:
                raise RuntimeError(f"Cannot generate val video: validation is disabled (enable_validation=false)")
            cache_dict = self._val_video_datasets
        elif split == 'test':
            cache_dict = self._test_video_datasets
        else:
            raise ValueError(f"Unknown split: {split}")

        # Check cache, create dataset if not exists
        if skip_step_size not in cache_dict:
            logger.info(f"Creating {split} dataset for step_size={skip_step_size} (first time, will be cached)")

            # Disable random_step_size to get fixed step_size sequences
            augmentation_config = self.config['data'].get('augmentation', {}).copy()
            # Ensure no rotation/permutation augmentation for video generation (deterministic)
            if 'rotation_augmentation' in augmentation_config:
                augmentation_config['rotation_augmentation'] = augmentation_config['rotation_augmentation'].copy()
                augmentation_config['rotation_augmentation']['enable'] = False
            if 'anchor_resampling' in augmentation_config:
                augmentation_config['anchor_resampling'] = dict(augmentation_config['anchor_resampling'], enable=False)
            if 'object_permutation' in augmentation_config:
                augmentation_config['object_permutation'] = augmentation_config['object_permutation'].copy()
                augmentation_config['object_permutation']['enable'] = False

            cache_dict[skip_step_size] = MultiObjectDynamics(
                data_path=self.config['data']['dataset_path'],
                split_file=self.config['data']['split_file'],
                split=split,
                sequence_length=self.config['data']['sequence_length'],
                skip_step_size=skip_step_size,  # Fixed step_size (NOT random!)
                random_step_size={'enable': False},  # Disable random sampling (use fixed skip_step_size)
                max_vertices_per_object=self.config['data']['max_vertices_per_object'],
                max_objects_per_frame=self.config['data']['max_objects_per_frame'],
                anchor_k=self.config['data'].get('anchor_k', 32),
                skip_first_frames=self.config['data'].get('skip_first_frames', 0),
                augmentation=augmentation_config,
                anchor_fps_start=self.config['data'].get('anchor_fps_start', 'index0'),
                fps_order=self.config['data'].get('fps_order', False),
                object_scale_file=self.config['data'].get('object_scale_file'),
                if_phys_param=self.config.get('model', {}).get('if_phys_param', False)
            )
            logger.info(f"{split.capitalize()} dataset (step_size={skip_step_size}) created and cached: "
                       f"{len(cache_dict[skip_step_size].all_sequences)} sequences")
        else:
            logger.info(f"Reusing cached {split} dataset for step_size={skip_step_size} "
                       f"({len(cache_dict[skip_step_size].all_sequences)} sequences)")

        ds = cache_dict[skip_step_size]

        # Pick first sequence that belongs to dataset_name
        sample_idx = None
        for i, s in enumerate(ds.all_sequences):
            if s['dataset'] == dataset_name:
                sample_idx = i
                break
        if sample_idx is None:
            raise RuntimeError(f"No sequence found for dataset {dataset_name}")

        sample = ds[sample_idx]
        # Physical frame of the sequence's first frame (data.skip_first_frames shifts it): the rollout and the
        # ground-truth overlay start there
        if 'frame_ids' not in sample or sample['frame_ids'].numel() == 0:
            raise KeyError("video generation needs the sample's frame_ids (physical frame of the first frame)")
        video_start_frame = int(sample['frame_ids'][0])
        # Move tensors to device and handle special keys (dataset_name, faces_list)
        to_dev = lambda x: x.unsqueeze(0).to(self.device) if isinstance(x, torch.Tensor) else x
        batch = {}
        for k, v in sample.items():
            if k == 'dataset_name':
                # dataset_name must be a list (as in training collate_fn)
                batch[k] = [v] if isinstance(v, str) else v
            elif k == 'faces_list':
                # faces_list is a list of arrays per object, wrap it to simulate collate_fn behavior
                # collate_fn does: [sample0['faces_list'], sample1['faces_list'], ...]
                # We need: [sample['faces_list']] (batch size 1)
                batch[k] = [v]
            elif k == 'permutation':
                # permutation is a numpy array, wrap it to simulate collate_fn behavior
                batch[k] = [v]
            else:
                batch[k] = to_dev(v)

        # Decide rollout length & output FPS to match source duration
        ds_dir = Path(self.config['data']['dataset_path']) / dataset_name
        simparams_path = ds_dir / 'simulation_parameters.json'
        # Prefer explicit config override for original fps
        fps_orig = 240.0
        try:
            fps_cfg = self.config['training']['checkpoint'].get('original_video_fps', None)
            if fps_cfg is not None:
                fps_orig = float(fps_cfg)
            elif simparams_path.exists():
                with open(simparams_path, 'r') as f:
                    simp = json.load(f)
                fps_orig = float(simp.get('fps', fps_orig))
        except Exception:
            pass

        # skip_step_size already set at the beginning of the function

        # Prefer explicit config override for original frame count
        cfg_base = self.config['training']['checkpoint'].get('original_video_num_frames', None)
        if cfg_base is None:
            try:
                T = len(list(ds_dir.glob('frame_*')))
                base_num_frames = max(0, T)
            except Exception:
                base_num_frames = 480
        else:
            base_num_frames = int(cfg_base)

        # Two-stage sampling for efficiency:
        # Stage 1: Rollout prediction with skip_step_size (e.g., 480//1=480 or 480//10=48)
        # Stage 2: Render subsampling to fixed 48 frames for video
        s = max(1, skip_step_size)
        # Full remaining sequence after skip_step_size, from the sequence's first physical frame
        num_frames_rollout = (max(1, base_num_frames - video_start_frame) + s - 1) // s

        if video_renderer == 'filament':
            # every rollout frame is a key frame: the renderer samples the motion in real time at
            # training.checkpoint.video_fps, interpolating between key frames (step sizes above 1)
            render_subsample = 1
        else:
            # Calculate render subsampling rate to get ~48 frames
            target_render_frames = 48
            render_subsample = max(1, num_frames_rollout // target_render_frames)  # Subsample rollout for rendering
        num_frames_render = (num_frames_rollout + render_subsample - 1) // render_subsample  # Actual render frames

        # Use real-time duration from original data
        # original_duration = base_num_frames / fps_orig (e.g., 600 / 100 = 6s)
        original_duration = base_num_frames / fps_orig
        if video_renderer == 'filament':
            fps_out = float(self.config['training']['checkpoint'].get('video_fps', 30))
        else:
            fps_out = num_frames_render / original_duration  # Maintain real-time playback

        logger.info(
            f"[Sampling Strategy] base={base_num_frames} frames, skip_step_size={skip_step_size} "
            f"-> rollout={num_frames_rollout} frames, render_subsample={render_subsample} "
            f"-> render={num_frames_render} frames @ {fps_out:.2f} fps ({original_duration:.1f}s duration)"
        )

        # Prepare rollout using initial GT frames from the sampled sequence
        _model_ref = self.model.module if isinstance(self.model, DDP) else self.model

        # Apply GPU augmentation (rotation) if enabled for visualization
        # This ensures the visualization shows what the training sees
        rotation_angle_deg = 0.0  # Default: no rotation
        if self.gpu_augmentation is not None and self.gpu_augmentation.rotation_enabled:
            # Apply rotation to batch (modifies obj_positions and ref_obj_positions)
            batch = self.gpu_augmentation(batch, split=split)
            rotation_angle_deg = batch.get('rotation_angle_deg', 0.0)
            if rotation_angle_deg != 0.0:
                logger.info(f"[Video Gen] Applied rotation augmentation: {rotation_angle_deg:.0f}° for verification")

        # Override batch['skip_step_size'] with the video-generation parameter.
        # Dataset samples carry the training step_size (randomly sampled in
        # multi-timescale mode), while video generation evaluates the model at a
        # specific temporal resolution, so the sampled value must be replaced.
        if 'skip_step_size' in batch:
            original_step_size = batch['skip_step_size'].item() if isinstance(batch['skip_step_size'], torch.Tensor) else batch['skip_step_size']
            if if_debug:
                logger.info(f"[DEBUG Video Gen] Overriding batch skip_step_size: {original_step_size} -> {skip_step_size}")
            else:
                logger.info(f"[Video Gen] Overriding batch skip_step_size: {original_step_size} -> {skip_step_size}")
        batch['skip_step_size'] = torch.tensor([skip_step_size], dtype=torch.float32, device=self.device)

        # Optional debug logging of the final step_size before rollout
        if if_debug:
            final_step_size = batch['skip_step_size'].item() if isinstance(batch['skip_step_size'], torch.Tensor) else batch['skip_step_size']
            logger.info(f"[DEBUG Video Gen] Final step_size before rollout: {final_step_size}, num_frames_rollout={num_frames_rollout}")

        # Measure inference speed (FPS)
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        rollout_start_time = time.perf_counter()

        # Get eval_warmup_frames from config (consistent with evaluator.py)
        eval_warmup_frames = self.config['training'].get('metrics_test', {}).get('eval_warmup_frames', 2)

        # Flash Attention handles dtype internally in decoder.py - rest of model runs in FP32 for stability
        with torch.no_grad():
            rollout_out = _model_ref.rollout(
                initial_batch=batch,
                num_steps=num_frames_rollout,
                dt=1.0,
                warmup_frames=eval_warmup_frames
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        rollout_end_time = time.perf_counter()
        rollout_total_time = rollout_end_time - rollout_start_time

        # Calculate inference FPS (frames predicted per second)
        # num_frames_rollout includes eval_warmup_frames GT warmup frames
        num_predicted_frames = max(1, num_frames_rollout - eval_warmup_frames)
        inference_fps = num_predicted_frames / rollout_total_time if rollout_total_time > 0 else 0.0

        logger.info(
            f"[Inference Speed] dataset={dataset_name} split={split} | "
            f"warmup_frames={eval_warmup_frames} | "
            f"Total rollout frames: {num_frames_rollout} | Predicted frames: {num_predicted_frames} | "
            f"Rollout time: {rollout_total_time:.4f}s | "
            f"Inference FPS: {inference_fps:.2f} frames/sec | "
            f"Time per frame: {rollout_total_time/num_predicted_frames*1000:.2f}ms"
        )

        # Ensure FP32 for numpy/Polyscope (BF16 can cause visualization issues)
        pred_full = rollout_out['obj_positions'][0].float().cpu().numpy()  # (T, Nobj, Nv, 3)
        Ttot, Nobj, Nv, _ = pred_full.shape

        # Apply render subsampling to rollout predictions for video efficiency
        # Generate frame indices: uniformly sample from full rollout sequence
        render_frame_indices = np.arange(0, Ttot, render_subsample)[:num_frames_render]
        actual_render_frames = len(render_frame_indices)

        logger.info(
            f"[Render Sampling] Rollout has {Ttot} frames, rendering {actual_render_frames} frames "
            f"(subsample={render_subsample}, indices: {render_frame_indices[0]}..{render_frame_indices[-1]})"
        )

        # Subsample predictions for rendering
        pred_full_render = pred_full[render_frame_indices]  # (num_frames_render, Nobj, Nv, 3)

        # Get predicted and GT accelerations for visualization
        # Note: accelerations are computed at frame centers, so T frames -> T-2 accelerations
        pred_accel_anchor_seq = None    # (T_rollout-2, Nobj, Nq, 3) anchors only
        gt_accel_seq = None             # (T0-2, Nobj, Nv, 3)
        pred_accel_anchor_seq_render = None  # Subsampled for rendering
        gt_accel_seq_render = None           # Subsampled for rendering

        try:
            # Predicted anchor accelerations from rollout (denormalized)
            if 'obj_accelerations_anchor' not in rollout_out:
                raise RuntimeError("Rollout output missing 'obj_accelerations_anchor' for visualization")
            pred_accel_anchor_seq = rollout_out['obj_accelerations_anchor'][0].cpu().numpy()  # (T_rollout-2, Nobj, Nq, 3)

            # Subsample accelerations for rendering
            # Acceleration at index i corresponds to frames i, i+1, i+2 (centered at i+1)
            # For render_frame_indices, we need accelerations at indices render_frame_indices[t]-1 (if t >= 1)
            accel_indices_render = []
            for idx in render_frame_indices:
                if idx >= 1:  # Frame idx has acceleration at idx-1
                    accel_idx = idx - 1
                    if accel_idx < pred_accel_anchor_seq.shape[0]:
                        accel_indices_render.append(accel_idx)
            if len(accel_indices_render) > 0:
                pred_accel_anchor_seq_render = pred_accel_anchor_seq[accel_indices_render]
                logger.info(f"[Acceleration Sampling] Pred accel: {pred_accel_anchor_seq.shape} -> {pred_accel_anchor_seq_render.shape}")
            else:
                pred_accel_anchor_seq_render = None
        except Exception as e:
            logger.error(f"Failed to prepare predicted accelerations for visualization: {e}")
            pred_accel_anchor_seq_render = None

        try:
            # GT accelerations: compute from ground truth positions
            # batch['obj_positions'] shape: (B, T0, Nobj, Nv, 3)
            gt_pos = batch['obj_positions'][0].cpu().numpy()  # (T0, Nobj, Nv, 3)
            if gt_pos.shape[0] >= 3:
                # Compute GT accelerations: a = (x_{t+1} - 2*x_t + x_{t-1}) / dt^2
                # For frames 0,1,2,...,T0-1, we can compute accel at centers 1,2,...,T0-2
                gt_accel_seq = gt_pos[2:] - 2*gt_pos[1:-1] + gt_pos[:-2]  # (T0-2, Nobj, Nv, 3)

                # Subsample GT accelerations (same logic as predicted accelerations)
                gt_accel_indices_render = []
                for idx in render_frame_indices:
                    if idx >= 1:
                        accel_idx = idx - 1
                        if accel_idx < gt_accel_seq.shape[0]:
                            gt_accel_indices_render.append(accel_idx)
                if len(gt_accel_indices_render) > 0:
                    gt_accel_seq_render = gt_accel_seq[gt_accel_indices_render]
                    logger.info(f"[Acceleration Sampling] GT accel: {gt_accel_seq.shape} -> {gt_accel_seq_render.shape}")
                else:
                    gt_accel_seq_render = None
            else:
                pass
        except Exception as e:
            logger.error(f"Failed to prepare GT accelerations for visualization: {e}")
            gt_accel_seq_render = None

        # Masks for valid vertices per frame/object from the sample batch
        try:
            vmask_all = batch['valid_vtx_mask'][0].cpu().numpy()  # (T0, Nobj, Nv)
        except Exception:
            vmask_all = None

        # Load faces for mesh rendering from batch (required; no fallback).
        # This ensures faces match the permuted object order in training data.
        if 'faces_list' not in batch or batch['faces_list'] is None:
            raise RuntimeError(
                f"Missing 'faces_list' in batch! DataLoader must provide faces. "
                f"Check that faces_list is included in sample dict."
            )
        # Use permuted faces from batch (handles object permutation augmentation)
        faces_list = batch['faces_list'][0]  # First sample in batch
        if len(faces_list) != Nobj:
            raise RuntimeError(f"Faces count mismatch: batch has {len(faces_list)} faces but {Nobj} objects")
        logger.info(f"[Faces] Using permuted faces from batch: {len(faces_list)} objects")

        frames_dir = self.log_dir / f'temp_frames_epoch_{self.epoch}_mobj_pred'
        if video_renderer == 'polyscope':
            frames_dir.mkdir(exist_ok=True)
            try:
                if hasattr(ps, 'set_allow_headless'):
                    ps.set_allow_headless(True)
                if hasattr(ps, 'set_backend'):
                    ps.set_backend('openGL3_egl')
                ps.init('openGL3_egl')
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Polyscope headless backend: {e}")
            ps.set_autocenter_structures(False)
            ps.set_ground_plane_mode("none")
            ps.set_ground_plane_height(0.0)
            ps.set_navigation_style('turntable')
            ps.set_up_dir('z_up')

        # Colors per object
        rng = np.random.default_rng(123)
        colors = rng.random((Nobj, 3)) * 0.6 + 0.2

        # Visualization parameters - READ FROM CONFIG
        checkpoint_config = self.config['training']['checkpoint']

        # Camera settings from config
        camera_config = checkpoint_config.get('camera', {})
        camera_mode = camera_config.get('mode', 'auto')

        first = batch['obj_positions'][0, 0].cpu().numpy().reshape(-1, 3)

        if camera_mode == 'fixed':
            # Use fixed camera from config
            center = np.array(camera_config.get('center', [0.0, 0.0, 0.5]), dtype=np.float64)
            eye = np.array(camera_config.get('eye', [3.0, -3.0, 3.0]), dtype=np.float64)
        else:
            # Auto camera: compute from scene bounds
            center = first.mean(axis=0)
            bbox_min = first.min(axis=0)
            bbox_max = first.max(axis=0)
            radius = 0.5 * np.linalg.norm(bbox_max - bbox_min)
            auto_direction = np.array(camera_config.get('auto_direction', [2.0, -2.0, 2.0]), dtype=np.float64)
            auto_scale = camera_config.get('auto_scale', 3.0)
            eye = center + auto_direction * max(radius, 1e-3) * auto_scale

        # Ground plane settings from config
        ground_config = checkpoint_config.get('ground', {})
        ground_size = ground_config.get('size', 10.0)
        ground_color = ground_config.get('color', [0.8, 0.8, 0.8])
        ground_vertices = np.array([
            [-ground_size, -ground_size, 0.0],
            [ ground_size, -ground_size, 0.0],
            [ ground_size,  ground_size, 0.0],
            [-ground_size,  ground_size, 0.0]
        ])
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]])

        total_frames = actual_render_frames

        # GT overlay settings (from config, can be overridden by override_if_overlay parameter)
        if override_if_overlay is not None:
            if_overlay = int(override_if_overlay)  # Use override value
        else:
            if_overlay = 1 if checkpoint_config.get('if_overlay', True) else 0  # Config -> bool, convert to int
        gt_transparency = checkpoint_config.get('gt_transparency', 0.1)  # Default: very transparent (0.1 = 90% transparent)

        # Arrow visualization settings (from config)
        arrow_enable = 1 if checkpoint_config.get('arrow_enable', False) else 0  # Default: disabled
        arrow_scale = checkpoint_config.get('arrow_scale', 1.0)
        arrow_subsample = checkpoint_config.get('arrow_subsample', 1)
        arrow_thickness = checkpoint_config.get('arrow_thickness', 0.0001)
        point_size = checkpoint_config.get('point_size', 0.0001)

        logger.info(f"[Visualization Config] if_overlay={if_overlay}, gt_transparency={gt_transparency}, "
                    f"arrow_enable={arrow_enable}, arrow_scale={arrow_scale}, arrow_thickness={arrow_thickness}, "
                    f"arrow_subsample={arrow_subsample}, point_size={point_size}")
        
        # Prepare GT positions if overlay is enabled
        # Need to load full GT sequence from disk because batch only contains sequence_length frames
        # (e.g., batch has 4 frames but rollout predicts 48 frames)
        gt_full = None
        if if_overlay == 1:
            logger.info(f"[GT Overlay] if_overlay=1, starting GT loading...")
            try:
                # Check batch GT length vs rollout length
                batch_gt = batch['obj_positions'][0].cpu().numpy()  # (T_batch, Nobj, Nv, 3)
                T_batch = batch_gt.shape[0]
                T_rollout = pred_full.shape[0]

                logger.info(f"GT overlay: Batch has {T_batch} GT frames, Rollout has {T_rollout} frames")

                # If rollout is longer than batch (common case), load full GT from disk
                if T_rollout > T_batch:
                    logger.info(f"Loading full GT sequence from disk to match rollout length...")

                    # Find all frame directories
                    frame_dirs = sorted([d for d in ds_dir.glob('frame_*') if d.is_dir()])
                    if len(frame_dirs) == 0:
                        raise RuntimeError(f"No frame_* directories found in {ds_dir}")

                    # Same frames as the rollout: from the sequence's first physical frame at skip_step_size
                    _frame_numbers = [int(d.name.split('_')[-1]) for d in frame_dirs]
                    if video_start_frame not in _frame_numbers:
                        raise RuntimeError(f"frame_{video_start_frame:05d} not found in {ds_dir}")
                    frame_dirs_sampled = frame_dirs[_frame_numbers.index(video_start_frame)::skip_step_size]
                    T_gt = min(len(frame_dirs_sampled), T_rollout)  # Don't exceed rollout length

                    # Load first frame to get dimensions
                    # Use exact file pattern obj_{i}_pos.npy to avoid matching point cloud versions
                    # (e.g., obj_0_pc1024_pos.npy, obj_0_pc512_pos.npy, etc.)
                    first_frame = frame_dirs_sampled[0]

                    # Count objects by checking which obj_{i}_pos.npy files exist
                    obj_files_first = []
                    obj_idx = 0
                    while True:
                        obj_file = first_frame / f"obj_{obj_idx}_pos.npy"
                        if obj_file.exists():
                            obj_files_first.append(obj_file)
                            obj_idx += 1
                        else:
                            break

                    Nobj_gt = len(obj_files_first)
                    if Nobj_gt == 0:
                        raise RuntimeError(f"No obj_{{i}}_pos.npy files found in {first_frame}")

                    # Get max vertex count across all objects (different objects may have different Nv)
                    Nv_per_obj = []
                    for obj_file in obj_files_first:
                        obj_pos = np.load(obj_file)
                        Nv_per_obj.append(obj_pos.shape[0])
                    Nv_max = max(Nv_per_obj)

                    logger.info(f"GT loading: Nobj={Nobj_gt}, Nv per object={Nv_per_obj}, Nv_max={Nv_max}")

                    # Allocate array with max vertex count (pad smaller objects with zeros)
                    gt_full = np.zeros((T_gt, Nobj_gt, Nv_max, 3), dtype=np.float32)

                    # Load all sampled frames using exact file pattern obj_{i}_pos.npy
                    for t_idx in range(T_gt):
                        frame_dir = frame_dirs_sampled[t_idx]
                        for obj_idx in range(Nobj_gt):
                            obj_file = frame_dir / f"obj_{obj_idx}_pos.npy"
                            if obj_file.exists():
                                obj_pos = np.load(obj_file).astype(np.float32)
                                if self.config['data'].get('object_scale_file'):
                                    if not hasattr(self, '_video_object_scale'):
                                        from src.data.object_scale import from_config as _osf
                                        self._video_object_scale = _osf(self.config)
                                    obj_pos = self._video_object_scale.apply(obj_file, obj_pos)
                                Nv_curr = obj_pos.shape[0]
                                gt_full[t_idx, obj_idx, :Nv_curr, :] = obj_pos

                    logger.info(f"Loaded full GT from disk: shape {gt_full.shape}")
                    gt_from_disk = True

                    # Apply permutation to GT to match batch order (if augmentation is enabled)
                    # This ensures GT meshes match the permuted faces_list from batch
                    if 'permutation' in batch and batch['permutation'] is not None:
                        perm = batch['permutation'][0]  # Get permutation from batch (batch size 1)
                        if isinstance(perm, torch.Tensor):
                            perm = perm.cpu().numpy()
                        logger.info(f"[GT Permutation] Applying object permutation to GT: {perm.tolist()}")
                        gt_full = gt_full[:, perm]  # Permute object dimension: (T, Nobj, Nv, 3)[:, perm]
                        logger.info(f"[GT Permutation] GT permuted successfully")
                else:
                    # Batch GT is sufficient (training mode with short rollout); it already carries the batch's
                    # rotation augmentation
                    gt_full = batch_gt
                    gt_from_disk = False
                    logger.info(f"Using batch GT: shape {gt_full.shape}")
                    # Batch GT is already permuted by DataLoader, no need to permute again

                # Assert GT length matches rollout length for proper overlay
                assert gt_full.shape[0] == T_rollout, (
                    f"GT length mismatch! GT has {gt_full.shape[0]} frames, "
                    f"but rollout has {T_rollout} frames (num_steps={num_frames_rollout}). "
                    f"GT and rollout must have same length for proper overlay visualization."
                )
                logger.info(f"GT-Rollout length check passed: {gt_full.shape[0]} frames")
                logger.info(f"[GT Overlay] GT loaded successfully: gt_full.shape={gt_full.shape}")

                # Apply same rotation to GT as was applied to batch (for augmentation verification); ground truth
                # taken from the batch is already rotated
                if rotation_angle_deg != 0.0 and gt_from_disk:
                    import math
                    logger.info(f"[GT Overlay] Applying {rotation_angle_deg:.0f}° rotation to GT for consistency")
                    angle_rad = math.radians(rotation_angle_deg)
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    # Z-axis rotation matrix (numpy)
                    R_np = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a,  cos_a, 0],
                        [0,      0,     1]
                    ], dtype=np.float32)

                    # Rotate all GT positions: gt_full shape (T, Nobj, Nv, 3)
                    T_gt, Nobj_gt, Nv_gt, _ = gt_full.shape
                    gt_flat = gt_full.reshape(-1, 3)  # (T*Nobj*Nv, 3)
                    gt_rotated = np.matmul(gt_flat, R_np.T)  # (T*Nobj*Nv, 3)
                    gt_full = gt_rotated.reshape(T_gt, Nobj_gt, Nv_gt, 3)
                    logger.info(f"[GT Overlay] GT rotation applied successfully")

                # Apply same render subsampling to GT for video efficiency
                gt_full_render = gt_full[render_frame_indices]  # (num_frames_render, Nobj, Nv, 3)
                logger.info(f"[GT Overlay] Applied render subsampling: {gt_full.shape} -> {gt_full_render.shape}")
                gt_full = gt_full_render  # Use subsampled GT for rendering

            except Exception as e:
                if (self.config.get('inference', {}) or {}).get('mode', False):
                    raise RuntimeError(f"[GT Overlay] requested overlay, ground truth could not be loaded: {e}") from e
                logger.warning(f"[GT Overlay] Failed to load GT positions for overlay: {e}")
                import traceback
                traceback.print_exc()
                gt_full = None
                if_overlay = 0  # Disable overlay if GT is unavailable (training-time diagnostics only)
                logger.warning(f"[GT Overlay] GT loading failed, disabling overlay. if_overlay set to 0")

        if video_renderer == 'filament':
            # real time: key frame k is k * skip_step_size physical frames after the first
            times = render_frame_indices * s / fps_orig
            self._render_video_filament(ds_dir, batch, pred_full_render, gt_full if if_overlay == 1 else None,
                                        vmask_all, output_path, fps_out, times)
            return inference_fps

        for t in range(total_frames):
            # Register ground
            ps.register_surface_mesh("ground", ground_vertices, ground_faces, enabled=True)
            ps.get_surface_mesh("ground").set_color(ground_color)
            for i in range(Nobj):
                # Render predicted mesh (opaque) - using subsampled predictions
                V_pred = pred_full_render[t, i]
                F = faces_list[i]
                ps.register_surface_mesh(f"obj_{i}_pred", V_pred, F, enabled=True, smooth_shade=True)
                ps.get_surface_mesh(f"obj_{i}_pred").set_color(colors[i])
                ps.get_surface_mesh(f"obj_{i}_pred").set_transparency(1.0)  # Fully opaque

                # Render GT mesh (transparent) if overlay is enabled
                if if_overlay == 1 and gt_full is not None and t < gt_full.shape[0] and i < gt_full.shape[1]:
                    V_gt_full = gt_full[t, i]  # (Nv_max, 3) - may contain zero-padding

                    # Find actual number of non-zero vertices for this object
                    # (GT array is padded with zeros for objects with fewer vertices)
                    # Check if vertex has any non-zero coordinates
                    vertex_norms = np.linalg.norm(V_gt_full, axis=1)
                    # Count consecutive non-zero vertices from the start
                    # (assumes padding is at the end)
                    Nv_actual = V_pred.shape[0]  # Use predicted vertex count as reference

                    # Trim GT vertices to match predicted vertex count
                    V_gt = V_gt_full[:Nv_actual]

                    # Use faces as-is (they should match the actual vertex count)
                    F_gt = F

                    # Filter faces to only use valid vertex indices
                    F_gt = F_gt[np.all(F_gt < Nv_actual, axis=1)]

                    # GT mesh with same faces, rendered highly transparent so it does not obscure the prediction
                    if F_gt.shape[0] > 0:  # Only render if we have valid faces
                        ps.register_surface_mesh(f"obj_{i}_gt", V_gt, F_gt, enabled=True, smooth_shade=True)
                        ps.get_surface_mesh(f"obj_{i}_gt").set_color([0.9, 0.9, 0.9])  # Light gray
                        ps.get_surface_mesh(f"obj_{i}_gt").set_transparency(gt_transparency)  # From config (default: 0.1 = 90% transparent)

                # For backwards compatibility: also register without _pred suffix
                # (arrows reference "obj_{i}" positions)
                V = V_pred  # Use predicted positions for arrow attachment

                # Visualize predicted (red) and GT (green) acceleration arrows
                if arrow_enable == 1:
                    try:
                        # Optional subsampling for clarity/performance
                        arrow_k = max(1, arrow_subsample)

                        # Anchor indices for this frame/object
                        anchor_idx_np = batch['anchor_idx'][0, i].cpu().numpy()  # (Nq,)
                        # Build a unified index into the anchor set to apply mask and subsample consistently
                        sel_idx = np.arange(anchor_idx_np.shape[0])  # base 0..Nq-1
                        if vmask_all is not None:
                            mask_idx = min(t, vmask_all.shape[0] - 1)
                            m_full = vmask_all[mask_idx, i].astype(bool)  # (Nv,)
                            anchor_mask = m_full[anchor_idx_np]           # (Nq,)
                            sel_idx = sel_idx[anchor_mask]
                        if arrow_k > 1:
                            sel_idx = sel_idx[::arrow_k]
                        if sel_idx.size == 0:
                            continue
                        # Sampled anchor positions (for vectors attachment)
                        V_sampled = V[anchor_idx_np[sel_idx]]

                        # Predicted acceleration on anchors (RED arrows) - using subsampled accelerations
                        if pred_accel_anchor_seq_render is not None and t >= 1 and (t-1) < pred_accel_anchor_seq_render.shape[0]:
                            # render_t=0 has no accel (frame 0), render_t=1..N correspond to accel_idx=0..N-1
                            pred_accel_anchor_t = pred_accel_anchor_seq_render[t-1, i]  # (Nq, 3)
                            pred_accel_sampled = pred_accel_anchor_t[sel_idx]

                            # Register point cloud with red arrows (points invisible, only arrows visible)
                            pc_pred = ps.register_point_cloud(f"obj_{i}_pred_accel", V_sampled,
                                                              radius=0.0,
                                                              color=(1.0, 0.0, 0.0))
                            pc_pred.add_vector_quantity("pred_accel", pred_accel_sampled,
                                                        enabled=True,
                                                        length=arrow_scale,
                                                        radius=arrow_thickness,
                                                        color=(1.0, 0.0, 0.0))
                        else:
                            pass  # Frame 0 or accel index invalid

                        # GT acceleration (green arrows) at the same anchor vertices - using subsampled accelerations
                        if gt_accel_seq_render is not None and t >= 1 and (t-1) < gt_accel_seq_render.shape[0]:
                            gt_accel_t = gt_accel_seq_render[t-1, i]  # (Nv, 3)
                            gt_accel_sampled = gt_accel_t[anchor_idx_np[sel_idx]]

                            # Register point cloud with green arrows (points invisible, only arrows visible)
                            pc_gt = ps.register_point_cloud(f"obj_{i}_gt_accel", V_sampled,
                                                            radius=0.0,
                                                            color=(0.0, 1.0, 0.0))
                            pc_gt.add_vector_quantity("gt_accel", gt_accel_sampled,
                                                      enabled=True,
                                                      length=arrow_scale,
                                                      radius=arrow_thickness,
                                                      color=(0.0, 1.0, 0.0))
                        else:
                            pass

                    except Exception as e:
                        logger.debug(f"Failed to visualize acceleration arrows: {e}")
                        pass
            ps.look_at(eye, center)  # ps.look_at(camera_location, target)
            ps.screenshot(str(frames_dir / f'frame_{t:05d}.png'))
            ps.remove_all_structures()

        # Console log synthesis parameters
        try:
            expected_duration = float(total_frames) / max(1e-6, fps_out)
        except Exception:
            expected_duration = -1.0
        logger.info(
            f"[video] dataset={dataset_name} split={split} fps_orig={fps_orig} skip={skip_step_size} "
            f"fps_out={fps_out:.6f} base_num_frames={base_num_frames} "
            f"rollout_frames={num_frames_rollout} rendered_frames={total_frames} expected_duration_s={expected_duration:.3f}"
        )

        # Encode video
        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-r', f"{fps_out:.6f}",
            '-i', str(frames_dir / 'frame_%05d.png'),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'mpeg4', '-q:v', '2', '-pix_fmt', 'yuv420p', '-y',
            str(output_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        # Clean up temporary frames
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        logger.info(f"  [multiobj_pred] Cleaned up temp frames: {frames_dir}")

        # Return inference FPS for video file naming
        return inference_fps

    def _visualize_predictions_vs_gt(self, predictions, batch, batch_idx=0):
        """Visualize predictions vs ground truth (multi-object)."""
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        vis_dir = self.log_dir / 'per_frame_prediction'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        if 'obj_positions_denorm' not in predictions or 'obj_positions' not in batch:
            logger.warning("Multi-object tensors missing for visualization; skipping.")
            return
        
        pred = predictions['obj_positions_denorm'][0].detach().cpu().numpy()    # (T-2,Nobj,Nv,3)
        gt_all = batch['obj_positions'][0].detach().cpu().numpy()               # (T,Nobj,Nv,3)
        vmask_all = batch['valid_vtx_mask'][0].detach().cpu().numpy()           # (T,Nobj,Nv)
        obj_valid = batch['valid_obj_mask'][0].detach().cpu().numpy()           # (Nobj,)
        Tm2, Nobj, Nv, _ = pred.shape
        num_pred_frames = Tm2
        
        rng = np.random.default_rng(123)
        colors_pred = rng.random((Nobj, 3)) * 0.6 + 0.2
        colors_gt = np.clip(colors_pred + 0.4, 0.0, 1.0)
        
        fig = plt.figure(figsize=(20, 5 * ((num_pred_frames + 1) // 4 + 1)))
        for t in range(num_pred_frames):
            ax = fig.add_subplot((num_pred_frames + 3) // 4, 4, t + 1, projection='3d')
            frame_pred = pred[t]                # (Nobj,Nv,3)
            frame_gt = gt_all[t + 2]           # (Nobj,Nv,3)
            frame_mask = vmask_all[t + 2]      # (Nobj,Nv)
            any_points = False
            for j in range(Nobj):
                if not obj_valid[j]:
                    continue
                m = frame_mask[j]
                if not np.any(m):
                    continue
                any_points = True
                P = frame_pred[j][m]
                G = frame_gt[j][m]
                ax.scatter(P[:, 0], P[:, 2], P[:, 1], c=[colors_pred[j]], marker='o', s=18, alpha=0.7)
                ax.scatter(G[:, 0], G[:, 2], G[:, 1], c=[colors_gt[j]], marker='^', s=14, alpha=0.8)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_title(f'Frame {t+2}')
            if not any_points:
                ax.text2D(0.1, 0.5, "no valid points", transform=ax.transAxes)
        
        plot_path = vis_dir / f'pred_vs_gt_multi_epoch_{self.epoch:04d}_batch_{batch_idx:03d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    
    
    @staticmethod
    def _atomic_save(obj, path):
        """torch.save to <path>.part, then rename: a reader (the EMA watcher, a resume) never sees a partly written file,
        and an interrupted write leaves the previous file intact."""
        path = Path(path)
        tmp = path.with_name(path.name + '.part')
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint (rank0 only in DDP mode)."""
        # Only rank 0 saves checkpoints to prevent file conflicts in DDP
        if not self.is_main:
            return

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': float(self.best_val_loss),
            'config': self.config,
            # The scheduler state above was captured after the end-of-epoch step (see train()).
            'scheduler_stepped': True,
        }
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
            checkpoint['ema_num_updates'] = self.ema.num_updates
            # Weights-only file of the averaged model, loadable by the standard evaluation driver.
            _ema_every = int(self.config['training'].get('ema', {}).get('save_every_n_epochs', 1))
            if self.epoch % _ema_every == 0 or self.epoch == int(self.config['training']['max_epochs']) - 1:
                self._atomic_save({'epoch': self.epoch, 'model_state_dict': checkpoint['ema_state_dict'],
                                   'config': self.config, 'ema_decay': self.ema.decay,
                                   'ema_num_updates': self.ema.num_updates},
                                  self.checkpoint_dir / f'epoch_{self.epoch:04d}_ema.pth')

        # Save the best checkpoint before the latest one: the best-model threshold that latest.pth carries (and a resume
        # continues from) always belongs to a best.pth that was written
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            self._atomic_save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        self._atomic_save(checkpoint, latest_path)

        # Generate visualization video if enabled (do this first, independent of checkpoint saving)
        self.generate_visualization_video()

        # Save periodic checkpoint
        save_every = self.config['training']['checkpoint'].get('save_every_n_epochs', 10)
        if self.epoch % save_every == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.epoch:04d}.pth'
            self._atomic_save(checkpoint, epoch_path)
    
    def run_inference(self):
        """
        Inference-only mode: Generate videos and compute RMSE metrics.
        Called when --inference flag is passed or inference.mode=true in config.
        """
        logger.info("=" * 80)
        logger.info("INFERENCE MODE STARTED")
        logger.info("=" * 80)

        self.model.eval()
        self.epoch = 0  # Set epoch for logging purposes

        # Get inference config
        inference_config = self.config.get('inference', {})
        generate_videos = inference_config.get('generate_videos', True)
        compute_metrics = inference_config.get('compute_metrics', True)

        # Step 1: Compute RMSE metrics
        if compute_metrics:
            logger.info("-" * 40)
            logger.info("Step 1: Computing RMSE Metrics")
            logger.info("-" * 40)
            try:
                test_metrics = self.evaluator.evaluate_rollout_rmse(epoch=0)
                if test_metrics:
                    logger.info("=" * 60)
                    logger.info("INFERENCE RESULTS - RMSE Metrics:")
                    logger.info("=" * 60)
                    for key, value in sorted(test_metrics.items()):
                        if isinstance(value, float):
                            logger.info(f"  {key}: {value:.6f}")
                        else:
                            logger.info(f"  {key}: {value}")
                    logger.info("=" * 60)
                else:
                    logger.warning("No metrics returned from evaluation")
            except Exception as e:
                # an evaluation run must fail when its metrics fail (a partial eval.txt must not pass as a result)
                logger.error(f"RMSE evaluation failed: {e}")
                raise
        else:
            logger.info("Metrics computation skipped (inference.compute_metrics=false)")

        # Step 2: Generate visualization videos (after the metrics, which they cannot affect)
        if generate_videos:
            logger.info("-" * 40)
            logger.info("Step 2: Generating Visualization Videos")
            logger.info("-" * 40)
            # a failed requested video fails the run (generate_visualization_video raises in inference mode)
            self.generate_visualization_video()
            logger.info("Video generation completed")
        else:
            logger.info("Video generation skipped (inference.generate_videos=false)")

        # Step 3: Optional validation pass (if enabled)
        if self.enable_validation and self.val_loader is not None:
            logger.info("-" * 40)
            logger.info("Step 3: Running Validation Pass")
            logger.info("-" * 40)
            try:
                val_loss = self.validate()
                logger.info(f"Validation Loss: {val_loss:.6f}")
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                import traceback
                traceback.print_exc()

        logger.info("=" * 80)
        logger.info("INFERENCE MODE COMPLETED")
        logger.info("=" * 80)

        # Close tensorboard writer
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

    def train(self):
        """Main training loop (or inference-only for empty train set)."""
        if self.inference_only:
            logger.info("Inference-only mode: running single validation pass and generating videos")
            self.epoch = 0
            if self.enable_validation:
                val_loss = self.validate()
                logger.info(f"Validation Loss: {val_loss:.4f}")
            else:
                logger.info("Validation skipped (disabled by config)")

            # Try to generate videos, but gracefully handle GPU memory errors in CPU mode
            try:
                self.generate_visualization_video()
            except (RuntimeError, Exception) as e:
                if 'out of memory' in str(e).lower() or 'opengl' in str(e).lower():
                    logger.error(f"Video generation failed (GPU memory/OpenGL error): {e}")
                    logger.warning("Skipping video generation - GPU memory insufficient for Polyscope rendering")
                    logger.warning("To generate videos, free GPU memory or run on a machine with an available GPU")
                else:
                    raise

            logger.info("Inference-only mode completed")
            return

        logger.info(f"Starting training for {self.config['training']['max_epochs']} epochs")

        # Initial evaluation BEFORE training starts (epoch=-1 to distinguish from post-epoch evals)
        # Controlled by training.metrics_test.initial_eval config flag
        initial_eval_enabled = self.config['training'].get('metrics_test', {}).get('initial_eval', False)
        if self.evaluator.metrics_enabled and initial_eval_enabled:
            logger.info("=" * 80)
            logger.info("[Initial Evaluation] Running RMSE evaluation BEFORE training starts")
            logger.info("=" * 80)
            # Use epoch=-1 to indicate pre-training evaluation; a failure stops the run (as the per-epoch one does)
            initial_metrics = self.evaluator.evaluate_rollout_rmse(epoch=-1)
            if initial_metrics:
                logger.info("[Initial Evaluation] Completed successfully")
        elif self.evaluator.metrics_enabled:
            logger.info("[Initial Evaluation] Skipped (initial_eval=false in config)")

        max_epochs = self.config['training']['max_epochs']
        if self.start_epoch > 0:
            if self.start_epoch >= max_epochs:
                raise RuntimeError(
                    f"Resumed checkpoint already completed epoch {self.start_epoch - 1} but "
                    f"training.max_epochs={max_epochs}: raise max_epochs to continue, or fine-tune "
                    f"from a weights-only checkpoint to start a new schedule")
            logger.info(f"[Resume] Continuing training at epoch {self.start_epoch} of {max_epochs}")
        for epoch in range(self.start_epoch, max_epochs):
            self.epoch = epoch

            # Update DistributedSampler epoch for proper shuffling in multi-GPU training;
            # otherwise every epoch uses the same random seed and the same data order.
            try:
                if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                if hasattr(self.val_loader, 'sampler') and hasattr(self.val_loader.sampler, 'set_epoch'):
                    self.val_loader.sampler.set_epoch(epoch)
            except Exception:
                pass

            # Expose current epoch to model for warmup/ramp schedules (handle DDP)
            try:
                if isinstance(self.model, DDP):
                    setattr(self.model.module, 'current_epoch', epoch)
                else:
                    setattr(self.model, 'current_epoch', epoch)
            except Exception:
                pass

            # Update dataset epoch for noise scheduling
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
                # Log noise status
                noise_config = self.config['data'].get('augmentation', {}).get('random_walk', {})
                if noise_config.get('enable', False):
                    start_epoch = noise_config.get('start_epoch', 0)
                    noise_std = noise_config.get('noise_std', 0)
                    if epoch >= start_epoch:
                        logger.info(f"Epoch {epoch}: Random walk noise ACTIVE (std={noise_std})")
                    else:
                        logger.info(f"Epoch {epoch}: Random walk noise INACTIVE (starts at epoch {start_epoch})")

            # Training
            train_stats = self.train_epoch()
            logger.info(
                "Epoch %d: Train Loss = %.4g | pos_before=%.4g, pos_after=%.4g, accel_before=%.4g, accel_after=%.4g, "
                "edge=%.4g, pen_loss=%.4g, pen_acc=%.4g",
                epoch,
                train_stats['loss'],
                train_stats['pos_before_icp'],
                train_stats['pos_after_icp'],
                train_stats['accel_before_icp'],
                train_stats['accel_after_icp'],
                train_stats['edge'],
                train_stats['pen_sign_loss'],
                train_stats['pen_sign_acc'],
            )

            # Synchronize all ranks before validation in DDP mode
            # Ensures all GPUs finish the training epoch before starting validation
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            # Validation
            val_loss = None
            if self.enable_validation:
                val_loss = self.validate()
                # visualization is inside the self.validate() function
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Validation skipped (disabled by config)")

            # RMSE evaluation on test/val sets (HopNet-style checkpoints). A failure stops the run (with the
            # traceback): training on without evaluations would also skip best-checkpoint selection, and one failing
            # rank would leave the others waiting in the evaluator's collectives.
            test_metrics = {}
            evaluated = self.evaluator.should_evaluate(epoch)
            if evaluated:
                test_metrics = self.evaluator.evaluate_rollout_rmse(epoch) or {}

            # One-time consistency check after epoch 1 (only on main process)
            if epoch == 1 and self.is_main:
                if not hasattr(self, '_consistency_check_done'):
                    try:
                        self.check_forward_rollout_consistency()
                        self._consistency_check_done = True
                    except Exception as e:
                        logger.error(f"Consistency check failed: {e}")
                        raise

            # Check if best model
            # When validation is disabled, use test metrics (position_rmse_step050) instead
            if self.enable_validation and val_loss is not None:
                # Use validation loss as best criterion
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = float(val_loss)
                    logger.info(f"New best model: Val Loss = {val_loss:.4f}")
            else:
                # Use a rollout position-RMSE metric as best criterion (lower is better). The
                # evaluator reports keys of the form 'p_rmse_@<frame>/s=<step size>'.
                # training.checkpoint.best_metric: one key, or a list of keys whose mean is the criterion (e.g. the
                # step-size-1 long-horizon checkpoints @200/@300/@400)
                best_key = self.config['training'].get('checkpoint', {}).get('best_metric', 'p_rmse_@050/s=1')
                _best_keys = list(best_key) if isinstance(best_key, (list, tuple)) else [best_key]
                if evaluated and self.is_main and self.evaluator.metrics_enabled and not (
                        all(k in test_metrics for k in _best_keys) and all(np.isfinite(float(test_metrics[k])) for k in _best_keys)):
                    # the evaluator returns the metrics on the main rank; missing or non-finite selection metrics
                    # would silently stop best-checkpoint selection
                    raise RuntimeError(f"epoch {epoch}: checkpoint-selection metrics {_best_keys} missing or non-finite "
                                       f"in the evaluation ({ {k: test_metrics.get(k) for k in _best_keys} })")
                if test_metrics and all(k in test_metrics for k in _best_keys):
                    # float(): the evaluator's aggregates are numpy scalars, and a numpy scalar
                    # saved in the checkpoint breaks torch.load's default weights_only=True path.
                    current_rmse = float(sum(float(test_metrics[k]) for k in _best_keys) / len(_best_keys))
                    is_best = current_rmse < self.best_val_loss
                    if is_best:
                        self.best_val_loss = current_rmse
                        logger.info(f"New best model: {best_key} = {current_rmse:.6f} m")
                else:
                    # No metrics available, skip best check
                    is_best = False
                    logger.info("No test metrics available, skipping best checkpoint check")

            # Advance the LR schedule (cosine/constant schedulers) BEFORE saving, so that the
            # scheduler state stored in the checkpoint is the one a continuous run uses for the
            # next epoch; a resumed run then reproduces the same LR sequence.
            self.scheduler.step()

            # Save checkpoint and save the visualization video
            self.save_checkpoint(is_best=is_best)

            # Log per-parameter-group learning rates
            for i, pg in enumerate(self.optimizer.param_groups):
                group_name = 'base' if i == 0 else f'group{i}'
                actual_lr = pg.get('lr', 0.0)
                tag = f'train/lr_{group_name}'
                self.writer.add_scalar(tag, actual_lr, epoch)
                if epoch % 10 == 0:
                    logger.info(f"[LR Tracking] {group_name}: {actual_lr:.6e}")
            
            # Log teacher forcing ratio if using scheduled sampling
            if self.use_scheduled_sampling:
                teacher_ratio = self.get_teacher_forcing_ratio(epoch)
                self.writer.add_scalar('train/teacher_forcing_ratio', teacher_ratio, epoch)
                logger.info(f"Epoch {epoch}: Teacher Forcing Ratio = {teacher_ratio:.3f}")
        
        logger.info("Training completed!")
        self.writer.close()


def _adapt_state_dict_for_height_input(model, state_dict):
    """Zero-pad the encoder's first convolution for the optional height input channel so a
    checkpoint trained without it warm-starts with an identical function."""
    msd = model.state_dict()
    for key, w in list(state_dict.items()):
        if key.endswith('shared_obj_encoder.conv1.weight') and key in msd and w.dim() == 3:
            want = msd[key].shape[1]
            if w.shape[1] == want - 1:
                pad = torch.zeros(w.shape[0], 1, w.shape[2], dtype=w.dtype, device=w.device)
                state_dict[key] = torch.cat([w, pad], dim=1)
                print(f"[Checkpoint] height_input: {key} zero-padded {w.shape[1]}->{want} input channels (exact warm start)", flush=True)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='Train the RigidFormer multi-body dynamics model')
    parser.add_argument('--config', type=str, default='configs/movi_b.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from (alias for --resume)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--use_multi_dataset', action='store_true',
                       help='Use multi-dataset loader for training on multiple datasets')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference only (no training). Generates videos and computes metrics.')
    parser.add_argument('--compile', action='store_true',
                       help='Compile the encoder, decoder and predictor with torch.compile for faster '
                            'inference. One-time compilation cost on startup; results may differ from '
                            'the eager path at floating-point rounding level.')
    parser.add_argument('--cuda-graph', action='store_true',
                       help='Capture the rollout step as a CUDA graph and replay it, roughly halving '
                            'per-step inference time. Trajectories are identical to the standard path.')
    args = parser.parse_args()
    if args.cuda_graph:
        os.environ['RIGIDFORMER_CUDA_GRAPH'] = '1'

    # Handle both --resume and --checkpoint arguments
    checkpoint_path = args.resume or args.checkpoint

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create dataloaders with split ratios from config
    train_ratio = config['data'].get('train_split', 0.8)
    val_ratio = config['data'].get('val_split', 0.1)

    # Check if validation is enabled
    enable_validation = config['training'].get('enable_validation', True)
    if not enable_validation:
        logger.info("Validation disabled by config - val_loader will not be created")
        val_ratio = 0.0  # Allocate all non-train data to test set

    # DDP setup (optional)
    ddp_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    ddp_rank = int(os.environ.get('RANK', '0'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    use_ddp = ddp_world_size > 1
    if use_ddp:
        if args.device == 'cuda':
            torch.cuda.set_device(ddp_local_rank)
        # Set 8-hour timeout for NCCL operations (evaluation can be slow)
        from datetime import timedelta
        nccl_timeout = timedelta(hours=8)
        dist.init_process_group(
            backend='nccl' if args.device == 'cuda' else 'gloo',
            timeout=nccl_timeout
        )
        logger.info(f"Initialized DDP: rank={ddp_rank}, world_size={ddp_world_size}, local_rank={ddp_local_rank}, timeout=8h")

    # Set random seeds for reproducibility.
    # In DDP, each rank gets a different seed for data augmentation
    # but the same seed for model initialization.
    base_seed = args.seed
    seed = base_seed + (ddp_rank if use_ddp else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(base_seed)  # Same model init across all ranks
    torch.cuda.manual_seed_all(base_seed)  # Same model init across all GPUs
    logger.info(f"Set random seed: base={base_seed}, rank_seed={seed}")

    # Check for inference mode early (before dataloader creation)
    inference_mode = args.inference or config.get('inference', {}).get('mode', False)
    if inference_mode:
        config.setdefault('inference', {})['mode'] = True   # one source of truth for --inference and inference.mode
    if inference_mode:
        logger.info("=" * 60)
        logger.info("INFERENCE MODE DETECTED - Skipping train dataset indexing")
        logger.info("=" * 60)

    # Pure multi-object mode (MOVi datasets only)
    logger.info("Using multi-dataset loader")
    logger.info(f"Data path: {config['data']['dataset_path']}")
    logger.info(f"Split file: {config['data']['split_file']} (required)")
    from src.data.multi_dataloader import check_scene_set
    logger.info(f"Scene set: {check_scene_set(config)}")
    if ((config.get('data', {}) or {}).get('if_point_cloud') or {}).get('enable', False):
        # never forwarded to the datasets: the run would silently use mesh vertices
        raise NotImplementedError('data.if_point_cloud.enable: point-cloud inputs are not supported by train.py')
    if not inference_mode:
        logger.info("Creating dataloaders... this may take a few minutes on first run")
    # Get noise configuration from config
    noise_config = config['data'].get('augmentation', {})
    if noise_config:
        # Add dt for physics calculations
        noise_config['dt'] = config['data'].get('dt', 1.0)
        logger.info(f"Noise augmentation config: {noise_config}")

    train_loader, val_loader, test_loader = create_multi_dataloaders(
        data_path=config['data']['dataset_path'],
        train_batch_size=config['training']['train_batch_size'],
        val_batch_size=config['training']['val_batch_size'],
        eval_batch_size=config['training']['eval_batch_size'],
        sequence_length=config['data']['sequence_length'],
        num_workers=config['data']['num_workers'],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_file=config.get('data', {}).get('split_file', 'dataset_split.json'),
        skip_step_size=config['data'].get('skip_step_size', 1),
        skip_first_frames=config['data'].get('skip_first_frames', 2),
        load_gt_motion=config['data'].get('load_gt_motion', False),
        distributed=use_ddp,
        rank=ddp_rank,
        world_size=ddp_world_size,
        noise_config=noise_config,
        augmentation=config['data'].get('augmentation', {}),
        anchor_fps_start=config['data'].get('anchor_fps_start', 'index0'),
        fps_order=config['data'].get('fps_order', False),
        object_scale_file=config['data'].get('object_scale_file'),
        anchor_k=config['data'].get('anchor_k', None),
        max_vertices_per_object=config['data'].get('max_vertices_per_object', None),
        max_objects_per_frame=config['data'].get('max_objects_per_frame', None),
        if_phys_param=config.get('model', {}).get('if_phys_param', False),
        random_step_size=config['data'].get('random_step_size', {'enable': False}),  # Multi-timescale training config
        inference_mode=inference_mode  # Skip train dataset indexing in inference mode
    )
    try:
        if enable_validation:
            logger.info(f"Dataloaders ready. train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        else:
            logger.info(f"Dataloaders ready. train_batches={len(train_loader)} (val disabled)")
    except Exception:
        logger.info("Dataloaders ready.")
    
    # Create model
    model = CUT3RDynamicsModel(config)

    # Print parameter counts for each module
    logger.info("=" * 80)
    logger.info("Model Parameter Breakdown:")
    logger.info("=" * 80)

    # First pass: collect all parameter counts
    module_stats = []
    total_params = 0

    if hasattr(model, 'enable_multi_object') and model.enable_multi_object:
        # Multi-object modules
        if hasattr(model, 'shared_obj_encoder'):
            encoder_params = sum(p.numel() for p in model.shared_obj_encoder.parameters())
            total_params += encoder_params
            module_stats.append(("PointNet Encoder (shared)", encoder_params))

        if hasattr(model, 'obj_feature_proj'):
            obj_proj_params = sum(p.numel() for p in model.obj_feature_proj.parameters())
            total_params += obj_proj_params
            module_stats.append(("Object Feature Projection", obj_proj_params))

        if hasattr(model, 'anchor_query_proj'):
            anchor_proj_params = sum(p.numel() for p in model.anchor_query_proj.parameters())
            total_params += anchor_proj_params
            module_stats.append(("Anchor Query Projection", anchor_proj_params))

    # State tokens (learnable parameters)
    if hasattr(model, 'state_tokens'):
        state_params = model.state_tokens.numel()
        total_params += state_params
        module_stats.append(("State Tokens (learnable)", state_params))

    if hasattr(model, 'state_pos'):
        state_pos_params = model.state_pos.numel()
        total_params += state_pos_params
        module_stats.append(("State Positions (learnable)", state_pos_params))

    # Decoder
    if hasattr(model, 'decoder'):
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        total_params += decoder_params
        module_stats.append(("Bidirectional Decoder", decoder_params))

    # Predictor
    if hasattr(model, 'predictor'):
        predictor_params = sum(p.numel() for p in model.predictor.parameters())
        total_params += predictor_params
        module_stats.append(("Multi-scale Predictor", predictor_params))

    # Transform layer
    if hasattr(model, 'transform_layer'):
        transform_params = sum(p.numel() for p in model.transform_layer.parameters())
        total_params += transform_params
        module_stats.append(("Transform Layer", transform_params))

    # Second pass: print with percentages
    for name, param_count in module_stats:
        percentage = (param_count / total_params * 100) if total_params > 0 else 0
        logger.info(f"  {name:35s} {param_count:>12,} params  ({percentage:>5.2f}%)")



    logger.info("=" * 80)
    total_params_millions = total_params / 1_000_000
    model_size_mb = total_params * 4 / 1024 / 1024
    logger.info(f"  TOTAL TRAINABLE PARAMETERS:       {total_params:>12,} params ({total_params_millions:.1f}M)")
    logger.info(f"  TOTAL MODEL SIZE:                 {model_size_mb:>12.2f} MB (FP32)")
    logger.info("=" * 80)

    # (model.predictor.rmsnorm_eps is applied in the model constructor)

    # Optional torch.compile of fixed-shape submodules for TRAINING (training.compile). The decoder and the
    # predictor see (B, 16 object slots, 4 anchors) tensors, so they compile to a few graphs whose fused kernels
    # replace thousands of small launches per step. nn.Module.compile() compiles in place: parameter names and
    # checkpoints are unchanged. Results differ from eager at floating-point rounding level (fused kernels and
    # the compiled dropout RNG), so this is a per-run choice, recorded in the log.
    _tc = config.get('training', {}).get('compile', {}) or {}
    if _tc.get('enable', False) and not inference_mode:
        _tc_modules = list(_tc.get('modules', ['decoder', 'predictor']))
        _tc_mode = _tc.get('mode', 'default')
        _tc_dynamic = _tc.get('dynamic', False)
        # modules whose inputs change size from batch to batch (e.g. the encoder: vertex count with no padding)
        _tc_dyn_modules = set(_tc.get('dynamic_modules', []))
        for _name in _tc_modules:
            if not hasattr(model, _name):
                raise ValueError(f"training.compile.modules: model has no submodule '{_name}'")
            getattr(model, _name).compile(mode=_tc_mode, dynamic=(True if _name in _tc_dyn_modules else _tc_dynamic))
        logger.info(f"[Compile] training torch.compile ACTIVE: modules={_tc_modules} mode={_tc_mode} "
                    f"dynamic={_tc_dynamic} dynamic_modules={sorted(_tc_dyn_modules)}")

    if use_ddp:
        # Convert BatchNorm to SyncBatchNorm for multi-GPU training;
        # otherwise each GPU keeps different BN statistics.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted BatchNorm to SyncBatchNorm for multi-GPU training")

        device_id = ddp_local_rank if args.device == 'cuda' else None
        # training.ddp (optional): static_graph lets DDP record which parameters are used once and skip the
        # per-iteration autograd-graph search that find_unused_parameters=True otherwise performs (valid when
        # every iteration uses the same parameters, as the teacher-forced training step does);
        # gradient_as_bucket_view makes gradients views into the communication buckets (no copy).
        _ddp_cfg = config.get('training', {}).get('ddp', {}) or {}
        model = DDP(
            model.to(args.device),
            device_ids=[device_id] if device_id is not None else None,
            output_device=device_id if device_id is not None else None,
            find_unused_parameters=True,
            static_graph=bool(_ddp_cfg.get('static_graph', False)),
            gradient_as_bucket_view=bool(_ddp_cfg.get('gradient_as_bucket_view', False))
        )
        if _ddp_cfg:
            logger.info(f"[DDP] options: static_graph={bool(_ddp_cfg.get('static_graph', False))} "
                        f"gradient_as_bucket_view={bool(_ddp_cfg.get('gradient_as_bucket_view', False))}")
    # training.rank_seeded_rng: after the (identical, rank-0-broadcast) model initialisation, each rank's torch RNG
    # gets its own seed, so the per-batch rotation augmentation and dropout / stochastic-depth masks are drawn
    # independently per rank; with the shared seed every rank drew the same ones at every iteration
    if use_ddp and bool(config.get('training', {}).get('rank_seeded_rng', False)):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"[Seed] torch RNG re-seeded per rank after model setup: {seed}")
    
    # Create trainer
    # Visualization CLI params
    viz_params = {
        'arrow_scale': float(os.environ.get('ARROW_SCALE', '1.0')),
        'point_size': float(os.environ.get('POINT_SIZE', '0.001')),
        'arrow_thickness': float(os.environ.get('ARROW_THICKNESS', '0.002')),
        'arrow_subsample': int(os.environ.get('ARROW_SUBSAMPLE', '1')),
        'arrow_enable': int(os.environ.get('ARROW_ENABLE', '1')),
    }

    trainer = Trainer(config, model, train_loader, val_loader, device=args.device, viz_params=viz_params)
    
    # Resume from checkpoint if specified
    if checkpoint_path:
        # Load checkpoint to CPU to avoid GPU0 memory spike in multi-GPU resume
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle DDP/non-DDP checkpoint compatibility:
        # DDP models have a "module." prefix in state_dict keys, non-DDP models don't
        state_dict = checkpoint['model_state_dict']
        is_model_ddp = isinstance(model, DDP)

        # The clean-arm switches add no parameters, so a checkpoint loads under either setting
        # without a shape error. Make any difference between the checkpoint's config and the
        # current config explicit (fine-tuning across a switch change is intended; evaluation
        # with mismatched switches is not).
        _ckpt_cfg = checkpoint.get('config')
        if isinstance(_ckpt_cfg, dict):
            def _flat(d, prefix):
                out = {}
                for _k, _v in (d or {}).items():
                    if isinstance(_v, dict):
                        out.update(_flat(_v, f'{prefix}{_k}.'))
                    else:
                        out[f'{prefix}{_k}'] = _v
                return out
            # Every model.* setting is compared, except the ones that do not change the function a checkpoint
            # computes in evaluation: speed paths equal up to rounding, training-only settings, the rollout's
            # numerics-only Kabsch reference (equal in exact arithmetic), and rmsnorm_eps (its own rule below).
            _eval_neutral = ('model.knn_packed', 'model.encoder.packed', 'model.encoder.packed_pooling',
                             'model.kabsch_reference', 'model.intersection_handling.after_kabsch_loss',
                             'model.predictor.rmsnorm_eps', 'model.predictor.fp32_output')
            _numerics_only = ('model.kabsch_reference',)
            # Settings absent from a checkpoint's configuration take these defaults
            _defaults = {'model.decoder.block_attn_res.final_aggregate': False, 'model.kabsch_reference': 'advancing',
                         'model.encoder.subset_small_objects': 'all_below_8'}
            # data.* settings that change what the model sees in evaluation (the object-scale sidecar may be swapped
            # for the target dataset in cross-dataset evaluation: only whether one is used must agree)
            _data_switches = {
                'data.center_scene_by_t0': (lambda c: bool(c.get('data', {}).get('center_scene_by_t0', False))),
                'data.anchor_fps_start': (lambda c: c.get('data', {}).get('anchor_fps_start', 'index0')),
                'data.fps_order': (lambda c: bool(c.get('data', {}).get('fps_order', False))),
                'data.anchor_k': (lambda c: c.get('data', {}).get('anchor_k')),
                'data.object_scale_file (set)': (lambda c: bool(c.get('data', {}).get('object_scale_file'))),
            }
            _defaults['model.predictor.use_rope'] = True
            _fk, _fc = _flat(_ckpt_cfg.get('model'), 'model.'), _flat(config.get('model'), 'model.')

            def _inactive(name, cfg):
                # settings without effect in evaluation: dropout / stochastic depth rates, frozen-BN training mode,
                # the object-object handler's parameters while it is disabled
                leaf = name.rsplit('.', 1)[-1]
                if 'dropout' in leaf or 'drop_path' in leaf or leaf == 'freeze_bn_stats':
                    return True
                return (name.startswith('model.intersection_handling.object.') and name != 'model.intersection_handling.object.enable'
                        and not cfg.get('model.intersection_handling.object.enable', False))
            _mismatch = []
            for _name in sorted(set(_fk) | set(_fc)):
                _a = _fk.get(_name, _defaults.get(_name, '<absent>'))
                _b = _fc.get(_name, _defaults.get(_name, '<absent>'))
                if _a == _b or (_inactive(_name, _fk) and _inactive(_name, _fc)):
                    continue
                if _name.startswith(_eval_neutral) and not _name.startswith(_numerics_only):
                    continue
                logger.warning(f"[Checkpoint] switch mismatch: {_name} checkpoint={_a} -> current={_b}")
                print(f"[Checkpoint] switch mismatch: {_name} checkpoint={_a} -> current={_b}", flush=True)
                if not _name.startswith(_eval_neutral):
                    _mismatch.append(_name)
            for _name, _get in _data_switches.items():
                _a, _b = _get(_ckpt_cfg), _get(config)
                if _a != _b:
                    logger.warning(f"[Checkpoint] switch mismatch: {_name} checkpoint={_a} -> current={_b}")
                    print(f"[Checkpoint] switch mismatch: {_name} checkpoint={_a} -> current={_b}", flush=True)
                    _mismatch.append(_name)
            # The predictor RMSNorm eps pin reproduces the training function; a checkpoint trained with it pinned
            # must be evaluated with it pinned
            _eps = lambda c: c.get('model', {}).get('predictor', {}).get('rmsnorm_eps', 'default') or 'default'
            if _eps(_ckpt_cfg) != 'default' and _eps(config) != _eps(_ckpt_cfg):
                print(f"[Checkpoint] switch mismatch: model.predictor.rmsnorm_eps checkpoint={_eps(_ckpt_cfg)} -> "
                      f"current={_eps(config)}", flush=True)
                _mismatch.append('model.predictor.rmsnorm_eps')
            # Evaluation must run the function the checkpoint was trained as: refuse behaviour-setting mismatches
            # unless a name is listed in inference.allow_switch_mismatch (deliberate knob experiments)
            _allowed = set(config.get('inference', {}).get('allow_switch_mismatch', []) or [])
            _refused = [n for n in _mismatch if n not in _allowed]
            if (args.inference or config.get('inference', {}).get('mode', False)) and _refused:
                raise ValueError(f"[Checkpoint] evaluation config differs from the checkpoint's training config in "
                                 f"{_refused}; pass the checkpoint's own config (BASE_CONFIG) or list the names in "
                                 f"inference.allow_switch_mismatch")
        else:
            print("[Checkpoint] no config stored in checkpoint; clean-arm switches not cross-checked", flush=True)

        # Check if checkpoint is from DDP (has "module." prefix)
        is_checkpoint_ddp = any(k.startswith('module.') for k in state_dict.keys())

        if is_model_ddp and not is_checkpoint_ddp:
            # Loading non-DDP checkpoint into DDP model: add "module." prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[f'module.{k}'] = v
            state_dict = new_state_dict
            logger.info("Converted non-DDP checkpoint to DDP format (added 'module.' prefix)")
        elif not is_model_ddp and is_checkpoint_ddp:
            # Loading DDP checkpoint into non-DDP model: remove "module." prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # Remove 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            logger.info("Converted DDP checkpoint to non-DDP format (removed 'module.' prefix)")

        # ============================================================
        # Adapt state tokens if size changed (e.g., 48 -> 27)
        # Uses 3D interpolation to preserve learned patterns
        # ============================================================
        state_dict = _adapt_state_tokens_if_needed(state_dict, model, logger)
        # Height input channel: zero-pad conv1 before the size-mismatch filter (exact warm start).
        state_dict = _adapt_state_dict_for_height_input(model, state_dict)

        model_state = model.state_dict()

        # Filter out size-mismatched keys before loading (these layers are re-initialized)
        keys_to_remove = []
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                logger.warning(f"[Checkpoint] Size mismatch for '{k}': checkpoint {v.shape} vs model {model_state[k].shape}. "
                              f"Will SKIP (re-initialize from scratch).")
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del state_dict[k]
        if keys_to_remove:
            logger.warning(f"[Checkpoint] Removed {len(keys_to_remove)} size-mismatched keys. These layers will be randomly initialized.")
        model.load_state_dict(state_dict, strict=True)

        if args.compile:
            if inference_mode:
                for name in ('shared_obj_encoder', 'decoder', 'predictor'):
                    if hasattr(model, name):
                        setattr(model, name, torch.compile(getattr(model, name),
                                                           mode='max-autotune-no-cudagraphs',
                                                           dynamic=False))
                logger.info("torch.compile enabled for encoder, decoder and predictor")
            else:
                logger.warning("--compile is supported for inference only; ignoring it for training")

        # Restore optimizer/scheduler state.
        # A weights-only checkpoint (no optimizer state) starts the optimizer fresh.
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')

        if optimizer_state is None:
            logger.info("Checkpoint contains model weights only; optimizer starts fresh")
        elif trainer.optimizer is not None:
            try:
                trainer.optimizer.load_state_dict(optimizer_state)
            except (ValueError, RuntimeError) as e:
                # A different parameter-group layout (e.g. training.optimizer.decay_matrices_only) cannot resume the
                # optimizer, and restoring the scheduler and epoch without it would mis-assign the group learning rates
                raise ValueError(f"[Checkpoint] optimizer state does not match the configured optimizer ({e}); resume "
                                 f"with the checkpoint's own optimizer settings, or start from its weights with a "
                                 f"weights-only checkpoint file") from e
            # The loaded param groups carry the eps they were saved with; the config decides (default 1e-8).
            _cfg_eps = float(config['training'].get('optimizer', {}).get('eps', 1e-8))
            for _g in trainer.optimizer.param_groups:
                if _g.get('eps') != _cfg_eps:
                    logger.info(f"[Optimizer] eps {_g.get('eps')} from the checkpoint replaced by the configured {_cfg_eps:g}")
                    _g['eps'] = _cfg_eps
        else:
            logger.info("Inference-only mode: skipping optimizer state loading")

        scheduler_loaded = False
        if scheduler_state is not None and trainer.scheduler is not None:
            try:
                trainer.scheduler.load_state_dict(scheduler_state)
                scheduler_loaded = True
                logger.info("Loaded scheduler state from checkpoint")
            except (KeyError, TypeError) as e:
                logger.warning(f"Could not load scheduler state (likely different scheduler type): {e}")
                logger.info("Scheduler will start fresh with current configuration")
        elif trainer.scheduler is None:
            logger.info("Inference-only mode: skipping scheduler state loading")

        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        ckpt_epoch = checkpoint.get('epoch', None)
        if optimizer_state is not None and ckpt_epoch is not None and not inference_mode:
            # A resumed run writes into a new directory: carry the source run's best.pth (and its metric) over, so the
            # new directory always holds the best checkpoint of the whole trajectory, also when the source run was
            # interrupted between publishing best.pth and latest.pth (best then holds the lower metric).
            _src_best = Path(checkpoint_path).parent / 'best.pth'
            if _src_best.exists():
                _b = torch.load(_src_best, map_location='cpu', weights_only=False)
                _b_metric = float(_b.get('best_val_loss', float('inf')))
                if _b_metric < trainer.best_val_loss:
                    logger.info(f"[Resume] {_src_best} (epoch {_b.get('epoch')}) holds a lower best metric {_b_metric:.6f} "
                                f"than the resumed checkpoint ({trainer.best_val_loss:.6f}); using it")
                    trainer.best_val_loss = _b_metric
                if trainer.is_main:
                    Trainer._atomic_save(_b, trainer.checkpoint_dir / 'best.pth')
                    logger.info(f"[Resume] best checkpoint of the source run copied to {trainer.checkpoint_dir / 'best.pth'}")
                del _b
        if optimizer_state is not None and ckpt_epoch is not None and not inference_mode:
            # Full training checkpoint (weights + optimizer + epoch): resume, i.e. continue the
            # epoch count where the run stopped. A weights-only checkpoint instead starts a new
            # schedule at epoch 0 (fine-tuning / warm start).
            trainer.start_epoch = int(ckpt_epoch) + 1
            if scheduler_loaded and not checkpoint.get('scheduler_stepped', False):
                # A checkpoint without 'scheduler_stepped' stored the scheduler state from before the
                # end-of-epoch step; advance once so the LR matches a continuous run.
                trainer.scheduler.step()
                logger.info("Checkpoint without scheduler_stepped: advanced the LR scheduler by one epoch to match a continuous run")
            logger.info(f"Resuming training after completed epoch {ckpt_epoch} (next epoch: {trainer.start_epoch})")
        else:
            trainer.start_epoch = 0
            if ckpt_epoch is not None and not inference_mode:
                logger.info(f"Loaded weights from epoch {ckpt_epoch}; training restarts at epoch 0 with a fresh schedule")
        trainer.epoch = trainer.start_epoch
        if getattr(trainer, 'ema', None) is not None:
            if trainer.start_epoch > 0 and 'ema_state_dict' in checkpoint:
                trainer.ema.load_state_dict(checkpoint['ema_state_dict'], checkpoint.get('ema_num_updates'))
                logger.info(f"[EMA] restored averaged weights from the checkpoint (updates={trainer.ema.num_updates})")
            else:
                trainer.ema.reset()
                logger.info("[EMA] average restarted from the loaded weights")
    
    # inference_mode already determined earlier (before dataloader creation)
    if inference_mode:
        # ============================================================
        # INFERENCE MODE: Only generate videos and compute metrics
        # ============================================================
        logger.info("=" * 80)
        logger.info("INFERENCE MODE: Generating videos and computing metrics")
        logger.info("=" * 80)

        if not checkpoint_path:
            raise ValueError("Inference mode requires --checkpoint argument!")

        try:
            trainer.run_inference()
        except KeyboardInterrupt:
            logger.info("Inference interrupted by user (Ctrl+C)")
            raise
        except Exception as e:
            logger.error(f"Inference failed with error: {e}")
            raise
        finally:
            if use_ddp:
                try:
                    dist.destroy_process_group()
                    logger.info("DDP process group destroyed successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to destroy DDP process group: {cleanup_error}")
    else:
        # ============================================================
        # TRAINING MODE: Normal training loop
        # ============================================================
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user (Ctrl+C)")
            raise
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Always clean up the DDP process group, even on error
            if use_ddp:
                try:
                    dist.destroy_process_group()
                    logger.info("DDP process group destroyed successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to destroy DDP process group: {cleanup_error}")


if __name__ == '__main__':
    main()
