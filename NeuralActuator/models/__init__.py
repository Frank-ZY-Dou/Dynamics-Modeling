"""
Neural Actuator Models Package.

Provides modular neural network architectures for actuator modeling.
All models follow the same interface and are selected by the model_type config key.

Supported model types:
- mlp: Standard MLP (NeuralActuator)
- lnn: Liquid Neural Network (LNNActuator)
- gru: GRU-based (GRUActuator)
- lstm: LSTM-based (LSTMActuator)
- transformer: Transformer-based (TransformerActuator)

Usage:
    from models import create_model

    model = create_model(
        model_type="lnn",
        hidden_dim=32,
        latent_dim=16,
        dropout_rate=0.1,
        backbone_activation="silu"
    )
"""

from .base import TorqueNet, ForceNet, ConditionNet, gumbel_sigmoid
from .mlp import NeuralActuator
from .lnn import LNNActuator, CfCCell
from .transformer import TransformerActuator
from .gru import GRUActuator, GRUCell
from .lstm import LSTMActuator, LSTMCell

# Model registry
MODEL_REGISTRY = {
    "mlp": NeuralActuator,
    "lnn": LNNActuator,
    "transformer": TransformerActuator,
    "gru": GRUActuator,
    "lstm": LSTMActuator,
}

# Supported model types (for documentation)
SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())


def create_model(model_type: str, **kwargs):
    """Factory function to create actuator models.

    Args:
        model_type: Type of model to create. Options: mlp, lnn, gru, lstm, transformer
        **kwargs: Model-specific parameters:
            - hidden_dim: Hidden layer dimension (default: 32)
            - latent_dim: Latent space dimension (default: 16)
            - dropout_rate: Dropout rate (default: 0.1)
            - backbone_activation: Activation function for LNN (default: "silu")

    Returns:
        Flax nn.Module instance

    Raises:
        ValueError: If model_type is not supported

    Example:
        >>> model = create_model("lnn", hidden_dim=64, backbone_activation="tanh")
    """
    model_type = model_type.lower()

    if model_type not in MODEL_REGISTRY:
        available = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available options: {available}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Filter kwargs to only pass valid parameters for each model
    if model_type == "mlp":
        valid_keys = {"hidden_dim", "latent_dim", "dropout_rate"}
    elif model_type == "lnn":
        valid_keys = {"hidden_dim", "latent_dim", "dropout_rate", "backbone_activation"}
    elif model_type == "transformer":
        valid_keys = {"hidden_dim", "latent_dim", "dropout_rate", "num_heads", "num_layers", "d_ff", "pool_type", "use_gated_attention", "zero_init_head", "n_joints"}
    elif model_type == "gru":
        valid_keys = {"hidden_dim", "latent_dim", "dropout_rate", "num_layers"}
    elif model_type == "lstm":
        valid_keys = {"hidden_dim", "latent_dim", "dropout_rate", "num_layers"}
    else:
        valid_keys = set(kwargs.keys())

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    return model_class(**filtered_kwargs)


def get_model_type_from_config(config: dict) -> str:
    """Extract model type from config dict with backward compatibility.

    Supports both new 'model_type' and legacy 'if_liquid_NN' config keys.

    Args:
        config: Configuration dictionary

    Returns:
        model_type string ("mlp" or "lnn")
    """
    # New config: model_type
    if "model_type" in config:
        return config["model_type"].lower()

    # Legacy config: if_liquid_NN
    if_liquid_NN_raw = config.get("if_liquid_NN", False)
    if isinstance(if_liquid_NN_raw, str):
        if_liquid_NN = if_liquid_NN_raw.lower() == "true"
    else:
        if_liquid_NN = bool(if_liquid_NN_raw)

    return "lnn" if if_liquid_NN else "mlp"


# Export all public symbols
__all__ = [
    # Factory
    "create_model",
    "get_model_type_from_config",
    "MODEL_REGISTRY",
    "SUPPORTED_MODELS",
    # Models
    "NeuralActuator",
    "LNNActuator",
    "TransformerActuator",
    "GRUActuator",
    "LSTMActuator",
    # Components
    "TorqueNet",
    "ForceNet",
    "ConditionNet",
    "CfCCell",
    "GRUCell",
    "LSTMCell",
    "gumbel_sigmoid",
]
