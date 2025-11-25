from torch import nn
from architectures.shared_model import SharedModel

def create_model(model_type: str, num_feat: int, num_vars: int) -> nn.Module:
    """
    Create a model based on the specified configuration.
    
    Args:
        model_type: Type of model to create: 'linear' or 'shared'. 'linear' refers to a large num_feats -> num_vars
        model that maps all input features to all cost coefficients. 'shared' refers to a shared num_features x 1 model
        that is and invoked separately for each cost coefficient (num_vars times in total).
        num_feat: Number of input features
        num_vars: Number of output variables
        
    Returns:
        nn.Module: The created model
    """

    if model_type == "linear":
        return nn.Linear(num_feat, num_vars)

    elif model_type == "shared":
        return SharedModel(num_feat, 1)

    else:
        raise ValueError(f"Unknown model type: {model_type}") 