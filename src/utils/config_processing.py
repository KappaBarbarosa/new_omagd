"""
配置处理工具 - 确保graph reconstruction配置能正确读取和验证
"""
from types import SimpleNamespace
from collections.abc import Mapping


def ensure_graph_reconstruction_config(config_dict):
    """
    确保graph reconstruction配置存在并设置默认值
    
    Args:
        config_dict: 配置字典
    
    Returns:
        更新后的配置字典
    """
    # 基础配置默认值
    defaults = {
        'use_graph_reconstruction': False,
        'graph_reconstruction_weight': 0.1,
        'graph_training_stage': 'stage1',
        'stage2_model_type': 'masked_predictor',
    }
    
    # 设置基础配置默认值
    for key, default_value in defaults.items():
        if key not in config_dict:
            config_dict[key] = default_value
    
    # 确保tokenizer_config存在
    if 'tokenizer_config' not in config_dict:
        config_dict['tokenizer_config'] = {}
    
    tokenizer_defaults = {
        'hid': 64,
        'code_dim': 32,
        'n_codes': 1024,
        'commitment_weight': 1.0,
        'use_cosine': False,
        'decay': 0.9,
        'eps': 1e-5,
        'revive_threshold': 0.01,
        'encoder_type': 'mlp',
        'encoder_hid': 64,
        'encoder_layers': 2,
        'dropout': 0.1,
        'decoder_type': 'mlp',
        'decoder_hid': 64,
        'decoder_layers': 2,
    }
    
    # 设置tokenizer配置默认值
    for key, default_value in tokenizer_defaults.items():
        if key not in config_dict['tokenizer_config']:
            config_dict['tokenizer_config'][key] = default_value
    
    # 确保mask_predictor_config存在
    if 'mask_predictor_config' not in config_dict:
        config_dict['mask_predictor_config'] = {}
    
    mask_predictor_defaults = {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'num_node_types': 3,
        'max_nodes': 32,
        'mask_ratio': 0.15,
    }
    
    # 设置mask_predictor配置默认值
    for key, default_value in mask_predictor_defaults.items():
        if key not in config_dict['mask_predictor_config']:
            config_dict['mask_predictor_config'][key] = default_value
    
    return config_dict


def get_nested_config(args, key, default=None):
    """
    安全地从args中读取嵌套配置
    
    Args:
        args: SimpleNamespace或字典
        key: 配置键，支持点号分隔的嵌套键（如 'tokenizer_config.hid'）
        default: 默认值
    
    Returns:
        配置值
    """
    if isinstance(args, dict):
        # 如果是字典，支持嵌套键
        keys = key.split('.')
        value = args
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    else:
        # 如果是SimpleNamespace
        if '.' in key:
            # 处理嵌套键
            keys = key.split('.')
            value = args
            for k in keys:
                value = getattr(value, k, None)
                if value is None:
                    return default
            return value
        else:
            return getattr(args, key, default)


def get_config_dict(args, key, default=None):
    """
    获取嵌套配置字典
    
    Args:
        args: SimpleNamespace或字典
        key: 配置键（如 'tokenizer_config'）
        default: 默认值（通常是空字典）
    
    Returns:
        配置字典
    """
    if default is None:
        default = {}
    
    config = get_nested_config(args, key, default)
    
    # 确保返回的是字典
    if isinstance(config, dict):
        return config
    elif isinstance(config, SimpleNamespace):
        return vars(config)
    else:
        return default


def validate_graph_reconstruction_config(config_dict):
    """
    验证graph reconstruction配置的有效性
    
    Args:
        config_dict: 配置字典
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # 检查基础配置
    if config_dict.get('use_graph_reconstruction', False):
        # 如果启用了graph reconstruction，检查必需配置
        stage = config_dict.get('graph_training_stage', 'stage1')
        if stage not in ['stage1', 'stage2']:
            errors.append(f"Invalid graph_training_stage: {stage}. Must be 'stage1' or 'stage2'")
        
        stage2_type = config_dict.get('stage2_model_type', 'masked_predictor')
        if stage2_type not in ['diffusion', 'masked_predictor']:
            errors.append(f"Invalid stage2_model_type: {stage2_type}. Must be 'diffusion' or 'masked_predictor'")
        
        # 检查tokenizer配置
        if 'tokenizer_config' not in config_dict:
            errors.append("tokenizer_config is missing")
        else:
            tokenizer_config = config_dict['tokenizer_config']
            if not isinstance(tokenizer_config, dict):
                errors.append("tokenizer_config must be a dictionary")
            else:
                # 检查必需的tokenizer参数
                required_tokenizer_keys = ['hid', 'code_dim', 'n_codes']
                for key in required_tokenizer_keys:
                    if key not in tokenizer_config:
                        errors.append(f"tokenizer_config.{key} is missing")
        
        # 如果使用masked_predictor，检查其配置
        if stage2_type == 'masked_predictor':
            if 'mask_predictor_config' not in config_dict:
                errors.append("mask_predictor_config is missing")
            else:
                mask_config = config_dict['mask_predictor_config']
                if not isinstance(mask_config, dict):
                    errors.append("mask_predictor_config must be a dictionary")
                else:
                    # 检查必需的mask_predictor参数
                    required_mask_keys = ['d_model', 'nhead', 'num_encoder_layers']
                    for key in required_mask_keys:
                        if key not in mask_config:
                            errors.append(f"mask_predictor_config.{key} is missing")
    
    is_valid = len(errors) == 0
    return is_valid, errors

