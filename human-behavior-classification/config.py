from pathlib import Path
import torch

class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Paths to your pre-trained models and data ---
    MODEL_A_MODELS_DIR = Path("/kaggle/input/cmi-models-public/pytorch/train_fold_model05_tof16_raw/1/")
    MODEL_A_UNIVERSE_PATH = Path("/kaggle/input/cmi-precompute/pytorch/all/1/tof-1_raw.csv")
    MODEL_B_DIR = Path("/kaggle/input/gaupardatamodel/final_artifacts/")
    
    # --- Ensemble Weights ---
    WEIGHT_A = 0.70
    WEIGHT_B = 0.30

    # --- Dataset Config for Model A ---
    DATASET_CONFIG_A = {
        "tof_raw": True, 
        "tof_mode": 16, 
        "fbfill": {"imu": True, "thm": True, "tof": True}, 
        "nan_ratio": {"imu":0, "thm":0, "tof":0}, 
        "percent": 95
    }

    # --- Model A Architecture Arguments ---
    MODEL_ARGS_A = {
        "feat_dim": 500, "imu1_channels": 219, "imu1_dropout": 0.294, 
        "imu2_dropout": 0.269, "imu1_layers": 0, "imu2_layers": 0, 
        "thm1_channels": 82, "thm1_dropout": 0.264, "thm2_dropout": 0.302, 
        "tof1_channels": 82, "tof1_dropout": 0.264, "tof2_dropout": 0.302, 
        "bert_layers": 8, "bert_heads": 10, "cls1_channels": 937, 
        "cls2_channels": 303, "cls1_dropout": 0.228, "cls2_dropout": 0.225
    }