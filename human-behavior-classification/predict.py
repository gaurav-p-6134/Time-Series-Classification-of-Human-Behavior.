import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

# Add Kaggle's evaluation API to the path
sys.path.append('/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/')
import kaggle_evaluation.cmi_inference_server

# Import our modularized code
from config import CFG
from src.dataset import CMIFeDataset
from src.model import ModelA_Architecture
from src.feature_engineering import create_features_b

def load_artifacts():
    """Loads all models and necessary artifacts."""
    print(f"Using device: {CFG.DEVICE}")
    print("--- Loading Artifacts for Model A (BERT) ---")
    dataset_a = CMIFeDataset(CFG.MODEL_A_UNIVERSE_PATH, CFG.DATASET_CONFIG_A)
    model_args_a = CFG.MODEL_ARGS_A
    model_args_a.update({"imu_dim": dataset_a.imu_dim, "thm_dim": dataset_a.thm_dim, "tof_dim": dataset_a.tof_dim, "n_classes": dataset_a.class_num})
    
    models_a = []
    for fold in range(5):
        model_path = CFG.MODEL_A_MODELS_DIR / f"fold{fold}/best_ema.pt"
        model = ModelA_Architecture(**model_args_a).to(CFG.DEVICE)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in torch.load(model_path, map_location=CFG.DEVICE).items()}
        model.load_state_dict(state_dict)
        model.eval()
        models_a.append(model)
        print(f"Loaded Model A from Fold {fold}")

    print("\n--- Loading Artifacts for Model B (LGBM) ---")
    model_b_all = joblib.load(CFG.MODEL_B_DIR / 'model_all_sensors.pkl')
    model_b_imu = joblib.load(CFG.MODEL_B_DIR / 'model_imu_only.pkl')
    le = joblib.load(CFG.MODEL_B_DIR / 'label_encoder.pkl')
    features_b_all = joblib.load(CFG.MODEL_B_DIR / 'features_all.pkl')
    features_b_imu = joblib.load(CFG.MODEL_B_DIR / 'features_imu.pkl')
    print("Model B artifacts loaded.")
    
    return dataset_a, models_a, model_b_all, model_b_imu, le, features_b_all, features_b_imu

def main():
    """Main function to load artifacts and serve the model."""
    dataset_a, models_a, model_b_all, model_b_imu, le, features_b_all, features_b_imu = load_artifacts()

    def to_device(*tensors):
        return [t.to(CFG.DEVICE) for t in tensors]

    def predict(sequence, demographics):
        is_imu_only = sequence['tof_1_v0'].is_null().all()
        
        # --- Model A Prediction ---
        imu_a, thm_a, tof_a = dataset_a.inference_process(sequence)
        imu_a, thm_a, tof_a = to_device(imu_a, thm_a, tof_a)
        if is_imu_only:
            _, thm_a, tof_a = dataset_a.get_scaled_nan_tensors(imu_a, thm_a, tof_a)
        with torch.no_grad(), autocast(device_type=CFG.DEVICE.split(':')[0]):
            all_logits_a = [model(imu_a, thm_a, tof_a) for model in models_a]
            avg_logits_a = torch.mean(torch.stack(all_logits_a), dim=0)
            probs_a = F.softmax(avg_logits_a, dim=-1).cpu().numpy()

        # --- Model B Prediction ---
        data_pd_b = pd.merge(sequence.to_pandas(), demographics.to_pandas(), on='subject', how='left')
        features_df_b = create_features_b(data_pd_b)
        probs_b = model_b_imu.predict_proba(features_df_b[features_b_imu]) if is_imu_only else model_b_all.predict_proba(features_df_b[features_b_all])
            
        # --- Ensemble Predictions ---
        final_probs = CFG.WEIGHT_A * probs_a + CFG.WEIGHT_B * probs_b
        prediction_int = np.argmax(final_probs, axis=1)
        return le.inverse_transform(prediction_int)[0]

    # --- Setup Kaggle Inference Server ---
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # For local testing
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )

if __name__ == "__main__":
    main()