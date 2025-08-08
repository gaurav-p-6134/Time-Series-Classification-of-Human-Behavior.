import pandas as pd
import polars as pl
import numpy as np
import pywt

DEMO_FEATS_B = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='db6', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    if len(coeff[-level]) == 0:
        return x
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode='per')
    return ret[:len(x)]

def create_features_b(data_pd: pd.DataFrame):
    if len(data_pd) > 1:
        for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']:
            data_pd[col] = denoise(data_pd[col].values)
    
    data_pl = pl.from_pandas(data_pd)
    all_cols = [c for c in data_pl.columns if c not in ['sequence_id', 'subject']]
    data_pl = data_pl.with_columns([pl.col(c).forward_fill().over("sequence_id") for c in all_cols]).with_columns([pl.col(c).backward_fill().over("sequence_id") for c in all_cols])
    
    acc_cols = [c for c in data_pl.columns if 'acc_' in c]
    rot_cols = [c for c in data_pl.columns if 'rot_' in c]
    thm_cols = [c for c in data_pl.columns if 'thm_' in c]
    tof_cols = [c for c in data_pl.columns if 'tof_' in c]
    
    agg_exprs = []
    for col in acc_cols + rot_cols + thm_cols + tof_cols:
        agg_exprs.extend([
            pl.mean(col).alias(f"{col}_mean"), pl.std(col).alias(f"{col}_std"),
            pl.min(col).alias(f"{col}_min"), pl.max(col).alias(f"{col}_max"),
            pl.median(col).alias(f"{col}_median"), pl.col(col).skew().alias(f"{col}_skew")
        ])
        
    features_df = data_pl.group_by("sequence_id").agg(agg_exprs)
    static_df = data_pl.group_by("sequence_id").first()
    final_df = features_df.join(static_df[['sequence_id'] + DEMO_FEATS_B], on='sequence_id', how='left')
    
    return final_df.to_pandas().set_index('sequence_id').fillna(0)