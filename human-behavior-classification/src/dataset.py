import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

# ===================================================================
# HELPER FUNCTIONS (Physics-based Feature Engineering)
# ===================================================================

def remove_gravity_from_acc(acc_data, rot_data):
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values if isinstance(acc_data, pd.DataFrame) else acc_data
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values if isinstance(rot_data, pd.DataFrame) else rot_data
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    for i in range(len(acc_values)):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values if isinstance(rot_data, pd.DataFrame) else rot_data
    angular_vel = np.zeros((len(quat_values), 3))
    for i in range(len(quat_values) - 1):
        q_t, q_t_plus_dt = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        try:
            delta_rot = R.from_quat(q_t).inv() * R.from_quat(q_t_plus_dt)
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    return angular_vel

def calculate_angular_distance(rot_data):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values if isinstance(rot_data, pd.DataFrame) else rot_data
    angular_dist = np.zeros(len(quat_values))
    for i in range(len(quat_values) - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            angle = np.linalg.norm((R.from_quat(q1).inv() * R.from_quat(q2)).as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
    return angular_dist

# ===================================================================
# PYTORCH DATASET CLASS
# ===================================================================

class CMIFeDataset(Dataset):
    def __init__(self, data_path, config):
        self.config=config
        self.init_feature_names(data_path)
        df=self.generate_features(pd.read_csv(data_path,usecols=set(self.base_cols+self.feature_cols)))
        self.generate_dataset(df)

    def init_feature_names(self, data_path):
        self.imu_engineered_features=['acc_mag','rot_angle','acc_mag_jerk','rot_angle_vel','linear_acc_mag','linear_acc_mag_jerk','angular_vel_x','angular_vel_y','angular_vel_z','angular_distance']
        self.tof_mode=self.config.get("tof_mode","stats")
        self.tof_region_stats=['mean','std','min','max']
        self.tof_cols=self.generate_tof_feature_names()
        columns=pd.read_csv(data_path,nrows=0).columns.tolist()
        imu_cols_base=['linear_acc_x','linear_acc_y','linear_acc_z']
        imu_cols_base.extend([c for c in columns if c.startswith('rot_') and c not in ['rot_angle','rot_angle_vel']])
        self.imu_cols=list(dict.fromkeys(imu_cols_base+self.imu_engineered_features))
        self.thm_cols=[c for c in columns if c.startswith('thm_')]
        self.feature_cols=self.imu_cols+self.thm_cols+self.tof_cols
        self.imu_dim=len(self.imu_cols)
        self.thm_dim=len(self.thm_cols)
        self.tof_dim=len(self.tof_cols)
        self.base_cols=['acc_x','acc_y','acc_z','rot_x','rot_y','rot_z','rot_w','sequence_id','subject','sequence_type','gesture','orientation']+[c for c in columns if c.startswith('thm_')]+[f"tof_{i}_v{p}" for i in range(1,6) for p in range(64)]
        self.fold_cols=['subject','sequence_type','gesture','orientation']

    def generate_tof_feature_names(self):
        features=[]
        if self.config.get("tof_raw",False):
            for i in range(1,6): features.extend([f"tof_{i}_v{p}" for p in range(64)])
        for i in range(1,6):
            if self.tof_mode!=0:
                for s in self.tof_region_stats: features.append(f'tof_{i}_{s}')
                if self.tof_mode>1:
                    for r in range(self.tof_mode):
                        for s in self.tof_region_stats: features.append(f'tof{self.tof_mode}_{i}_region_{r}_{s}')
                if self.tof_mode==-1:
                    for m in [2,4,8,16,32]:
                        for r in range(m):
                            for s in self.tof_region_stats: features.append(f'tof{m}_{i}_region_{r}_{s}')
        return features

    def compute_features(self, df):
        df['acc_mag']=np.sqrt(df['acc_x']**2+df['acc_y']**2+df['acc_z']**2)
        df['rot_angle']=2*np.arccos(df['rot_w'].clip(-1,1))
        df['acc_mag_jerk']=df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel']=df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
        la_list=[]
        for _,g in df.groupby('sequence_id'):
            la_list.append(pd.DataFrame(remove_gravity_from_acc(g[['acc_x','acc_y','acc_z']],g[['rot_x','rot_y','rot_z','rot_w']]),columns=['linear_acc_x','linear_acc_y','linear_acc_z'],index=g.index))
        df=pd.concat([df,pd.concat(la_list)],axis=1)
        df['linear_acc_mag']=np.sqrt(df['linear_acc_x']**2+df['linear_acc_y']**2+df['linear_acc_z']**2)
        df['linear_acc_mag_jerk']=df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
        av_list=[]
        for _,g in df.groupby('sequence_id'):
            av_list.append(pd.DataFrame(calculate_angular_velocity_from_quat(g[['rot_x','rot_y','rot_z','rot_w']]),columns=['angular_vel_x','angular_vel_y','angular_vel_z'],index=g.index))
        df=pd.concat([df,pd.concat(av_list)],axis=1)
        ad_list=[]
        for _,g in df.groupby('sequence_id'):
            ad_list.append(pd.DataFrame(calculate_angular_distance(g[['rot_x','rot_y','rot_z','rot_w']]),columns=['angular_distance'],index=g.index))
        df=pd.concat([df,pd.concat(ad_list)],axis=1)
        if self.tof_mode!=0:
            new_cols={}
            for i in range(1,6):
                pc=[f"tof_{i}_v{p}" for p in range(64)]
                tof_data=df[pc].replace(-1,np.nan)
                new_cols.update({f'tof_{i}_mean':tof_data.mean(axis=1),f'tof_{i}_std':tof_data.std(axis=1),f'tof_{i}_min':tof_data.min(axis=1),f'tof_{i}_max':tof_data.max(axis=1)})
                if self.tof_mode>1:
                    rs=64//self.tof_mode
                    for r in range(self.tof_mode):
                        rd=tof_data.iloc[:,r*rs:(r+1)*rs]
                        new_cols.update({f'tof{self.tof_mode}_{i}_region_{r}_mean':rd.mean(axis=1),f'tof{self.tof_mode}_{i}_region_{r}_std':rd.std(axis=1),f'tof{self.tof_mode}_{i}_region_{r}_min':rd.min(axis=1),f'tof{self.tof_mode}_{i}_region_{r}_max':rd.max(axis=1)})
                if self.tof_mode==-1:
                    for m in [2,4,8,16,32]:
                        rs=64//m
                        for r in range(m):
                            rd=tof_data.iloc[:,r*rs:(r+1)*rs]
                            new_cols.update({f'tof{m}_{i}_region_{r}_mean':rd.mean(axis=1),f'tof{m}_{i}_region_{r}_std':rd.std(axis=1),f'tof{m}_{i}_region_{r}_min':rd.min(axis=1),f'tof{m}_{i}_region_{r}_max':rd.max(axis=1)})
            df=pd.concat([df,pd.DataFrame(new_cols)],axis=1)
        return df

    def generate_features(self,df):
        self.le=LabelEncoder()
        self.class_num=len(self.le.fit(df['gesture']).classes_)
        df['gesture_int']=self.le.transform(df['gesture'])
        if all(c in df.columns for c in self.imu_engineered_features) and all(c in df.columns for c in self.tof_cols):
            print("Have precomputed, skip compute.")
        else:
            print("Not precomputed, do compute.")
            df=self.compute_features(df)
        return df

    def scale(self,data_unscaled):
        scaler=self.config.get("scaler_function",StandardScaler()).fit(np.concatenate(data_unscaled,axis=0))
        return [scaler.transform(x) for x in data_unscaled],scaler

    def pad(self,data_scaled,cols):
        pad_data=np.zeros((len(data_scaled),self.pad_len,len(cols)),dtype='float32')
        for i,s in enumerate(data_scaled):
            sl=min(len(s),self.pad_len)
            pad_data[i,:sl]=s[:sl]
        return pad_data

    def get_nan_value(self,data,ratio):
        return -data.max().max()*ratio if not data.empty else 0

    def generate_dataset(self,df):
        sg=df.groupby('sequence_id')
        iu,tu,tou,cs,ls=[],[],[],[],[]
        self.imu_nan_value=self.get_nan_value(df[self.imu_cols],self.config["nan_ratio"]["imu"])
        self.thm_nan_value=self.get_nan_value(df[self.thm_cols],self.config["nan_ratio"]["thm"])
        self.tof_nan_value=self.get_nan_value(df[self.tof_cols],self.config["nan_ratio"]["tof"])
        self.fold_feats=defaultdict(list)
        for _,sdf in sg:
            id=sdf[self.imu_cols]
            if self.config["fbfill"]["imu"]: id=id.ffill().bfill()
            iu.append(id.fillna(self.imu_nan_value).values.astype('float32'))
            td=sdf[self.thm_cols]
            if self.config["fbfill"]["thm"]: td=td.ffill().bfill()
            tu.append(td.fillna(self.thm_nan_value).values.astype('float32'))
            tod=sdf[self.tof_cols]
            if self.config["fbfill"]["tof"]: tod=tod.ffill().bfill()
            tou.append(tod.fillna(self.tof_nan_value).values.astype('float32'))
            cs.append(sdf['gesture_int'].iloc[0])
            ls.append(len(id))
            for c in self.fold_cols: self.fold_feats[c].append(sdf[c].iloc[0])
        self.dataset_indices=cs
        self.pad_len=int(np.percentile(ls,self.config.get("percent",95)))
        if self.config.get("one_scale",True):
            xu=[np.concatenate([i,t,o],axis=1) for i,t,o in zip(iu,tu,tou)]
            xs,self.x_scaler=self.scale(xu)
            x=self.pad(xs,self.imu_cols+self.thm_cols+self.tof_cols)
            self.imu=x[...,:self.imu_dim]
            self.thm=x[...,self.imu_dim:self.imu_dim+self.thm_dim]
            self.tof=x[...,self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        else:
            is_,self.imu_scaler=self.scale(iu)
            ts,self.thm_scaler=self.scale(tu)
            tos,self.tof_scaler=self.scale(tou)
            self.imu=self.pad(is_,self.imu_cols)
            self.thm=self.pad(ts,self.thm_cols)
            self.tof=self.pad(tos,self.tof_cols)
        self.precompute_scaled_nan_values()
        self.class_=F.one_hot(torch.from_numpy(np.array(cs)).long(),num_classes=len(self.le.classes_)).float().numpy()

    def precompute_scaled_nan_values(self):
        ddf=pd.DataFrame(np.array([[self.imu_nan_value]*len(self.imu_cols)+[self.thm_nan_value]*len(self.thm_cols)+[self.tof_nan_value]*len(self.tof_cols)]),columns=self.imu_cols+self.thm_cols+self.tof_cols)
        if self.config.get("one_scale",True):
            s=self.x_scaler.transform(ddf)
            self.imu_scaled_nan=s[0,:self.imu_dim].mean()
            self.thm_scaled_nan=s[0,self.imu_dim:self.imu_dim+self.thm_dim].mean()
            self.tof_scaled_nan=s[0,self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim].mean()
        else:
            self.imu_scaled_nan=self.imu_scaler.transform(ddf[self.imu_cols])[0].mean()
            self.thm_scaled_nan=self.thm_scaler.transform(ddf[self.thm_cols])[0].mean()
            self.tof_scaled_nan=self.tof_scaler.transform(ddf[self.tof_cols])[0].mean()

    def get_scaled_nan_tensors(self,i,t,o):
        return torch.full(i.shape,self.imu_scaled_nan,device=i.device),torch.full(t.shape,self.thm_scaled_nan,device=t.device),torch.full(o.shape,self.tof_scaled_nan,device=o.device)

    def inference_process(self,s):
        ds=s.to_pandas().copy()
        if not all(c in ds for c in self.imu_engineered_features):
            ds['acc_mag']=np.sqrt(ds['acc_x']**2+ds['acc_y']**2+ds['acc_z']**2)
            ds['rot_angle']=2*np.arccos(ds['rot_w'].clip(-1,1))
            ds['acc_mag_jerk']=ds['acc_mag'].diff().fillna(0)
            ds['rot_angle_vel']=ds['rot_angle'].diff().fillna(0)
            if all(c in ds for c in ['acc_x','acc_y','acc_z','rot_x','rot_y','rot_z','rot_w']):
                la=remove_gravity_from_acc(ds[['acc_x','acc_y','acc_z']],ds[['rot_x','rot_y','rot_z','rot_w']])
                ds[['linear_acc_x','linear_acc_y','linear_acc_z']]=la
            else:
                ds['linear_acc_x']=ds.get('acc_x',0)
                ds['linear_acc_y']=ds.get('acc_y',0)
                ds['linear_acc_z']=ds.get('acc_z',0)
            ds['linear_acc_mag']=np.sqrt(ds['linear_acc_x']**2+ds['linear_acc_y']**2+ds['linear_acc_z']**2)
            ds['linear_acc_mag_jerk']=ds['linear_acc_mag'].diff().fillna(0)
            if all(c in ds for c in ['rot_x','rot_y','rot_z','rot_w']):
                av=calculate_angular_velocity_from_quat(ds[['rot_x','rot_y','rot_z','rot_w']])
                ds[['angular_vel_x','angular_vel_y','angular_vel_z']]=av
            else:
                ds[['angular_vel_x','angular_vel_y','angular_vel_z']]=0
            if all(c in ds for c in ['rot_x','rot_y','rot_z','rot_w']):
                ds['angular_distance']=calculate_angular_distance(ds[['rot_x','rot_y','rot_z','rot_w']])
            else:
                ds['angular_distance']=0
        if self.tof_mode!=0:
            nc={}
            for i in range(1,6):
                pc=[f"tof_{i}_v{p}" for p in range(64)]
                td=ds[pc].replace(-1,np.nan)
                nc.update({f'tof_{i}_mean':td.mean(axis=1),f'tof_{i}_std':td.std(axis=1),f'tof_{i}_min':td.min(axis=1),f'tof_{i}_max':td.max(axis=1)})
                if self.tof_mode>1:
                    rs=64//self.tof_mode
                    for r in range(self.tof_mode):
                        rd=td.iloc[:,r*rs:(r+1)*rs]
                        nc.update({f'tof{self.tof_mode}_{i}_region_{r}_mean':rd.mean(axis=1),f'tof{self.tof_mode}_{i}_region_{r}_std':rd.std(axis=1),f'tof{self.tof_mode}_{i}_region_{r}_min':rd.min(axis=1),f'tof{self.tof_mode}_{i}_region_{r}_max':rd.max(axis=1)})
                if self.tof_mode==-1:
                    for m in [2,4,8,16,32]:
                        rs=64//m
                        for r in range(m):
                            rd=td.iloc[:,r*rs:(r+1)*rs]
                            nc.update({f'tof{m}_{i}_region_{r}_mean':rd.mean(axis=1),f'tof{m}_{i}_region_{r}_std':rd.std(axis=1),f'tof{m}_{i}_region_{r}_min':rd.min(axis=1),f'tof{m}_{i}_region_{r}_max':rd.max(axis=1)})
            ds=pd.concat([ds,pd.DataFrame(nc)],axis=1)
        iu=ds[self.imu_cols]
        if self.config["fbfill"]["imu"]: iu=iu.ffill().bfill()
        iu=iu.fillna(self.imu_nan_value).values.astype('float32')
        tu=ds[self.thm_cols]
        if self.config["fbfill"]["thm"]: tu=tu.ffill().bfill()
        tu=tu.fillna(self.thm_nan_value).values.astype('float32')
        tou=ds[self.tof_cols]
        if self.config["fbfill"]["tof"]: tou=tou.ffill().bfill()
        tou=tou.fillna(self.tof_nan_value).values.astype('float32')
        if self.config.get("one_scale",True):
            xu=np.concatenate([iu,tu,tou],axis=1)
            xs=self.x_scaler.transform(xu)
            is_=xs[...,:self.imu_dim]
            ts=xs[...,self.imu_dim:self.imu_dim+self.thm_dim]
            tos=xs[...,self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        else:
            is_=self.imu_scaler.transform(iu)
            ts=self.thm_scaler.transform(tu)
            tos=self.tof_scaler.transform(tou)
        c=np.concatenate([is_,ts,tos],axis=1)
        p=np.zeros((self.pad_len,c.shape[1]),dtype='float32')
        sl=min(c.shape[0],self.pad_len)
        p[:sl]=c[:sl]
        i=p[...,:self.imu_dim]
        t=p[...,self.imu_dim:self.imu_dim+self.thm_dim]
        o=p[...,self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        return torch.from_numpy(i).float().unsqueeze(0),torch.from_numpy(t).float().unsqueeze(0),torch.from_numpy(o).float().unsqueeze(0)