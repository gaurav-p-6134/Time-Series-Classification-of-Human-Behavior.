import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        se = F.relu(self.fc1(se), inplace=True)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)
        return x * se

class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.BatchNorm1d(out_channels))
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)

class ModelA_Architecture(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, n_classes, **kwargs):
        super().__init__()
        self.imu_branch = nn.Sequential(
            self.residual_se_cnn_block(imu_dim, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.thm_branch = nn.Sequential(
            nn.Conv1d(thm_dim, kwargs["thm1_channels"], 3, padding=1, bias=False), nn.BatchNorm1d(kwargs["thm1_channels"]), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True), nn.Dropout(kwargs["thm1_dropout"]),
            nn.Conv1d(kwargs["thm1_channels"], kwargs["feat_dim"], 3, padding=1, bias=False), nn.BatchNorm1d(kwargs["feat_dim"]), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True), nn.Dropout(kwargs["thm2_dropout"])
        )
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, kwargs["tof1_channels"], 3, padding=1, bias=False), nn.BatchNorm1d(kwargs["tof1_channels"]), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True), nn.Dropout(kwargs["tof1_dropout"]),
            nn.Conv1d(kwargs["tof1_channels"], kwargs["feat_dim"], 3, padding=1, bias=False), nn.BatchNorm1d(kwargs["feat_dim"]), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True), nn.Dropout(kwargs["tof2_dropout"])
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, kwargs["feat_dim"]))
        self.bert = BertModel(BertConfig(hidden_size=kwargs["feat_dim"], num_hidden_layers=kwargs["bert_layers"], num_attention_heads=kwargs["bert_heads"], intermediate_size=kwargs["feat_dim"] * 4))
        self.classifier = nn.Sequential(
            nn.Linear(kwargs["feat_dim"], kwargs["cls1_channels"], bias=False), nn.BatchNorm1d(kwargs["cls1_channels"]), nn.ReLU(inplace=True), nn.Dropout(kwargs["cls1_dropout"]),
            nn.Linear(kwargs["cls1_channels"], kwargs["cls2_channels"], bias=False), nn.BatchNorm1d(kwargs["cls2_channels"]), nn.ReLU(inplace=True), nn.Dropout(kwargs["cls2_dropout"]),
            nn.Linear(kwargs["cls2_channels"], n_classes)
        )

    def residual_se_cnn_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3, wd=1e-4):
        return nn.Sequential(*[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)], ResNetSEBlock(in_channels, out_channels, wd=wd), nn.MaxPool1d(pool_size), nn.Dropout(drop))

    def forward(self, imu, thm, tof):
        imu_feat = self.imu_branch(imu.permute(0, 2, 1))
        thm_feat = self.thm_branch(thm.permute(0, 2, 1))
        tof_feat = self.tof_branch(tof.permute(0, 2, 1))
        bert_input = torch.cat([imu_feat, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)
        bert_input = torch.cat([cls_token, bert_input], dim=1)
        outputs = self.bert(inputs_embeds=bert_input)
        pred_cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pred_cls)