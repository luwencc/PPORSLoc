#!/usr/bin/env python3
"""CNN-LSTM-SE dual-branch network for RP classification and xy regression."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .config import Config

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(in_channels, in_channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv1d(in_channels // reduction, in_channels, 1), nn.Sigmoid())
        self.scale = 1.2

    def forward(self, x):
        se_out = self.se(x)
        return x * (se_out * self.scale)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.layernorm = nn.LayerNorm(out_channels)
        self.downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.layernorm(out.transpose(1, 2)).transpose(1, 2)
        return self.relu(out)
drop_0 = 0.15
drop_1 = 0.2
drop_2 = 0.15
drop_3 = 0.2
drop_4 = 0.25
drop_5 = 0.2

class MixedModel(nn.Module):

    def __init__(self, num_classes: Optional[int]=None):
        super().__init__()
        num_classes = len(Config.RP_LIST) if num_classes is None else num_classes
        in_ch = int(Config.NUM_PHASES)
        self.conv0 = nn.Conv1d(in_ch, 32, 3, padding=1)
        self.bn0 = nn.BatchNorm1d(32)
        self.dropout0 = nn.Dropout(drop_0)
        self.res1 = ResidualBlock(32, 64, kernel_size=5, padding=2, dropout=drop_1)
        self.lstm = nn.LSTM(64, 64, batch_first=True, num_layers=2, bidirectional=True, dropout=drop_2)
        self.dropout2 = nn.Dropout(drop_2)
        self.lstm_fc = nn.Linear(64 * 2, 128)
        self.conv2 = nn.Conv1d(in_ch, 64, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(drop_3)
        self.se1 = SqueezeExcitation(64, reduction=16)
        self.res2 = ResidualBlock(64, 128, kernel_size=5, padding=2, dropout=drop_4)
        self.se2 = SqueezeExcitation(128, reduction=8)
        self.conv4 = nn.Conv1d(128, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(drop_5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128 + 64, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc_dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.xy_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]=None):
        B, _, T = x.shape
        device = x.device
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device='cpu')
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
        lengths = torch.clamp(lengths, min=1, max=T)
        x_0 = x
        x0 = self.relu(self.bn0(self.conv0(x_0)))
        x0 = self.dropout0(x0)
        x0 = self.res1(x0)
        x0_t = x0.permute(0, 2, 1).contiguous()
        packed = pack_padded_sequence(x0_t, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out_pad, _ = pad_packed_sequence(out_packed, batch_first=True)
        lens_dev = lengths.to(device)
        idx = (lens_dev - 1).clamp(min=0)
        ar = torch.arange(B, device=device)
        x1 = out_pad[ar, idx]
        x1 = self.dropout2(x1)
        x1 = self.lstm_fc(x1)
        x2 = self.relu(self.bn2(self.conv2(x_0)))
        x2 = self.dropout3(x2)
        x2 = self.se1(x2)
        x2 = self.res2(x2)
        x2 = self.se2(x2)
        x2 = self.relu(self.bn4(self.conv4(x2)))
        x2 = self.dropout5(x2)
        x2 = self.gap(x2).squeeze(-1)
        out = torch.cat([x1, x2], dim=-1)
        out = self.relu(self.fc_bn(self.fc1(out)))
        out = self.fc_dropout(out)
        feat = self.relu(self.fc2(out))
        logits = self.fc3(feat)
        pred_xy = self.xy_head(feat)
        return (logits, pred_xy, None)
