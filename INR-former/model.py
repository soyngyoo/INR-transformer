import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class INRformer(nn.Module):
    def __init__(self, num_classes=10, d_model=128):
        super().__init__()
        self.d_model = d_model
        
        # 좌표 임베딩
        self.coord_embedding = nn.Linear(2, d_model)
        
        # 라벨 임베딩
        self.label_embedding = nn.Embedding(num_classes, d_model)
        
        # Self Attention 블록
        self.self_attention = nn.MultiheadAttention(d_model, num_heads=4)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp1 = MLP(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross Attention 블록
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp2 = MLP(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # 최종 픽셀 값 예측
        self.final_mlp = nn.Linear(d_model, 1)
        
    def positional_encoding(self, coords, max_len=128):
        batch_size = coords.shape[0]
        encoding = torch.zeros(batch_size, max_len, device=coords.device)
        
        # 각 좌표에 대한 포지셔널 인코딩
        for i, (x, y) in enumerate(coords):
            # x 좌표에 대한 인코딩 (짝수 차원)
            encoding[i, 0::2] = torch.sin(x / (10000 ** (2 * torch.arange(0, max_len // 2, device=coords.device) / max_len)))
            # y 좌표에 대한 인코딩 (홀수 차원)
            encoding[i, 1::2] = torch.cos(y / (10000 ** (2 * torch.arange(0, max_len // 2, device=coords.device) / max_len)))
            
        return encoding
    
    def forward(self, coords, labels):
        # 1. 좌표 임베딩
        input_emb = self.coord_embedding(coords)  # [batch_size, d_model]
        
        # 2. 공간적 포지셔널 인코딩
        pos_encoding = self.positional_encoding(coords, self.d_model)
        
        # 3. 임베딩 결합
        x = input_emb + pos_encoding  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 4. Self Attention 블록
        sa_out, _ = self.self_attention(x, x, x)
        x = x + sa_out
        x = self.norm1(x)
        mlp_out = self.mlp1(x)
        x = x + mlp_out
        x = self.norm2(x)
        
        # 5. Cross Attention 블록
        label_emb = self.label_embedding(labels).unsqueeze(1)
        
        ca_out, _ = self.cross_attention(
            query=x.transpose(0, 1),
            key=label_emb.transpose(0, 1),
            value=label_emb.transpose(0, 1)
        )
        ca_out = ca_out.transpose(0, 1)
        
        x = x + ca_out
        x = self.norm3(x)
        mlp_out = self.mlp2(x)
        x = x + mlp_out
        x = self.norm4(x)
        
        # 6. 픽셀 값 예측
        pixel_values = torch.sigmoid(self.final_mlp(x.squeeze(1)))
        
        return pixel_values