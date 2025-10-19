import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ★ 若你的 SpectralGate3D 放在其它路径，请把下面这一行改成对应的 import
from net.modules.ssp_freq_gate import SpectralGate3D


# ****************************************************************************************
# ------------------------------ TFusion Basic blocks  ----------------------------------
# ****************************************************************************************
class TF_3D(nn.Module):
    """
    带 SSP 频谱-形态提示门的 3D Token 融合块：
    - project(): 把各模态 [B,C,D,H,W] 池化到 patch token [B,P^3,C] 并在 token 维 concat
    - fusion_block: TransformerEncoder
    - （可选）SSP：对每个模态做 3D FFT 频谱门，得到 token，和融合 token concat+Sigmoid 做门控
    - reproject(): 把融合后的 token 还原到每模态的体素尺度 -> softmax(模态维) -> 加权求和
    """
    def __init__(
        self,
        embedding_dim: int = 1024,
        volumn_size: int = 8,
        nhead: int = 4,
        num_layers: int = 8,
        method: str = 'TF',
        trans: str = 'ff',                 # 兼容你们上游的传参习惯
        use_ssp: bool = True,              # ★ 开关：是否启用 SSP 频谱门
        ssp_n_bases: int = 8,
        ssp_basis_type: str = 'loggabor',  # 'loggabor' | 'spherical' | 'learnable'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.d_model = embedding_dim
        self.patch_dim = 8
        self.method = method
        self.scale_factor = max(1, volumn_size // self.patch_dim)
        self.use_ssp = use_ssp

        # --- 基础 Transformer 融合 ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, batch_first=True,
            dim_feedforward=self.d_model * 4
        )
        self.fusion_block = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool3d((self.patch_dim, self.patch_dim, self.patch_dim))
        self.upsample = DUpsampling3D(self.embedding_dim, self.scale_factor)

        # （可选）token 形式的频谱提示：和融合 token concat 后用 Sigmoid 做门控
        if self.use_ssp:
            self.ssp = SpectralGate3D(
                in_channels=self.embedding_dim,
                n_bases=ssp_n_bases,
                basis_type=ssp_basis_type,
                learn_gamma=True,
                use_token=True,
                token_dim=self.embedding_dim,   # 直接对齐 C，方便 concat
                patch_dim=self.patch_dim
            )
            self.gate_proj = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.Sigmoid()
            )

        # 可选的 Token 学习（你原来的接口保留）
        if method == 'Token':
            self.fusion_token = nn.Parameter(
                torch.zeros((1, self.patch_dim ** 3, self.d_model))
            )

    # ----------------- 前向 -----------------
    def forward(self, all_content):
        """
        all_content: list[Tensor]，每个模态 [B,C,D,H,W]，C == embedding_dim
        """
        n_modality = len(all_content)

        # 1) 空间分支：各模态转 token 并 concat
        token_content = self.project(all_content)                              # [B, N=P^3*M, C]
        position_enc = PositionalEncoding(self.d_model, token_content.size(1))
        spatial_tok = self.dropout(position_enc(token_content))                # [B,N,C]

        # 2) Transformer 融合
        out = self.fusion_block(spatial_tok)                                   # [B,N,C]

        # 3) （可选）SSP 频谱门：拿频域 token 与 out concat 后 Sigmoid 门控
        if self.use_ssp:
            ssp_toks = []
            for x in all_content:
                # z_f 未直接用到；需要的话可做平行分支消融
                _, tok = self.ssp(x)                                           # tok:[B,P^3,C]
                ssp_toks.append(tok)
            ssp_tok = torch.cat(ssp_toks, dim=1)                               # [B,N,C]
            gate = self.gate_proj(torch.cat([out, ssp_tok], dim=2))            # [B,N,C]
            out = out * gate

        # 4) 还原到每模态体素尺度 -> 模态维 softmax -> 加权求和
        atten_map = self.reproject(out, n_modality, self.method)               # [M,B,C,D,H,W] (经上采样)
        return self.atten(all_content, atten_map, n_modality)                  # [B,C,D,H,W]

    # ----------------- token 化 -----------------
    def project(self, all_content):
        n_modality = len(all_content)
        token_content_in = None
        for i in range(n_modality):
            content = self.avgpool(all_content[i])                              # [B,C,P,P,P]
            content = content.permute(0, 2, 3, 4, 1).contiguous()               # [B,P,P,P,C]
            content2 = content.view(content.size(0), -1, self.embedding_dim)    # [B,P^3,C]
            token_content_in = content2 if token_content_in is None else torch.cat([token_content_in, content2], dim=1)
        return token_content_in

    # ----------------- 还原/上采样 -----------------
    def reproject(self, atten_map, n_modality, method):
        n_patch = self.patch_dim ** 3
        a_m0 = None
        for i in range(n_modality):
            atten_mapi = atten_map[:, n_patch * i: n_patch * (i + 1), :].view(
                atten_map.size(0),
                self.patch_dim, self.patch_dim, self.patch_dim,
                self.embedding_dim,
            )
            atten_mapi = atten_mapi.permute(0, 4, 1, 2, 3).contiguous()        # [B,C,P,P,P]
            atten_mapi = self.upsample(atten_mapi).unsqueeze(dim=0)            # [1,B,C,D,H,W]
            a_m0 = atten_mapi if a_m0 is None else torch.cat([a_m0, atten_mapi], dim=0)

        a_m = F.softmax(a_m0, dim=0)                                           # 模态维 softmax
        return a_m

    # ----------------- 模态加权 -----------------
    def atten(self, all_content, atten_map, n_modality):
        output = None
        for i in range(n_modality):
            a_m = atten_map[i, :, :, :, :, :]                                  # [B,C,D,H,W]
            assert all_content[i].shape == a_m.shape, 'all_content and a_m cannot match!!'
            output = all_content[i] * a_m if output is None else output + all_content[i] * a_m
        return output

