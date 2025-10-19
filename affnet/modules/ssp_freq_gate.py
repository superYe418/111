import torch, math
import torch.nn as nn
import torch.nn.functional as F

# ---------- 频率网格 ----------
def meshgrid_freq_3d(D, H, W, device, dtype):
    fz = torch.fft.fftfreq(D, d=1., device=device).view(D,1,1)
    fy = torch.fft.fftfreq(H, d=1., device=device).view(1,H,1)
    fx = torch.fft.fftfreq(W, d=1., device=device).view(1,1,W)
    fr = torch.sqrt(fz*fz + fy*fy + fx*fx)                     # [D,H,W]
    return fr.to(dtype)

# ---------- 几类频谱基（可扩展） ----------
def log_gabor_3d(fr, mu, sigma):
    # fr∈[0,0.5]，mu∈(0,0.5]，sigma>0
    # 避免 log(0)
    eps = 1e-6
    lg = torch.exp(-(torch.log(fr + eps) - math.log(mu))**2 / (2 * (sigma**2)))
    # 抑制直流
    lg = lg * (fr > 0).to(lg.dtype)
    return lg

def spherical_shell_3d(fr, mu, sigma):
    return torch.exp(-0.5 * ((fr - mu) ** 2) / (sigma ** 2))

# ---------- 频谱门主体 ----------
class SpectralGate3D(nn.Module):
    """
    x:[B,C,D,H,W] -> (z_f:[B,C,D,H,W], tok:[B,P^3,C] or None)
    - K 个频谱基 Φ_k；γ 可学习或由数据自适应产生
    - basis_type: 'loggabor' | 'spherical' | 'learnable'
    - reduce: 'none' | 'gap'（若需要把 z_f 也做 token 可用）
    """
    def __init__(self,
                 in_channels: int,
                 n_bases: int = 8,
                 basis_type: str = "loggabor",
                 learn_gamma: bool = True,
                 reduce: str = "none",
                 token_dim: int = 32,
                 use_token: bool = True,
                 patch_dim: int = 8):
        super().__init__()
        self.C = in_channels
        self.K = n_bases
        self.basis_type = basis_type
        self.learn_gamma = learn_gamma
        self.reduce = reduce
        self.use_token = use_token
        self.patch_dim = patch_dim

        # 基的参数（径向中心与宽度）；learnable 以 mu∈(0,0.5], sigma>0 为目标
        self.mu = nn.Parameter(torch.linspace(0.05, 0.45, n_bases)[None,:])
        self.log_sigma = nn.Parameter(torch.log(torch.full((1, n_bases), 0.15)))

        # γ 生成：learnable 向量 or 由上下文产生（更推荐）
        if learn_gamma:
            self.gamma_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_channels, in_channels//4, 1),
                nn.GELU(),
                nn.Conv3d(in_channels//4, n_bases, 1)
            )
            self.gamma_act = nn.Softmax(dim=1)
        else:
            self.gamma = nn.Parameter(torch.zeros(1, n_bases))

        # token 投影（可选）：z_f → token
        if use_token:
            self.avgpool = nn.AdaptiveAvgPool3d((patch_dim, patch_dim, patch_dim))
            self.to_tok = nn.Linear(in_channels, token_dim)

        # 缓存频率网格
        self.register_buffer('_fr', torch.tensor(0.), persistent=False)

    def _build_bases(self, shape, device, dtype):
        B,C,D,H,W = shape
        fr = meshgrid_freq_3d(D,H,W, device, dtype)            # [D,H,W]
        mu = self.mu.to(dtype); sigma = torch.exp(self.log_sigma).to(dtype)
        if self.basis_type == "loggabor":
            basis = [log_gabor_3d(fr, float(mu[0,i]), float(sigma[0,i])) for i in range(self.K)]
        elif self.basis_type == "spherical":
            basis = [spherical_shell_3d(fr, float(mu[0,i]), float(sigma[0,i])) for i in range(self.K)]
        else:
            # 完全可学习基：初始化为球壳
            basis = [spherical_shell_3d(fr, float(mu[0,i]), float(sigma[0,i])) for i in range(self.K)]
            basis = [nn.Parameter(b.clone().detach()) for b in basis]
            self.basis_params = nn.Parameter(torch.stack(basis, dim=0))
            return self.basis_params  # [K,D,H,W]
        return torch.stack(basis, dim=0)                        # [K,D,H,W]

    def forward(self, x):
        B,C,D,H,W = x.shape
        Xf = torch.fft.fftn(x, dim=(2,3,4))                    # 复数
        bases = self._build_bases(x.shape, x.device, x.dtype)  # [K,D,H,W]

        if self.learn_gamma:
            gamma = self.gamma_act(self.gamma_head(x).view(B, self.K))  # [B,K]
        else:
            gamma = self.gamma.expand(B, -1)                            # [B,K]

        # M_b(ω) = sum_k γ_bk Φ_k(ω)
        M = torch.einsum('bk,kdhw->bdhw', gamma, bases).unsqueeze(1)    # [B,1,D,H,W]
        Xf_hat = Xf * M
        z_f = torch.fft.ifftn(Xf_hat, dim=(2,3,4)).real                 # [B,C,D,H,W]

        tok = None
        if self.use_token:
            z = self.avgpool(z_f)                                       # [B,C,P,P,P]
            z = z.permute(0,2,3,4,1).reshape(B, -1, C)                  # [B,P^3,C]
            tok = self.to_tok(z)                                        # [B,P^3,token_dim]
        return z_f, tok
