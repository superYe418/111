
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _cp  # ★ 新增

from net.BasicBlock import UNetEncoder, UNetDecoder, ConvNormRelu3D, TF_3D
from net.RestormerCnnBlock import Restormer_CNN_block, RSEncoder
from process.utils import missing_list


# ---------- 小工具：在训练阶段用 checkpoint 包裹模块调用 ----------
def _cp_call(module, x, use_cp: bool):
    """对单输入 Tensor 的模块进行 checkpoint（encoder 这类）"""
    if use_cp:
        return _cp(module, x)
    return module(x)

def _cp_call_list(module, xs, use_cp: bool):
    """对接收 list[Tensor] 的模块进行 checkpoint（TF_3D 融合、skip 融合这类）"""
    if use_cp:
        # checkpoint 接口要求 *args 形式的 Tensor；用 lambda 重新打包成 list 传给 module
        return _cp(lambda *args: module(list(args)), *xs)
    return module(xs)


class U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method=None, phase='train'):
        super(U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)
        self.fusion = ConvNormRelu3D(2**(self.levels)*feature_maps, 2**(self.levels-1)*feature_maps)

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = ConvNormRelu3D(2**(self.levels-i-1) * feature_maps, 2**(self.levels-i-2) * feature_maps)
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        use_cp = self.training  # 仅训练时启用 checkpoint

        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = _cp_call(getattr(self.encoders, 'encoder%d' % (k)), inputs[k], use_cp)
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = None
            for k in range(len(all_encoder_outputs)):
                if e_output is None:
                    e_output = all_encoder_levelputs[k][i].unsqueeze(dim=0)
                else:
                    e_output = torch.cat([e_output, all_encoder_levelputs[k][i].unsqueeze(dim=0)], dim=0)

            vs, ms = torch.var_mean(e_output, dim=0)
            if len(all_encoder_outputs) == 1:
                vs = torch.zeros_like(ms)
            else:
                vs = vs / (len(all_encoder_outputs)-1)
            encoder_outputs.append(getattr(self.skipfuion, 'skip_fusion%d' % (i+1))(torch.cat([ms,vs],dim=1)))

        output = None
        for k in range(len(all_encoder_outputs)):
            if output is None:
                output = all_encoder_outputs[k].unsqueeze(dim=0)
            else:
                output = torch.cat([output, all_encoder_outputs[k].unsqueeze(dim=0)], dim=0)
        v, m = torch.var_mean(output, dim=0)
        if len(all_encoder_outputs) == 1:
            v = torch.zeros_like(m)
        else:
            v = v / (len(all_encoder_outputs) - 1)
        output = self.fusion(torch.cat([m,v],dim=1))
        seg = self.decoder(output, encoder_outputs)

        return seg


class TF_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train', trans='basic'):
        super(TF_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps,
                            volumn_size=(128//(2**(self.levels-1))), method=method, trans=trans)

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps,
                                volumn_size=(128//(2**(self.levels-2-i))), method=method, trans=trans)
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        use_cp = self.training

        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = _cp_call(getattr(self.encoders, 'encoder%d' % (k)), inputs[k], use_cp)
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            oput = _cp_call_list(getattr(self.skipfuion, 'skip_fusion%d' % (i+1)), e_output, use_cp)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = _cp_call_list(self.fusion, output, use_cp)

        seg = self.decoder(output, encoder_outputs)
        return seg


class AF_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train'):
        super(AF_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps,
                            volumn_size=(128//(2**(self.levels-1))), method=method, trans='ff')

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps,
                                volumn_size=(128//(2**(self.levels-2-i))), method=method)
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        use_cp = self.training

        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = _cp_call(getattr(self.encoders, 'encoder%d' % (k)), inputs[k], use_cp)
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            oput = _cp_call_list(getattr(self.skipfuion, 'skip_fusion%d' % (i+1)), e_output, use_cp)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = _cp_call_list(self.fusion, output, use_cp)

        seg = self.decoder(output, encoder_outputs)
        return seg


class InOut_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train'):
        super(InOut_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps,
                            volumn_size=(128//(2**(self.levels-1))), method=method, trans='ff')

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps,
                                volumn_size=(128//(2**(self.levels-2-i))), method='la')
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        use_cp = self.training

        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = _cp_call(getattr(self.encoders, 'encoder%d' % (k)), inputs[k], use_cp)
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            oput = _cp_call_list(getattr(self.skipfuion, 'skip_fusion%d' % (i + 1)), e_output, use_cp)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = _cp_call_list(self.fusion, output, use_cp)

        seg = self.decoder(output, encoder_outputs)
        return seg


class RsInOut_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32,
                 method='TF', phase='train', cfg=None):   # ★ 新增 cfg
        super(RsInOut_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        # ★ 从 cfg 读 SSP 配置（若没传 cfg 也有默认）
        self.use_ssp        = bool(getattr(cfg, 'use_ssp', False))
        self.ssp_n_bases    = int(getattr(cfg, 'ssp_n_bases', 8))
        self.ssp_basis_type = str(getattr(cfg, 'ssp_basis_type', 'loggabor'))
        self.ssp_use_token  = bool(getattr(cfg, 'ssp_use_token', True))

        # encoders
        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = RSEncoder(self.in_channels, feature_maps)
            self.encoders.add_module(f'encoder{i}', encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels, feature_maps=feature_maps, levels=self.levels)

        # outfusion（bridge 融合）
        self.fusion = TF_3D(
            embedding_dim=2**(self.levels-1) * feature_maps,
            volumn_size=(128 // (2**(self.levels-1))),
            method=method,
            trans='ff'
        )
        # ★ 把 SSP 配置挂到 fusion（TF_3D 在 forward 里读取这些属性）
        self.fusion.use_ssp        = self.use_ssp
        self.fusion.ssp_n_bases    = self.ssp_n_bases
        self.fusion.ssp_basis_type = self.ssp_basis_type
        self.fusion.ssp_use_token  = self.ssp_use_token

        # skip 融合
        self.skipfuion = nn.Sequential()
        for i in range(self.levels - 1):
            skip_fusion = TF_3D(
                embedding_dim=2**(self.levels - i - 2) * feature_maps,
                volumn_size=(128 // (2**(self.levels - 2 - i))),
                method='la'
            )
            # ★ 同样挂 SSP 配置，保证一致性
            skip_fusion.use_ssp        = self.use_ssp
            skip_fusion.ssp_n_bases    = self.ssp_n_bases
            skip_fusion.ssp_basis_type = self.ssp_basis_type
            skip_fusion.ssp_use_token  = self.ssp_use_token

            self.skipfuion.add_module(f'skip_fusion{self.levels - 1 - i}', skip_fusion)

    def forward(self, inputs, m_d):
        use_cp = self.training

        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = _cp_call(getattr(self.encoders, f'encoder{k}'), inputs[k], use_cp)
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels - 1):
            e_output = [all_encoder_levelputs[k][i] for k in range(len(all_encoder_outputs))]
            oput = _cp_call_list(getattr(self.skipfuion, f'skip_fusion{i + 1}'), e_output, use_cp)
            encoder_outputs.append(oput)

        output = [all_encoder_outputs[k] for k in range(len(all_encoder_outputs))]
        output = _cp_call_list(self.fusion, output, use_cp)

        seg = self.decoder(output, encoder_outputs)
        return seg
