# ======================= train.py (clean, stable, AMP-enabled) =======================
import os, sys, argparse, time, datetime, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.amp import autocast, GradScaler

# ---------- 【长期稳态设置】 ----------
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')
torch.backends.cudnn.enabled = True          # 开启 cuDNN
torch.backends.cudnn.benchmark = True         # 让 cuDNN 选最快 kernel（更快）
torch.backends.cudnn.deterministic = False    # 允许非确定性换速度
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
# 调试期需要时再打开：
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True)

# ==== EARLY PARSE to protect from external libs (affnet) ====
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument('--phase', type=str, default='train')
_early.add_argument('--model_name', type=str, default='RsInOut_U_Hemis3D')
_early_args, _early_rest = _early.parse_known_args()

# 清空 argv，防止外部库 import 时抢解析
_sys_argv_backup = sys.argv[:]
sys.argv = [sys.argv[0]]

# ==== 再 import 其它依赖 ====
import SimpleITK as sitk

from loader.Dataloader import get_loaders
from utils.util import check_dirs, print_net, re_crop, MulticlassDiceLoss
from loss import get_dice, U_Hemis_loss, RMBTS_loss, LMCR_loss
from net.Network_HEMIS import U_Hemis3D, TF_U_Hemis3D, AF_U_Hemis3D, InOut_U_Hemis3D, RsInOut_U_Hemis3D
from net.Network_RMBTS import RMBTS, TF_RMBTS
from net.Network_LMCR import LMCR, TF_LMCR
from process.utils import parse_image_name, missing_list


# ======================= Solver =======================
class Solver:
    def __init__(self, data_files, opt):
        self.opt = opt
        print(self.opt)

        # records
        self.best_epoch = 0
        self.best_dice = 0.0
        self.best_epoch_extra = 0
        self.best_dice_extra = 0.0

        # options
        self.further_train = self.opt.further_train
        self.further_epoch = self.opt.further_epoch
        self.trans = self.opt.trans
        self.model_name = self.opt.model_name
        self.TF_methods = self.opt.TF_methods
        self.phase = self.opt.phase
        self.out_channels = self.opt.out_channels
        self.in_channels = self.opt.in_channels
        self.levels = self.opt.levels
        self.feature_maps = self.opt.feature_maps
        self.selected_modal = self.opt.selected_modal
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers

        loaders = get_loaders(data_files, self.selected_modal, self.batch_size, self.num_workers)
        print('Get Loader')
        self.loaders = {x: loaders[x] for x in ('train', 'val', 'test')}

        self.c_dim = len(self.selected_modal)

        self.max_epoch = self.opt.max_epoch
        self.decay_epoch = self.opt.decay_epoch
        self.lr = self.opt.lr
        self.min_lr = self.opt.min_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.ignore_index = self.opt.ignore_index
        self.seg_loss_type = self.opt.seg_loss_type
        self.n_critic = self.opt.n_critic
        self.miss_list = missing_list()
        self.mdice = MulticlassDiceLoss()

        self.test_epoch = self.opt.test_epoch
        self.use_tensorboard = self.opt.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # dirs
        self.checkpoint_dir = self.opt.checkpoint_dir
        self.sample_dir = os.path.join(self.checkpoint_dir, 'sample_dir')
        self.model_save_dir = os.path.join(self.checkpoint_dir, 'model_save_dir')
        self.result_dir = os.path.join(self.checkpoint_dir, 'result_dir')
        check_dirs([self.model_save_dir, self.result_dir, self.sample_dir, self.checkpoint_dir])
        print('Checked DIR')

        self.log_step = self.opt.log_step
        self.val_epoch = self.opt.val_epoch
        self.lr_update_epoch = self.opt.lr_update_epoch

        self.G = None
        self.scaler = GradScaler(device='cuda')  # 【新增】AMP scaler
        self.build_model()
        print('......')

    def build_model(self):
        print('Building...')
        print('model_name==', self.model_name)

        # 【改】避免 eval，安全映射
        MODEL_ZOO = {
            'U_Hemis3D': U_Hemis3D,
            'TF_U_Hemis3D': TF_U_Hemis3D,
            'AF_U_Hemis3D': AF_U_Hemis3D,
            'InOut_U_Hemis3D': InOut_U_Hemis3D,
            'RsInOut_U_Hemis3D': RsInOut_U_Hemis3D,
            'RMBTS': RMBTS, 'TF_RMBTS': TF_RMBTS,
            'LMCR': LMCR, 'TF_LMCR': TF_LMCR,
        }
        assert self.model_name in MODEL_ZOO, f'Unknown model: {self.model_name}'
        
        self.G = MODEL_ZOO[self.model_name](
            in_channels=self.in_channels, out_channels=self.out_channels,
            levels=self.levels, feature_maps=self.feature_maps,
            method=self.TF_methods, phase=self.phase,
            cfg=self.opt  # ★ 关键：把 args 下传
        )

        if self.phase == 'train':
            print_net(self.G, self.model_name)
        self.G.to(self.device)
        # 方式 A：Adam（推荐）
        self.g_optimizer = optim.Adam(
            self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=0.0
        )
        '''
        # 方式 B：AdamW 但无衰减
        self.g_optimizer = optim.AdamW(
            self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=1e-4
        )
        '''
    def restore_model(self, epoch):
        G_path = os.path.join(self.model_save_dir, f'{epoch}-G.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=self.device))

    def save_model(self, save_iters):
        G_path = os.path.join(self.model_save_dir, f'{save_iters}-G.ckpt')
        torch.save(self.G.state_dict(), G_path)
        print(f'Saved model checkpoints into {self.model_save_dir}...')

    def update_lr(self, lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.g_optimizer.zero_grad(set_to_none=True)

    def clip_grad(self):
        nn.utils.clip_grad_norm_(self.G.parameters(), 20.0, 2)

    @staticmethod
    def classification_loss(logit, target):
        return F.cross_entropy(logit, target)

    def check_m_d(self, m_d):
        n = m_d[0]
        for i in range(len(m_d)):
            if n != m_d[i]:
                print('error for different m_d in one batch!')

    # 原实现保持（注意：按你数据缺失定义再评估是否需要改语义）
    def replace_modality(self, inputs, m_d):
        sum_modal, num = None, 0
        for i in range(len(inputs)):
            if m_d in self.miss_list[i]:
                sum_modal = inputs[i] if sum_modal is None else (sum_modal + inputs[i])
                num += 1
        aver = sum_modal / max(1, num)
        for i in range(len(inputs)):
            if m_d not in self.miss_list[i]:
                inputs[i] = aver.clone()
        return inputs

    def train(self):
        loaders = {'train': self.loaders['train']}
        lr = self.lr
        start_epoch = 0
        cur_step = -1

        print('\nStart training...')
        if self.further_train:
            self.restore_model(self.further_epoch)
            start_epoch = self.further_epoch + 1

        start_time = time.time()

        for epoch in range(start_epoch, self.max_epoch):
            self.G.train()

            for p in loaders.keys():
                for i, batch_data in enumerate(loaders[p]):
                    cur_step += 1
                    loss = {}

                    # 【改】统一设备与 dtype
                    volume_label = batch_data[4].unsqueeze(1).float().to(self.device, non_blocking=True)
                    pid = batch_data[5]
                    m_d = batch_data[6].to(self.device)

                    if (i + 1) % self.n_critic == 0:
                        inputs = []
                        for k in range(len(self.selected_modal)):
                            xk = batch_data[k].unsqueeze(1).float().to(self.device, non_blocking=True)
                            inputs.append(xk)

                        if self.model_name == 'LMCR':
                            inputs = self.replace_modality(inputs, m_d[0])

                        seg, g_loss = None, None

                        # 【新增】AMP 前向 + loss
                        with autocast('cuda'):
                            if 'Hemis3D' in self.model_name:
                                seg = self.G(inputs, m_d[0])
                                g_loss = U_Hemis_loss(seg, volume_label, self.mdice)
                            elif self.model_name in ['RMBTS', 'TF_RMBTS']:
                                re_dic = self.G(inputs, m_d[0])
                                seg = re_dic['seg']
                                g_loss, dice_loss, ce_loss, rec_loss, KL_loss, shape_loss = RMBTS_loss(
                                    re_dic, volume_label, inputs, self.mdice, m_d[0], self.miss_list, self.device,
                                    use_shape=self.opt.use_shape_reg,
                                    lambda_shape=self.opt.lambda_shape,
                                    shape_cls=self.opt.shape_cls,
                                    shape_eps=self.opt.shape_eps,
                                )
                                loss['shape'] = float(shape_loss)

                            elif self.model_name in ['LMCR', 'TF_LMCR']:
                                re_dic = self.G(inputs, m_d[0])
                                seg = re_dic['seg']
                                g_loss, dice_loss, rec_loss, shape_loss = LMCR_loss(
                                    re_dic, volume_label, inputs, self.mdice, m_d[0], self.miss_list, self.device,
                                    use_shape=self.opt.use_shape_reg,
                                    lambda_shape=self.opt.lambda_shape,
                                    shape_cls=self.opt.shape_cls,
                                    shape_eps=self.opt.shape_eps,
                                )
                                loss['shape'] = float(shape_loss)

                            else:
                                raise RuntimeError('error methods!!')

                        dice_wt, dice_tc, dice_et = get_dice(seg, volume_label)
                        
                        with torch.no_grad():
                            et_vox_true = int((volume_label == 4).sum().item())  # 这批标签里 ET 体素数
                            et_vox_pred = int((torch.argmax(F.softmax(seg, dim=1), dim=1, keepdim=True) == 3).sum().item())  # 预测为 ET 的体素数（索引3）
                            p_et_mean   = float(F.softmax(seg, dim=1)[:, 3].mean().item())  # ET 通道平均概率

                        loss['et_vox_t'] = et_vox_true
                        loss['et_vox_p'] = et_vox_pred
                        loss['p_et']     = p_et_mean
                        
                        # NaN/Inf 防护
                        with torch.no_grad():
                            if not torch.isfinite(g_loss):
                                print('[NaN guard] g_loss nan/inf, skip this iter')
                                continue

                        # 反传（AMP）
                        self.reset_grad()
                        self.scaler.scale(g_loss).backward()
                        self.clip_grad()
                        self.scaler.step(self.g_optimizer)
                        self.scaler.update()

                        loss['G/s'] = float(g_loss)
                        loss['dc_wt'] = float(dice_wt)
                        loss['dc_tc'] = float(dice_tc)
                        loss['dc_et'] = float(dice_et)
                        loss['pid'] = int(pid[0])
                        loss['m_d'] = int(m_d[0])

                    # logs
                    if (cur_step + 1) % self.log_step == 0:
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=int(et)))
                        line = f"Elapsed [{et}], Epoch [{epoch + 1}/{self.max_epoch}], Iters [{cur_step}]"
                        for k, v in loss.items():
                            if k in ['pid', 'm_d']:
                                line += f", {k}: {v}"
                            else:
                                line += f", {k}: {v:.4f}"
                        print(line)

            if (epoch + 1) % self.val_epoch == 0:
                print('\n')
                d1, d2, d3, dps = self.val(epoch + 1)
                print(f'Current dps of validation WT: {d1:.4f} TC: {d2:.4f} ET: {d3:.4f} Aver: {dps:.4f}')

            if (epoch + 1) % self.lr_update_epoch == 0 and (epoch + 1) > (self.max_epoch - self.decay_epoch) and self.decay_epoch > 0:
                dlr = self.lr - self.min_lr
                lr -= dlr / (self.decay_epoch / self.lr_update_epoch)
                lr = max(lr, self.min_lr)
                self.update_lr(lr)
                print(f'Decayed learning rates, lr: {lr}.')

    def val(self, epoch):
        save_dir = os.path.join(self.result_dir, str(epoch))
        print(f'Start validation at iter {epoch}...')
        self.G.eval()
        loaders = {'val': self.loaders['val']}
        d1 = d2 = d3 = 0.0
        n = 0

        with torch.no_grad():
            for p in loaders.keys():
                for i, batch_data in enumerate(loaders[p]):
                    volume_label = batch_data[4].unsqueeze(1).float().to(self.device, non_blocking=True)
                    m_d = batch_data[6].to(self.device)

                    inputs = []
                    for k in range(len(self.selected_modal)):
                        xk = batch_data[k].unsqueeze(1).float().to(self.device, non_blocking=True)
                        inputs.append(xk)

                    if self.model_name in ['LMCR']:
                        inputs = self.replace_modality(inputs, m_d[0])

                    with autocast('cuda'):
                        if 'Hemis3D' in self.model_name:
                            seg = self.G(inputs, m_d[0])
                        else:
                            re_dic = self.G(inputs, m_d[0])
                            seg = re_dic['seg']

                    dice_wt, dice_tc, dice_et = get_dice(seg, volume_label)
                    d1 += float(dice_wt)
                    d2 += float(dice_tc)
                    d3 += float(dice_et)
                    n += 1

        dps = (d1 + d2 + d3) / (3 * max(1, n))

        # 【改】仅在提升时保存
        if dps > self.best_dice:
            self.best_epoch = epoch
            self.best_dice = dps
            self.save_model(epoch)

        print(f'Current best dps : {self.best_dice:.4f} of epoch : {self.best_epoch}')
        return d1 / n, d2 / n, d3 / n, dps

    def infer(self, epoch, method='forward'):
        loaders = {'test': self.loaders['test']}
        save_dir = os.path.join(self.result_dir, str(epoch))
        check_dirs(save_dir)
        self.restore_model(epoch)
        self.G.eval()
        num_max = 69 * 15

        with torch.no_grad():
            for p in loaders.keys():
                for i, batch_data in enumerate(loaders[p]):
                    pid = batch_data[5]
                    m_d = batch_data[6].to(self.device)
                    crop_size = batch_data[7]
                    print(f'{i}/{num_max}\t p_id:{pid[0]}\t m_id:{int(m_d[0])}')
                    inputs = []
                    for k in range(len(self.selected_modal)):
                        xk = batch_data[k].unsqueeze(1).float().to(self.device, non_blocking=True)
                        inputs.append(xk)

                    if self.model_name in ['LMCR']:
                        inputs = self.replace_modality(inputs, m_d[0])

                    with autocast('cuda'):
                        if self.model_name in ['TF_U_Hemis3D', 'U_Hemis3D', 'RsInOut_U_Hemis3D']:
                            seg = self.G(inputs, m_d[0])
                        else:
                            re_dic = self.G(inputs, m_d[0])
                            seg = re_dic['seg']

                    seg = F.softmax(seg, dim=1)
                    _, pr = torch.max(seg, dim=1, keepdim=True)
                    pr[pr == 3] = 4
                    pre = np.array(re_crop(pr.squeeze(0).squeeze(0), crop_size).cpu()).astype(np.float32)
                    out = sitk.GetImageFromArray(pre)
                    sitk.WriteImage(out, os.path.join(save_dir, f'{pid[0]}_{int(m_d[0])}.nii.gz'))
        return


# ======================= main =======================
if __name__ == '__main__':
    # 保持非基准搜索、确定性（与上方一致）
    cudnn.benchmark = False
    cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    # 用早解析默认值（这样即使命令行被清空，也能拿到你传入的设定）
    parser.add_argument('--model_name', type=str, default=_early_args.model_name, help='Model name')
    parser.add_argument('--phase', type=str, default=_early_args.phase, help='Phase (train or test)')

    parser.add_argument('--trans', type=str, default='trans')
    parser.add_argument('--train_list', type=str, default='./process/partition/0-train.txt')
    parser.add_argument('--val_list', type=str, default='./process/partition/0-val.txt')
    parser.add_argument('--test_list', type=str, default='./process/partition/0-test.txt')

    parser.add_argument('--selected_modal', nargs='+', default=['t1ce', 't1', 't2', 'flair'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_channels', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--feature_maps', type=int, default=8)
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--norm_type', type=str, default='instance')
    parser.add_argument('--use_dropout', type=bool, default=True)

    parser.add_argument('--decay_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--ignore_index', type=int, default=None)
    parser.add_argument('--seg_loss_type', type=str, default='cross-entropy')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_weight', type=bool, default=True)
    parser.add_argument('--n_critic', type=int, default=1)

    parser.add_argument('--method', type=str, default='forward')

    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--device', type=bool, default=True)
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--train_epoch', nargs='+', default=['50', '100', '150', '200'])
    parser.add_argument('--test_epoch', nargs='+', default=['50', '100', '150', '200'])
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--further_train', type=bool, default=False)
    parser.add_argument('--further_epoch', type=int, default=0)

    parser.add_argument('--TF_methods', type=str, default='TF')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')

    parser.add_argument('--log_step', type=int, default=60)
    parser.add_argument('--val_epoch', type=int, default=5)
    parser.add_argument('--lr_update_epoch', type=int, default=1000)

    # === SSP 频谱门相关（启用 + 配置） ===
    parser.add_argument('--use_ssp', action='store_true', default=False)
    parser.add_argument('--ssp_n_bases', type=int, default=8)
    parser.add_argument('--ssp_basis_type', type=str, default='loggabor',
                        choices=['loggabor','spherical','learnable'])
    parser.add_argument('--ssp_use_token', type=bool, default=True)

    # === 形态曲率正则 ===
    parser.add_argument('--use_shape_reg', action='store_true', default=False)
    parser.add_argument('--lambda_shape', type=float, default=0.05)
    parser.add_argument('--shape_eps', type=float, default=1e-3)
    parser.add_argument('--shape_cls', type=int, default=3)  # 3: ET

    # === 训练小技巧：给 ET 通道 logits 加偏置，扶一把 argmax ===
    parser.add_argument('--et_logit_bias', type=float, default=0.0)

    # 【改】把“早解析剩余参数”继续传进来，命令行参数有效
    args = parser.parse_args(_early_rest)

    # === 强制开启 SSP（兜底，最省事） ===
    args.use_ssp = True
    args.ssp_n_bases = 8
    args.ssp_basis_type = 'loggabor'

    print('-----Config-----')
    for k, v in sorted(vars(args).items()):
        print(f'{k}:\t{v}')
    print('-------End------\n')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    data_files = dict(train=args.train_list, val=args.val_list, test=args.test_list)
    args.checkpoint_dir = args.checkpoint_dir + args.model_name

    solver = Solver(data_files, args)
    if args.phase == 'train':
        solver.train()
    elif args.phase == 'test':
        print('calculating...')
        for test_iter in args.test_epoch:
            test_iter = int(test_iter)
            solver.infer(test_iter, args.method)
    
    loss['shape'] = float(shape_loss)

    # 可选：恢复 argv
    # sys.argv = _sys_argv_backup
    print('Done!')
# =====================================================================================
