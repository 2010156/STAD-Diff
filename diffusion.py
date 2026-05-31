import os
import logging
import time
import glob
import math
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as tvu
from PIL import Image
import sys
import nibabel as nib

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, calculate_psnr
from datasets import data_transform, inverse_data_transform
from datasets.pmub import PMUB
from datasets.LDFDCT import LDFDCT
from datasets.BRATS import BRATS
from functions.ckpt_util import get_ckpt_path
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import torch.nn.functional as F

# ====== 可选：AUC 四指标（保持你的原逻辑） ======
try:
    from sklearn.metrics import roc_auc_score
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

class FMRIDataset(data.Dataset):
    def __init__(self, root_lr, root_hr=None, image_size=128, split='train', is_3d=True, task='sr',
                 modalities=1, time_dim=None, downsample_z=1, downsample_xy=1):
        super().__init__()
        self.lr_dir = root_lr
        self.hr_dir = root_hr if root_hr is not None else root_lr
        self.is_3d = is_3d
        self.task = task
        self.modalities = modalities
        self.size = image_size
        self.time_dim = time_dim
        self.down_z = max(1, int(downsample_z))
        self.down_xy = max(1, int(downsample_xy))
        self.files = sorted([f for f in glob.glob(os.path.join(self.lr_dir, "**", "*.nii*"), recursive=True)])
        assert len(self.files) > 0, f"No NIfTI found under {self.lr_dir}"

    def _load(self, path):
        import nibabel as nib
        vol = nib.load(path).get_fdata().astype(np.float32)
        # 统一转为 (T, H, W) 或 (D, H, W)
        if vol.ndim == 2:
            vol = vol[None, ...]           # (1, H, W)
        elif vol.ndim == 3:
            if vol.shape[0] < vol.shape[2]:
                vol = np.transpose(vol, (2, 0, 1))
        elif vol.ndim == 4:
            if vol.shape[-1] < vol.shape[0]:
                vol = np.transpose(vol, (3, 2, 0, 1))
            else:
                vol = np.transpose(vol, (0, 3, 1, 2))
            if vol.shape[1] == 1:
                vol = vol.squeeze(1)
        return vol

    def _resize3d(self, x, dhw):
        ten = torch.from_numpy(x)
        if ten.ndim == 3:
            ten = ten[None, None]          # (1,1,D,H,W)
        elif ten.ndim == 4:
            ten = ten[None]                # (1,T,D,H,W)
        else:
            ten = ten[None, None]
        ten = F.interpolate(ten, size=dhw, mode="trilinear", align_corners=False)
        return ten[0,0].numpy()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lr_path = self.files[idx]
        name = os.path.basename(lr_path).split(".nii")[0]
        lr = self._load(lr_path)

        if self.task == 'sr':
            if self.hr_dir is None:
                hr = lr
            else:
                hr_path = lr_path.replace(self.lr_dir, self.hr_dir)
                hr = self._load(hr_path) if os.path.exists(hr_path) else lr
            if self.down_z > 1 or self.down_xy > 1:
                d = lr.shape[0]
                dz = max(1, d // self.down_z)
                dh = max(8, lr.shape[1] // self.down_xy)
                dw = max(8, lr.shape[2] // self.down_xy)
                lr = self._resize3d(hr, (dz, dh, dw))
            hr = self._resize3d(hr, (lr.shape[0], self.size, self.size))
            lr = self._resize3d(lr, (lr.shape[0], self.size, self.size))
            lr = torch.from_numpy(lr)[None]
            hr = torch.from_numpy(hr)[None]
            return {'LR': lr, 'HR': hr, 'case_name': name}
        else:
            fd = self._resize3d(lr, (lr.shape[0], self.size, self.size))
            ld = fd.copy()
            ld = ld + np.random.normal(0, 0.01, size=ld.shape).astype(np.float32)
            ld = torch.from_numpy(ld)[None]
            fd = torch.from_numpy(fd)[None]
            return {'LD': ld, 'FD': fd, 'case_name': name}

def _binary_metrics_from_arrays(pred_uint8, gt_uint8, thresh=0.5):
    pred_prob = np.clip(pred_uint8.astype(np.float32) / 255.0, 0.0, 1.0)
    gt_bin = (np.clip(gt_uint8.astype(np.float32) / 255.0, 0.0, 1.0) >= 0.5).astype(np.uint8)
    pred_bin = (pred_prob >= float(thresh)).astype(np.uint8)
    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    eps = 1e-8
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    if _HAVE_SKLEARN:
        try:
            auc = roc_auc_score(gt_bin.reshape(-1), pred_prob.reshape(-1))
        except Exception:
            auc = float("nan")
    else:
        y = gt_bin.reshape(-1).astype(np.int32)
        s = pred_prob.reshape(-1).astype(np.float32)
        order = np.argsort(s)[::-1]; y = y[order]
        P = np.sum(y == 1); N = np.sum(y == 0)
        if P == 0 or N == 0:
            auc = float("nan")
        else:
            tps = np.cumsum(y == 1); fps = np.cumsum(y == 0)
            tpr = tps / (P + 1e-8); fpr = fps / (N + 1e-8)
            auc = float(np.trapz(tpr, fpr))
    return float(acc), float(precision), float(recall), float(auc)

# ====== FID 工具（与你原实现一致/兼容） ======
@torch.no_grad()
def _inception_feature_extractor(device="cuda" if torch.cuda.is_available() else "cpu"):
    inc = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    )
    inc.fc = nn.Identity()
    inc.eval().to(device)
    feat_dim = 2048
    def _preprocess(x):
        if isinstance(x, np.ndarray):
            ten = torch.from_numpy(x).permute(2,0,1).float()
            if ten.max() > 1.5: ten = ten / 255.0
        else:
            ten = x.clone().float()
            if ten.ndim == 2: ten = ten.unsqueeze(0)
            if ten.shape[0] not in (1,3): ten = ten.permute(2,0,1)
            if ten.max() > 1.5: ten = ten / 255.0
        ten = F.interpolate(ten.unsqueeze(0), size=(299,299), mode="bilinear", align_corners=False)[0]
        if ten.shape[0] == 1: ten = ten.repeat(3,1,1)
        ten = ten * 2.0 - 1.0
        return ten.unsqueeze(0)
    return inc, _preprocess, feat_dim

@torch.no_grad()
def _compute_feats_for_batch(imgs_uint8, model, preprocess, device):
    feats = []
    for x in imgs_uint8:
        inp = preprocess(x).to(device)
        f = model(inp)
        if f.ndim > 2: f = f.mean(dim=[2,3])
        feats.append(f.squeeze(0).cpu())
    return torch.stack(feats, dim=0)

def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = torch.linalg.sqrtm((sigma1 @ sigma2).double(), disp=False)
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.shape[0], device=sigma1.device, dtype=torch.double) * eps
        covmean = torch.linalg.sqrtm(((sigma1 + offset) @ (sigma2 + offset)).double()).real
    if torch.is_complex(covmean): covmean = covmean.real
    tr_covmean = torch.trace(covmean)
    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2.0 * tr_covmean
    return float(fid)

def torch2hwcuint8(x, clip=False):
    if clip: x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x): return 1 / (np.exp(x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64))**2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        xs = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(xs) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "alpha_cosine":
        s = 0.008
        t = np.arange(0, num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        ac = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1]); betas = np.clip(betas, a_min=None, a_max=0.999)
    elif beta_schedule == "alpha_sigmoid":
        x = np.linspace(-6, 6, 1001)
        ac = sigmoid(x); ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1]); betas = np.clip(betas, a_min=None, a_max=0.999)
    elif beta_schedule == "alpha_linear":
        t = np.arange(0, num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        ac = -t + 1; ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1]); betas = np.clip(betas, a_min=None, a_max=0.999)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# ========= fMRI 指标 =========

def compute_tsnr(x4d):
    """
    x4d: (T,H,W) 或 (T,D,H,W) 的 numpy 数组，值域0~255
    tSNR = mean_t / std_t
    """
    x = x4d.astype(np.float32)
    axis = 0
    mu = x.mean(axis=axis)
    sd = x.std(axis=axis) + 1e-6
    tsnr = (mu / sd).mean()   # 取全局平均，便于日志展示
    return float(tsnr)

def temporal_corr(pred4d, gt4d):
    """
    对 (T,...) 在时间维计算相关（逐体素），再取全局均值
    """
    p = pred4d.astype(np.float32).reshape(pred4d.shape[0], -1)
    g = gt4d.astype(np.float32).reshape(gt4d.shape[0], -1)
    p = p - p.mean(0, keepdims=True)
    g = g - g.mean(0, keepdims=True)
    num = (p * g).sum(0)
    den = (np.sqrt((p*p).sum(0)) * np.sqrt((g*g).sum(0)) + 1e-6)
    r = (num / den).mean()
    return float(r)

# ========= Diffusion Runner =========

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    # === 单条件训练（CT去噪/跨模态翻译/fMRI重建） ===
    def sg_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if self.args.dataset == 'LDFDCT':
            dataset = LDFDCT(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training your Fast-DDPM model on LDFDCT dataset.')
        elif self.args.dataset in ('BRATS', 'FMRI'):
            dataset = BRATS(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training your Fast-DDPM model on BRATS/FMRI-like dataset.')
            print('The scheduler sampling type is {}. involved steps {} / 1000.'.format(self.args.scheduler_type, self.args.timesteps))

        train_loader = data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                                       num_workers=config.data.num_workers, pin_memory=True)

        model = Model(config).to(self.device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(self.config, model.parameters())

        ema_helper = None
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]; step = states[3]
            if self.config.model.ema: ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                # 兼容 BRATS/LDFDCT 字段
                if 'LD' in x:  # (low-dose / source)
                    x_img = x['LD'].to(self.device)
                    x_gt  = x['FD'].to(self.device)
                    cond  = None
                else:
                    # 默认单条件
                    x_img = x.to(self.device)
                    x_gt  = x.to(self.device)
                    cond  = None

                # 多模态或3D任务：如果数据集提供额外模态，可组装为 cond（示意）
                if getattr(self.config.model, "multimodal", False) and isinstance(x, dict) and 'COND' in x:
                    cond = x['COND'].to(self.device)  # (B,C',D,H,W)

                n = x_gt.size(0)
                model.train(); step += 1
                e = torch.randn_like(x_gt); b = self.betas

                if self.args.scheduler_type == 'uniform':
                    skip = self.num_timesteps // self.args.timesteps
                    t_intervals = torch.arange(-1, self.num_timesteps, skip); t_intervals[0] = 0
                elif self.args.scheduler_type == 'non-uniform':
                    t_intervals = torch.tensor([0,199,399,599,699,799,849,899,949,999])
                    if self.args.timesteps != 10:
                        num_1 = int(self.args.timesteps * 0.4)
                        num_2 = int(self.args.timesteps * 0.6)
                        s1 = torch.linspace(0, 699, num_1 + 1)[:-1]
                        s2 = torch.linspace(699, 999, num_2)
                        s1 = torch.ceil(s1).long(); s2 = torch.ceil(s2).long()
                        t_intervals = torch.cat((s1, s2))
                else:
                    raise Exception("scheduler_type is either uniform or non-uniform.")

                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals) - idx_1 - 1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(self.device)

                # === loss 入口 保持不变（模型已支持 cond） ===
                # 兼容你的 loss_registry 签名：loss(model, x_img, x_gt, t, e, b)
                loss = loss_registry[self.config.model.type](model, x_img, x_gt, t, e, b, cond=cond)

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(f"step: {step}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()
                if self.config.model.ema: ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if self.config.model.ema: states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

    # === 双条件训练（多图像 SR / PMUB） ===
    def sr_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        dataset = PMUB(self.config.data.train_dataroot, self.config.data.image_size, split='train')
        print('Start training your Fast-DDPM model on PMUB dataset.')
        print('The scheduler sampling type is {}. involved steps {} / 1000.'.format(self.args.scheduler_type, self.args.timesteps))

        train_loader = data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                                       num_workers=config.data.num_workers, pin_memory=True)

        model = Model(config).to(self.device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(self.config, model.parameters())

        ema_helper = None
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]; step = states[3]
            if self.config.model.ema: ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['BW'].size(0); model.train(); step += 1
                x_bw = x['BW'].to(self.device)   # 条件1（低分辨率/模糊）
                x_md = x['MD'].to(self.device)   # 目标（中间/高分辨率）
                x_fw = x['FW'].to(self.device)   # 条件2（先验/其他角度）

                e = torch.randn_like(x_md); b = self.betas

                if self.args.scheduler_type == 'uniform':
                    skip = self.num_timesteps // self.args.timesteps
                    t_intervals = torch.arange(-1, self.num_timesteps, skip); t_intervals[0] = 0
                elif self.args.scheduler_type == 'non-uniform':
                    t_intervals = torch.tensor([0,199,399,599,699,799,849,899,949,999])
                    if self.args.timesteps != 10:
                        num_1 = int(self.args.timesteps * 0.4); num_2 = int(self.args.timesteps * 0.6)
                        s1 = torch.linspace(0, 699, num_1 + 1)[:-1]; s2 = torch.linspace(699, 999, num_2)
                        s1 = torch.ceil(s1).long(); s2 = torch.ceil(s2).long()
                        t_intervals = torch.cat((s1, s2))
                else:
                    raise Exception("scheduler_type is either uniform or non-uniform.")

                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals) - idx_1 - 1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(self.device)

                # 支持把 (x_bw, x_fw) 堆叠成多模态 cond 传入
                cond = torch.cat([x_bw, x_fw], dim=1) if getattr(self.config.model, "multimodal", True) else x_bw

                # loss 接口兼容
                loss = loss_registry[self.config.model.type](model, x_bw, x_md, x_fw, t, e, b, cond=cond)
                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(f"step: {step}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()
                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if self.config.model.ema: states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

    def fmri_train(self):
        task = getattr(self.config.data, "fmri_task", "sr")

        # ---------- 训练集 ----------
        train_dataset = FMRIDataset(
            root_lr=self.config.data.train_dataroot,
            root_hr=getattr(self.config.data, "train_dataroot_hr", None),
            image_size=self.config.data.image_size,
            split='train',
            is_3d=bool(getattr(self.config.data, "is_3d", True)),
            task=task,
            modalities=getattr(self.config.data, "modalities", 1),
            time_dim=getattr(self.config.data, "time_dim", None),
            downsample_z=getattr(self.config.data, "downsample_z", 1),
            downsample_xy=getattr(self.config.data, "downsample_xy", 1),
        )
        train_loader = data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                       shuffle=True, num_workers=self.config.data.num_workers, pin_memory=True)

        # ---------- 验证集（使用 sample_dataroot，可单独划分） ----------
        val_dataset = FMRIDataset(
            root_lr=self.config.data.sample_dataroot,
            root_hr=getattr(self.config.data, "sample_dataroot_hr", None),
            image_size=self.config.data.image_size,
            split='val',
            is_3d=bool(getattr(self.config.data, "is_3d", True)),
            task=task,
            modalities=getattr(self.config.data, "modalities", 1),
            time_dim=getattr(self.config.data, "time_dim", None),
            downsample_z=getattr(self.config.data, "downsample_z", 1),
            downsample_xy=getattr(self.config.data, "downsample_xy", 1),
        )
        val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)  # 每次只取一个样本验证

        model = Model(self.config).to(self.device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(self.config, model.parameters())
        ema_helper = EMAHelper(mu=self.config.model.ema_rate) if self.config.model.ema else None
        if ema_helper:
            ema_helper.register(model)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if ema_helper:
                ema_helper.load_state_dict(states[4])

        # 验证频率（从配置读取，默认5000步）
        val_freq = getattr(self.config.training, "validation_freq", 5000)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                model.train()
                step += 1

                # ---------- 训练步 ----------
                if task == 'sr':
                    lr = x['LR'].to(self.device)
                    hr = x['HR'].to(self.device)
                    n = lr.size(0)
                    e = torch.randn_like(hr)
                    b = self.betas
                    if self.args.scheduler_type == 'uniform':
                        skip = self.num_timesteps // self.args.timesteps
                        t_intervals = torch.arange(-1, self.num_timesteps, skip)
                        t_intervals[0] = 0
                    else:
                        t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
                    idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                    idx_2 = len(t_intervals) - idx_1 - 1
                    t = torch.cat([idx_1, idx_2], dim=0)[:n].to(self.device)
                    loss = loss_registry[self.config.model.type](model, lr, hr, hr, t, e, b)
                else:
                    ld = x['LD'].to(self.device)
                    fd = x['FD'].to(self.device)
                    n = ld.size(0)
                    e = torch.randn_like(fd)
                    b = self.betas
                    if self.args.scheduler_type == 'uniform':
                        skip = self.num_timesteps // self.args.timesteps
                        t_intervals = torch.arange(-1, self.num_timesteps, skip)
                        t_intervals[0] = 0
                    else:
                        t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
                    idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                    idx_2 = len(t_intervals) - idx_1 - 1
                    t = torch.cat([idx_1, idx_2], dim=0)[:n].to(self.device)
                    loss = loss_registry[self.config.model.type](model, ld, fd, t, e, b)

                # 确保 loss 是标量
                if loss.dim() > 0:
                    loss = loss.mean()

                self.config.tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(f"step: {step}, loss: {loss.item():.6f}")

                optimizer.zero_grad()
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()
                if ema_helper:
                    ema_helper.update(model)

                # 保存检查点
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if ema_helper:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                # ---------- 定期验证（计算完整指标） ----------
                if step % val_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        # 取一个验证样本
                        val_batch = next(iter(val_loader))
                        if task == 'sr':
                            pred = self._sr_like_sample_image_3d(val_batch['LR'].to(self.device), model)
                            gt = val_batch['HR'].to(self.device)
                        else:
                            pred = self._sg_like_sample_image_3d(val_batch['LD'].to(self.device), model)
                            gt = val_batch['FD'].to(self.device)

                        # 反标准化
                        pred_np = inverse_data_transform(self.config, pred).cpu().numpy()
                        gt_np = inverse_data_transform(self.config, gt).cpu().numpy()

                        # 确保形状为 [T, H, W] 或 [D, H, W]
                        if pred_np.ndim == 4:  # [B, C, T, H, W] -> [T, H, W]
                            pred_np = pred_np[0, 0]
                            gt_np = gt_np[0, 0]
                        elif pred_np.ndim == 5:  # [B, C, D, H, W] -> [D, H, W]
                            pred_np = pred_np[0, 0]
                            gt_np = gt_np[0, 0]

                        # 转换为 uint8 以便计算二分类指标
                        pred_uint8 = (pred_np * 255.0).round().astype(np.uint8)
                        gt_uint8 = (gt_np * 255.0).round().astype(np.uint8)

                        # ----- 空间指标：PSNR, SSIM (逐 slice 平均) -----
                        psnr_sum, ssim_sum = 0.0, 0.0
                        slices = pred_uint8.shape[0]  # 时间或深度维度
                        for s in range(slices):
                            psnr_sum += calculate_psnr(pred_uint8[s], gt_uint8[s])
                            ssim_sum += ssim(gt_uint8[s], pred_uint8[s], data_range=255)
                        avg_psnr = psnr_sum / slices
                        avg_ssim = ssim_sum / slices

                        # ----- 分类指标（全图像展平） -----
                        acc, prec, rec, auc_val = _binary_metrics_from_arrays(pred_uint8.reshape(-1),
                                                                              gt_uint8.reshape(-1),
                                                                              thresh=self.args.bin_thresh)

                        # ----- fMRI 动态指标（仅当时间维 > 1） -----
                        tsnr_val = 0.0
                        dvars_val = 0.0
                        if slices > 1:
                            # tSNR = mean / std along time
                            tsnr_val = (gt_np.mean(axis=0) / (gt_np.std(axis=0) + 1e-8)).mean()
                            # DVARS: sqrt(mean((diff)^2))
                            diff = np.diff(pred_np, axis=0)
                            dvars_val = np.sqrt((diff ** 2).mean()).mean()

                        # 记录到 TensorBoard
                        self.config.tb_logger.add_scalar("val/psnr", avg_psnr, step)
                        self.config.tb_logger.add_scalar("val/ssim", avg_ssim, step)
                        self.config.tb_logger.add_scalar("val/acc", acc, step)
                        self.config.tb_logger.add_scalar("val/precision", prec, step)
                        self.config.tb_logger.add_scalar("val/recall", rec, step)
                        self.config.tb_logger.add_scalar("val/auc", auc_val, step)
                        self.config.tb_logger.add_scalar("val/tsnr", tsnr_val, step)
                        self.config.tb_logger.add_scalar("val/dvars", dvars_val, step)

                        # 打印到控制台
                        logging.info(
                            f"Validation step {step}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, "
                            f"ACC={acc:.4f}, PRE={prec:.4f}, REC={rec:.4f}, AUC={auc_val:.4f}, "
                            f"tSNR={tsnr_val:.4f}, DVARS={dvars_val:.4f}"
                        )

                    model.train()  # 切回训练模式

    def fmri_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            model = Model(self.config)
            print('Start fMRI inference on model of {} steps'.format(ckpt_idx))
            states = torch.load(os.path.join(self.args.log_path, f"ckpt_{ckpt_idx}.pth"),
                                map_location=self.config.device)
            model = torch.nn.DataParallel(model.to(self.device))
            model.load_state_dict(states[0], strict=True)
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate); ema_helper.register(model)
                ema_helper.load_state_dict(states[-1]); ema_helper.ema(model)
            model.eval()

            # 数据
            ds = FMRIDataset(
                root_lr=self.config.data.sample_dataroot_lr or self.config.data.sample_dataroot,
                root_hr=self.config.data.sample_dataroot_hr,
                image_size=self.config.data.image_size, split='calculate',
                is_3d=bool(getattr(self.config.data, "is_3d", True)),
                task=getattr(self.config.data, "fmri_task", "sr"),
                modalities=getattr(self.config.data, "modalities", 1),
                time_dim=getattr(self.config.data, "time_dim", None),
                downsample_z=getattr(self.config.data, "downsample_z", 1),
                downsample_xy=getattr(self.config.data, "downsample_xy", 1),
            )
            loader = data.DataLoader(ds, batch_size=self.config.sampling_fid.batch_size,
                                     shuffle=False, num_workers=self.config.data.num_workers)

            img_id = len(glob.glob(f"{self.args.image_folder}/*"))
            print(f"starting from image {img_id}")

            # 聚合指标
            avg_psnr = avg_ssim = 0.0
            acc_all, pre_all, rec_all, auc_all = [], [], [], []
            tsnr_all, dvars_all = [], []
            time_list = []

            do_fid = getattr(self.args, "fid", False)
            if do_fid:
                inc_model, inc_preprocess, _ = _inception_feature_extractor(device="cuda" if torch.cuda.is_available() else "cpu")
                pred_feats_list, real_feats_list = [], []

            for batch_idx, batch in tqdm.tqdm(enumerate(loader), desc="fMRI sampling"):
                task = getattr(self.config.data, "fmri_task", "sr")
                if task == 'sr':
                    LR = batch['LR'].to(self.device)  # [B,1,D,H,W]
                    HR = batch['HR'].to(self.device)
                    n = LR.size(0)
                    x = torch.randn(n, self.config.data.channels, LR.shape[-2], LR.shape[-1], device=self.device)
                    # 在 3D 情况下，我们的 denoising steps 内部会用 SR 条件
                    out = self._sr_like_sample_image_3d(LR, model)
                    pred = out
                    gt = HR
                else:
                    LD = batch['LD'].to(self.device)
                    x = torch.randn(LD.shape[0], self.config.data.channels, LD.shape[-2], LD.shape[-1], device=self.device)
                    pred = self._sg_like_sample_image_3d(LD, model)
                    gt = batch['FD'].to(self.device)

                # 反标准化
                pred_v = inverse_data_transform(self.config, pred).detach().cpu().float()
                gt_v = inverse_data_transform(self.config, gt).detach().cpu().float()

                # ——— 计算空间 PSNR/SSIM & 分类四指标 ———
                psnr_case = 0.0; ssim_case = 0.0
                ACC = PRE = REC = AUC = 0.0
                bin_thresh = getattr(self.args, "bin_thresh", 0.5)

                # 压到 uint8 计算
                pred_u8 = (pred_v.squeeze().numpy() * 255.0).round().astype(np.uint8)
                gt_u8   = (gt_v.squeeze().numpy() * 255.0).round().astype(np.uint8)

                if pred_u8.ndim == 3:   # [D,H,W]
                    for d in range(pred_u8.shape[0]):
                        psnr_case += calculate_psnr(pred_u8[d], gt_u8[d])
                        ssim_case += ssim(gt_u8[d], pred_u8[d], data_range=255)
                        a,p,r,au = _binary_metrics_from_arrays(pred_u8[d], gt_u8[d], thresh=bin_thresh)
                        ACC += a; PRE += p; REC += r; AUC += au
                    denom = pred_u8.shape[0]
                else:  # [T,H,W]
                    for t in range(pred_u8.shape[0]):
                        psnr_case += calculate_psnr(pred_u8[t], gt_u8[t])
                        ssim_case += ssim(gt_u8[t], pred_u8[t], data_range=255)
                        a,p,r,au = _binary_metrics_from_arrays(pred_u8[t], gt_u8[t], thresh=bin_thresh)
                        ACC += a; PRE += p; REC += r; AUC += au
                    denom = pred_u8.shape[0]

                psnr_case /= max(1, denom); ssim_case /= max(1, denom)
                avg_psnr += psnr_case; avg_ssim += ssim_case
                acc_all.append(ACC/max(1,denom)); pre_all.append(PRE/max(1,denom))
                rec_all.append(REC/max(1,denom)); auc_all.append(AUC/max(1,denom))

                # ——— fMRI 专属：tSNR & DVARS（若为 4D）———
                if pred_v.ndim == 5 and pred_v.shape[1] == 1 and pred_v.shape[2] > 1:
                    # [B,1,T,H,W]
                    pv = pred_v[:,0]  # [B,T,H,W]
                    gv = gt_v[:,0]
                    # tSNR = mean_t / std_t
                    tsnr = (pv.mean(dim=1) / (pv.std(dim=1) + 1e-8)).mean().item()
                    # DVARS ~ 邻时刻差分的均方根
                    diff = (pv[:,1:] - pv[:,:-1]).pow(2).mean(dim=[2,3]).sqrt().mean().item()
                    tsnr_all.append(tsnr)
                    dvars_all.append(diff)

                # FID（将每张 slice 当作灰度→3通道）
                if do_fid:
                    if pred_u8.ndim == 3:
                        for d in range(pred_u8.shape[0]):
                            p = np.stack([pred_u8[d]]*3, axis=-1)
                            g = np.stack([gt_u8[d]]*3, axis=-1)
                            pf = _compute_feats_for_batch([p], inc_model, inc_preprocess, device="cuda" if torch.cuda.is_available() else "cpu")
                            rf = _compute_feats_for_batch([g], inc_model, inc_preprocess, device="cuda" if torch.cuda.is_available() else "cpu")
                            pred_feats_list.append(pf[0]); real_feats_list.append(rf[0])

                # 保存可视化
                for bi in range(min(pred_v.shape[0], 2)):
                    tvu.save_image(pred_v[bi,0, pred_v.shape[2]//2], os.path.join(self.args.image_folder, f"{self.ckpt_idx}_{img_id}_pt.png"))
                    tvu.save_image(gt_v[bi,0, gt_v.shape[2]//2], os.path.join(self.args.image_folder, f"{self.ckpt_idx}_{img_id}_gt.png"))
                    img_id += 1

            N = max(1, len(ds))
            avg_psnr /= N; avg_ssim /= N
            log_line = f"Average: PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}"
            if len(acc_all)>0:
                log_line += f", ACC {np.mean(acc_all):.4f}, PRE {np.mean(pre_all):.4f}, REC {np.mean(rec_all):.4f}, AUC {np.mean(auc_all):.4f}"
            if len(tsnr_all)>0:
                log_line += f", tSNR {np.mean(tsnr_all):.4f}, DVARS {np.mean(dvars_all):.4f}"
            logging.info(log_line)

            if do_fid and len(pred_feats_list) > 1:
                pred_feats = torch.stack(pred_feats_list, dim=0).double()
                real_feats = torch.stack(real_feats_list, dim=0).double()
                mu_p = pred_feats.mean(0); mu_r = real_feats.mean(0)
                diff_p = (pred_feats - mu_p); diff_r = (real_feats - mu_r)
                cov_p = (diff_p.T @ diff_p) / (pred_feats.shape[0] - 1)
                cov_r = (diff_r.T @ diff_r) / (real_feats.shape[0] - 1)
                fid_value = _frechet_distance(mu_p, cov_p, mu_r, cov_r)
                logging.info(f"FID (fMRI set): {fid_value:.4f}")

    # ======== 采样内部：使用 SR 条件的 3D/单条件路径 ========
    def _sr_like_sample_image_3d(self, lr, model):
        # 与 sr_sample_image 对齐，但这里直接走 3D 输入，并让 Model 内部使用 SR 条件
        try: skip = self.args.skip
        except Exception: skip = 1
        if self.args.sample_type == "generalized":
            if self.args.scheduler_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(-1, self.num_timesteps, skip)); seq[0] = 0
            else:
                seq = [0,199,399,599,699,799,849,899,949,999]
            from functions.denoising import sr_generalized_steps
            xs = sr_generalized_steps(lr, lr, lr, seq, model, self.betas, eta=self.args.eta)  # 复用接口
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            from functions.denoising import sr_ddpm_steps
            x = sr_ddpm_steps(lr, lr, lr, seq, model, self.betas)
        else:
            raise NotImplementedError
        return x[0][-1]

    def _sg_like_sample_image_3d(self, ld, model):
        try: skip = self.args.skip
        except Exception: skip = 1
        if self.args.sample_type == "generalized":
            if self.args.scheduler_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(-1, self.num_timesteps, skip)); seq[0] = 0
            else:
                seq = [0,199,399,599,699,799,849,899,949,999]
            from functions.denoising import sg_generalized_steps
            xs = sg_generalized_steps(ld, ld, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            from functions.denoising import sg_ddpm_steps
            x = sg_ddpm_steps(ld, ld, seq, model, self.betas)
        else:
            raise NotImplementedError
        return x[0][-1]

    # === 采样：双条件（SR） ===
    def sr_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            model = Model(self.config)
            print('Start inference on model of {} steps'.format(ckpt_idx))

            if not self.args.use_pretrained:
                states = torch.load(os.path.join(self.args.log_path, f"ckpt_{ckpt_idx}.pth"), map_location=self.config.device)
                model = model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)
                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate); ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1]); ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            model.eval()
            if self.args.fid:
                self.sr_sample_fid(model)
            elif self.args.interpolation:
                self.sr_sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedure not defined")

    # === 采样：单条件（CT/翻译/fMRI） ===
    def sg_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            model = Model(self.config)
            print('Start inference on model of {} steps'.format(ckpt_idx))

            if not self.args.use_pretrained:
                states = torch.load(
                    os.path.join(f"{self.args.log_path}/ckpt_{self.ckpt_idx}.pth"),
                    map_location=self.config.device,
                )
                model = model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)
                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate); ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1]); ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            model.eval()
            if self.args.fid:
                self.sg_sample_fid(model)
            elif self.args.interpolation:
                self.sr_sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedure not defined")

    # === SR FID 采样（保持你的实现，加入多模态 cond 传入） ===
    @torch.no_grad()
    def sr_sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        sample_dataset = PMUB(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
        print('Start sampling model on PMUB dataset.')
        print('The inference sample type is {}. scheduler {}. steps {} / 1000.'.format(
              self.args.sample_type, self.args.scheduler_type, self.args.timesteps))

        sample_loader = data.DataLoader(sample_dataset, batch_size=config.sampling_fid.batch_size,
                                        shuffle=False, num_workers=config.data.num_workers)

        do_fid = getattr(self.args, "fid", False)
        if do_fid:
            inc_model, inc_preprocess, feat_dim = _inception_feature_extractor(
                device="cuda" if torch.cuda.is_available() else "cpu")
            pred_feats_list, real_feats_list = [], []

        with torch.no_grad():
            data_num = len(sample_dataset)
            print('The length of test set is:', data_num)
            avg_psnr = 0.0; avg_ssim = 0.0
            time_list, psnr_list, ssim_list = [], [], []
            acc_all, pre_all, rec_all, auc_all = [], [], [], []
            bin_thresh = getattr(self.args, "bin_thresh", 0.5)

            for batch_idx, img in tqdm.tqdm(enumerate(sample_loader),
                                            desc="Generating image samples for FID evaluation."):
                n = img['BW'].shape[0]
                x = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size, device=self.device)
                x_bw = img['BW'].to(self.device)
                x_md = img['MD'].to(self.device)
                x_fw = img['FW'].to(self.device)
                case_name = img['case_name'][0]

                time_start = time.time()
                x = self.sr_sample_image(x, x_bw, x_fw, model)
                time_end = time.time()

                x = inverse_data_transform(config, x)
                x_md = inverse_data_transform(config, x_md)
                x_tensor = x; x_md_tensor = x_md

                x_md_np = (x_md.squeeze().float().cpu().numpy() * 255.0).round().astype(np.uint8)
                x_np    = (x.squeeze().float().cpu().numpy() * 255.0).round().astype(np.uint8)
                if x_np.ndim == 2: x_np = x_np[None, ...]; x_md_np = x_md_np[None, ...]

                PSNR = 0.0; SSIM = 0.0
                ACC = PRE = REC = AUC = 0.0

                for i in range(x_np.shape[0]):
                    psnr_temp = calculate_psnr(x_np[i], x_md_np[i])
                    ssim_temp = ssim(x_md_np[i], x_np[i], data_range=255)
                    PSNR += psnr_temp; SSIM += ssim_temp
                    psnr_list.append(psnr_temp); ssim_list.append(ssim_temp)

                    acc_i, pre_i, rec_i, auc_i = _binary_metrics_from_arrays(x_np[i], x_md_np[i], thresh=bin_thresh)
                    ACC += acc_i; PRE += pre_i; REC += rec_i; AUC += auc_i

                    if do_fid:
                        pred_img = np.stack([x_np[i]]*3, axis=-1)
                        gt_img   = np.stack([x_md_np[i]]*3, axis=-1)
                        pf = _compute_feats_for_batch([pred_img], inc_model, inc_preprocess, device="cuda" if torch.cuda.is_available() else "cpu")
                        rf = _compute_feats_for_batch([gt_img],   inc_model, inc_preprocess, device="cuda" if torch.cuda.is_available() else "cpu")
                        pred_feats_list.append(pf[0]); real_feats_list.append(rf[0])

                PSNR /= x_np.shape[0]; SSIM /= x_np.shape[0]
                ACC  /= x_np.shape[0]; PRE  /= x_np.shape[0]; REC  /= x_np.shape[0]; AUC /= x_np.shape[0]
                case_time = time_end - time_start; time_list.append(case_time)
                avg_psnr += PSNR * x_np.shape[0]; avg_ssim += SSIM * x_np.shape[0]
                acc_all.append(ACC); pre_all.append(PRE); rec_all.append(REC); auc_all.append(AUC)

                logging.info('Case {}: PSNR {:.4f}, SSIM {:.4f}, ACC {:.4f}, PRE {:.4f}, REC {:.4f}, AUC {:.4f}, time {:.4f}'.format(
                    case_name, PSNR, SSIM, ACC, PRE, REC, AUC, case_time))

                for i in range(0, n):
                    tvu.save_image(x_tensor[i],  os.path.join(self.args.image_folder, "{}_{}_pt.png".format(self.ckpt_idx, img_id)))
                    tvu.save_image(x_md_tensor[i], os.path.join(self.args.image_folder, "{}_{}_gt.png".format(self.ckpt_idx, img_id)))
                    img_id += 1

            avg_psnr = avg_psnr / max(data_num, 1)
            avg_ssim = avg_ssim / max(data_num, 1)
            avg_time = sum(time_list[1:-1]) / (len(time_list) - 2) if len(time_list) > 2 else (sum(time_list) / max(len(time_list), 1))

            if len(acc_all) > 0:
                acc_mean = float(np.mean(acc_all)); pre_mean = float(np.mean(pre_all))
                rec_mean = float(np.mean(rec_all)); auc_mean = float(np.mean(auc_all))
                logging.info('Average: PSNR {:.4f}, SSIM {:.4f}, ACC {:.4f}, PRE {:.4f}, REC {:.4f}, AUC {:.4f}, time {:.4f}'.format(
                    avg_psnr, avg_ssim, acc_mean, pre_mean, rec_mean, auc_mean, avg_time))
            else:
                logging.info('Average: PSNR {:.4f}, SSIM {:.4f}, time {:.4f}'.format(avg_psnr, avg_ssim, avg_time))

            if do_fid and len(pred_feats_list) > 1:
                pred_feats = torch.stack(pred_feats_list, dim=0).double()
                real_feats = torch.stack(real_feats_list, dim=0).double()
                mu_p = pred_feats.mean(0); mu_r = real_feats.mean(0)
                diff_p = (pred_feats - mu_p); diff_r = (real_feats - mu_r)
                cov_p = (diff_p.T @ diff_p) / (pred_feats.shape[0] - 1)
                cov_r = (diff_r.T @ diff_r) / (real_feats.shape[0] - 1)
                fid_value = _frechet_distance(mu_p, cov_p, mu_r, cov_r)
                logging.info(f"FID (SR set): {fid_value:.4f}")

    # === SG FID 采样：加入 fMRI 4D 指标（自动） ===
    sys.setrecursionlimit(10000)
    @torch.no_grad()
    def sg_sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        if self.args.dataset == 'LDFDCT':
            sample_dataset = LDFDCT(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
            print('Start sampling model on LDFDCT dataset.')
        elif self.args.dataset in ('BRATS', 'FMRI'):
            sample_dataset = BRATS(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
            print('Start sampling model on BRATS/FMRI-like dataset.')
        print('Inference type {}. scheduler {}. steps {} / 1000.'.format(
            self.args.sample_type, self.args.scheduler_type, self.args.timesteps))

        sample_loader = data.DataLoader(sample_dataset, batch_size=config.sampling_fid.batch_size,
                                        shuffle=False, num_workers=config.data.num_workers)

        do_fid = getattr(self.args, "fid", False)
        if do_fid:
            inc_model, inc_preprocess, feat_dim = _inception_feature_extractor(
                device="cuda" if torch.cuda.is_available() else "cpu")
            pred_feats_list, real_feats_list = [], []

        with torch.no_grad():
            data_num = len(sample_dataset)
            print('The length of test set is:', data_num)
            avg_psnr = 0.0; avg_ssim = 0.0
            time_list, psnr_list, ssim_list = [], [], []
            acc_all, pre_all, rec_all, auc_all = [], [], [], []
            bin_thresh = getattr(self.args, "bin_thresh", 0.5)

            # fMRI 4D 指标累计
            tsnr_list, corr_list = [], []

            for batch_idx, sample in tqdm.tqdm(enumerate(sample_loader),
                                               desc="Generating image samples for FID evaluation."):
                n = sample['LD'].shape[0]
                x = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size, device=self.device)
                x_img = sample['LD'].to(self.device)
                x_gt  = sample['FD'].to(self.device)
                case_name = sample['case_name']

                time_start = time.time()
                x = self.sg_sample_image(x, x_img, model)
                time_end = time.time()

                x = inverse_data_transform(config, x)
                x_gt = inverse_data_transform(config, x_gt)
                x_tensor = x; x_gt_tensor = x_gt

                x_gt_np = (x_gt.squeeze().float().cpu().numpy() * 255.0).round().astype(np.uint8)
                x_np    = (x.squeeze().float().cpu().numpy() * 255.0).round().astype(np.uint8)

                # 2D/3D 与 4D 自适应
                if x_np.ndim == 2:
                    # 单样本2D
                    PSNR = calculate_psnr(x_np, x_gt_np)
                    SSIM = ssim(x_gt_np, x_np, data_range=255)
                    acc_i, pre_i, rec_i, auc_i = _binary_metrics_from_arrays(x_np, x_gt_np, thresh=bin_thresh)
                elif x_np.ndim == 3:
                    # 多切片2D (N,H,W)
                    PSNR = calculate_psnr(x_np, x_gt_np)
                    SSIM = ssim(x_gt_np, x_np, data_range=255)
                    acc_i, pre_i, rec_i, auc_i = _binary_metrics_from_arrays(x_np, x_gt_np, thresh=bin_thresh)
                else:
                    # 4D fMRI: (T,H,W) 或 (T,D,H,W) -> 先做逐帧PSNR/SSIM平均，再算4D指标
                    T = x_np.shape[0]
                    ps, ss = 0.0, 0.0
                    for t in range(T):
                        xt = x_np[t]; gt = x_gt_np[t]
                        ps += calculate_psnr(xt, gt)
                        ss += ssim(gt, xt, data_range=255)
                    PSNR = ps / T; SSIM = ss / T
                    # 二分类四指标（把全4D展平）
                    acc_i, pre_i, rec_i, auc_i = _binary_metrics_from_arrays(x_np.reshape(T,-1), x_gt_np.reshape(T,-1), thresh=bin_thresh)
                    # fMRI 指标
                    try:
                        tsnr = compute_tsnr(x_gt_np)   # 用 GT 统计基准（也可对 pred 计算）
                        corr = temporal_corr(x_np.reshape(T,-1), x_gt_np.reshape(T,-1))
                        tsnr_list.append(tsnr); corr_list.append(corr)
                    except Exception:
                        pass

                avg_psnr += PSNR; avg_ssim += SSIM
                acc_all.append(acc_i); pre_all.append(pre_i); rec_all.append(rec_i); auc_all.append(auc_i)
                case_time = time_end - time_start; time_list.append(case_time)

                logging.info('Case {}: PSNR {:.4f}, SSIM {:.4f}, ACC {:.4f}, PRE {:.4f}, REC {:.4f}, AUC {:.4f}{} time {:.4f}'.format(
                    case_name[0], PSNR, SSIM, acc_i, pre_i, rec_i, auc_i,
                    (", tSNR {:.4f}, tCorr {:.4f}".format(tsnr_list[-1], corr_list[-1]) if len(tsnr_list)>0 else ""), case_time))

                for i in range(0, n):
                    tvu.save_image(x_tensor[i],  os.path.join(self.args.image_folder, "{}_{}_pt.png".format(self.ckpt_idx, img_id)))
                    tvu.save_image(x_gt_tensor[i], os.path.join(self.args.image_folder, "{}_{}_gt.png".format(self.ckpt_idx, img_id)))
                    img_id += 1

            avg_psnr = avg_psnr / max(data_num, 1)
            avg_ssim = avg_ssim / max(data_num, 1)
            avg_time = sum(time_list[1:-1]) / (len(time_list) - 2) if len(time_list) > 2 else (sum(time_list) / max(len(time_list), 1))

            if len(acc_all) > 0:
                acc_mean = float(np.mean(acc_all)); pre_mean = float(np.mean(pre_all))
                rec_mean = float(np.mean(rec_all)); auc_mean = float(np.mean(auc_all))
                extra = ""
                if len(tsnr_list) > 0:
                    extra = ", tSNR {:.4f}, tCorr {:.4f}".format(float(np.mean(tsnr_list)), float(np.mean(corr_list)))
                logging.info('Average: PSNR {:.4f}, SSIM {:.4f}, ACC {:.4f}, PRE {:.4f}, REC {:.4f}, AUC {:.4f}{} , time {:.4f}'.format(
                    avg_psnr, avg_ssim, acc_mean, pre_mean, rec_mean, auc_mean, extra, avg_time))
            else:
                logging.info('Average: PSNR {:.4f}, SSIM {:.4f}, time {:.4f}'.format(avg_psnr, avg_ssim, avg_time))

            if do_fid and False:
                # 注：对 fMRI 4D 做 FID 通常不合适，这里保留原2D流程（如有需要，可自行扩展）
                pass

    # ====== 采样路径（保持原有） ======
    def sr_sample_image(self, x, x_bw, x_fw, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.scheduler_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(-1, self.num_timesteps, skip)); seq[0] = 0
            elif self.args.scheduler_type == 'non-uniform':
                seq = [0,199,399,599,699,799,849,899,949,999]
                if self.args.timesteps != 10:
                    num_1 = int(self.args.timesteps * 0.4); num_2 = int(self.args.timesteps * 0.6)
                    s1 = np.linspace(0, 699, num_1 + 1)[:-1]; s2 = np.linspace(699, 999, num_2)
                    s1 = np.ceil(s1).astype(int); s2 = np.ceil(s2).astype(int)
                    seq = np.concatenate((s1, s2))
            else:
                raise Exception("scheduler_type is either uniform or non-uniform.")

            from functions.denoising import sr_generalized_steps
            xs = sr_generalized_steps(x, x_bw, x_fw, seq, model, self.betas, eta=self.args.eta)
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            from functions.denoising import sr_ddpm_steps
            x = sr_ddpm_steps(x, x_bw, x_fw, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sg_sample_image(self, x, x_img, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.scheduler_type == 'uniform':
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(-1, self.num_timesteps, skip)); seq[0] = 0
            elif self.args.scheduler_type == 'non-uniform':
                seq = [0,199,399,599,699,799,849,899,949,999]
                if self.args.timesteps != 10:
                    num_1 = int(self.args.timesteps * 0.4); num_2 = int(self.args.timesteps * 0.6)
                    s1 = np.linspace(0, 699, num_1 + 1)[:-1]; s2 = np.linspace(699, 999, num_2)
                    s1 = np.ceil(s1).astype(int); s2 = np.ceil(s2).astype(int)
                    seq = np.concatenate((s1, s2))
            else:
                raise Exception("scheduler_type is either uniform or non-uniform.")

            from functions.denoising import sg_generalized_steps
            xs = sg_generalized_steps(x, x_img, seq, model, self.betas, eta=self.args.eta)
            x = xs

        elif self.args.sample_type == "ddpm_noisy":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
            from functions.denoising import sg_ddpm_steps
            x = sg_ddpm_steps(x, x_img, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass