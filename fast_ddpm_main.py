import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import time
import stat
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except PermissionError:
        time.sleep(0.25)
        func(path)

def safe_rmtree(path, max_retries=20, retry_sleep=0.25):
    if not os.path.exists(path):
        return
    base = path
    try:
        ts = int(time.time())
        tmp = f"{path}.deleting_{ts}"
        os.rename(path, tmp)
        base = tmp
    except Exception:
        pass
    for _ in range(max_retries):
        try:
            shutil.rmtree(base, onerror=_handle_remove_readonly)
            return
        except PermissionError:
            time.sleep(retry_sleep)
    for root, dirs, files in os.walk(base, topdown=False):
        for name in files:
            fp = os.path.join(root, name)
            try:
                os.chmod(fp, stat.S_IWRITE)
                os.remove(fp)
            except Exception:
                pass
        for name in dirs:
            dp = os.path.join(root, name)
            try:
                os.rmdir(dp)
            except Exception:
                pass
    try:
        os.rmdir(base)
    except Exception:
        print(f"[warn] failed to fully remove: {base}, continue.")

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # === 关键路径参数 ===
    parser.add_argument("--config", type=str, default="fmri_sr.yml", help="Path to the config file")
    parser.add_argument("--dataset", type=str, default="BRATS", help="Name of dataset (LDFDCT, BRATS, PMUB, FMRI)")
    parser.add_argument("--seed", type=int, default=1244, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default="Fast-DDPM_experiments_run_1759570941", help="Log folder name")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")

    # === 运行模式 ===
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("--resume", action="store_true", help="Alias of --resume_training")

    # === I/O ===
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing log/image folders without prompt")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="The folder name of samples")
    parser.add_argument("--ni", action="store_false", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_pretrained", action="store_true")

    # === 采样/调度 ===
    parser.add_argument("--sample_type", type=str, default="generalized", help="generalized or ddpm_noisy")
    parser.add_argument("--scheduler_type", type=str, default="uniform", help="uniform or non-uniform")
    parser.add_argument("--timesteps", type=int, default=10, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta controls variance of sigma")
    parser.add_argument("--sequence", action="store_true")

    # === 指标：保留原有 + 新增二分类/医学指标 ===
    parser.add_argument("--bin_thresh", type=float, default=0.5, help="Binary threshold for ACC/Precision/Recall metrics")

    # === fMRI/3D 相关开关 ===
    parser.add_argument("--enable_3d", action="store_true", help="Use 3D conv UNet & SRUNet path")
    parser.add_argument("--enable_temporal", action="store_true", help="Enable temporal attention (4D fMRI)")
    parser.add_argument("--multimodal", action="store_true", help="Enable multi-modal fusion for inputs")

    # === 日志等级 ===
    parser.add_argument("--verbose", type=str, default="info", help="info | debug | warning | critical")

    args = parser.parse_args()
    if hasattr(args, "resume") and args.resume:
        args.resume_training = True

    # 日志与TB路径
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    # 解析配置
    with open(os.path.join("configs", args.config), "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # 训练模式：准备目录/记录器
    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if getattr(args, "overwrite", False) or args.ni:
                    overwrite = True
                else:
                    resp = input("Folder already exists. Overwrite? (Y/N)")
                    overwrite = (resp.upper() == "Y")
                if overwrite:
                    safe_rmtree(args.log_path)
                    safe_rmtree(tb_path)
                    os.makedirs(args.log_path, exist_ok=True)
                    if os.path.exists(tb_path):
                        safe_rmtree(tb_path)
                    if os.path.exists(args.log_path) and os.listdir(args.log_path):
                        ts = int(time.time())
                        args.log_path = args.log_path + f"_run_{ts}"
                        os.makedirs(args.log_path, exist_ok=True)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path, exist_ok=True)
            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)

        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.handlers = []
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)
    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))
        handler1 = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.handlers = []
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            if args.fid:
                args.image_folder = os.path.join(args.exp, "image_samples", args.doc, "images_fid")
            if args.interpolation:
                args.image_folder = os.path.join(args.exp, "image_samples", args.doc, "images_interpolation")

            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder, exist_ok=True)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if getattr(args, "overwrite", False) or args.ni:
                        overwrite = True
                    else:
                        resp = input(f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)")
                        overwrite = (resp.upper() == "Y")
                    if overwrite:
                        safe_rmtree(args.image_folder)
                        os.makedirs(args.image_folder, exist_ok=True)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # 把开关同步进 config，模型里直接读 config.model.*
    if not hasattr(new_config, "model"):
        new_config.model = argparse.Namespace()
    new_config.model.enable_3d = bool(args.enable_3d)
    new_config.model.enable_temporal = bool(args.enable_temporal)
    new_config.model.multimodal = bool(args.multimodal)

    return args, new_config

def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        if args.sample:
            if args.dataset == 'PMUB':
                runner.sr_sample()
            elif args.dataset == 'FMRI':
                runner.fmri_sample()  # 专门用于 fMRI 的采样
            elif args.dataset in ('LDFDCT', 'BRATS'):
                runner.sg_sample()
            else:
                raise Exception("Supported sampling datasets: LDFDCT, BRATS, PMUB, FMRI")
        else:
            if args.dataset == 'PMUB':
                runner.sr_train()
            elif args.dataset == 'FMRI':
                runner.fmri_train()  # 专门用于 fMRI 的训练
            elif args.dataset in ('LDFDCT', 'BRATS'):
                runner.sg_train()
            else:
                raise Exception("Supported training datasets: LDFDCT, BRATS, PMUB, FMRI")
        '''
        if args.sample:
                    if args.dataset == 'PMUB':
                        runner.sr_sample()
                    elif args.dataset in ('LDFDCT', 'BRATS', 'FMRI'):
                        runner.sg_sample()
                    else:
                        raise Exception("Supported sampling datasets: LDFDCT, BRATS, PMUB, FMRI")
                elif args.test:
                    runner.test()
                else:
                    if args.dataset == 'PMUB':
                        runner.sr_train()
                    elif args.dataset in ('LDFDCT', 'BRATS', 'FMRI'):
                        runner.sg_train()
                    else:
                        raise Exception("Supported training datasets: LDFDCT, BRATS, PMUB, FMRI")
        '''
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())