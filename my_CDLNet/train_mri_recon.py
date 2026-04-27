import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import math
import wandb
import random

from transforms import gaussian_blur_complex
from functools import partial
from model import CDLNet, LPDSNet, LPDSNetBase
from train_utils import awgn
from kspace_data import get_fit_loaders
from mri_utils import mri_encoding, mri_decoding, make_acc_mask, mri_awgn, ifftc
from mask import ssdu_mask_subsample
from utils import saveimg
from filter_utils import get_B_complex, complex_to_rgb_grid
from nle import whiten 

_mask_cache = {}

def get_mask(image, R, acs_lines):
    """
    Returns a cached acceleration mask on the correct device.
    Cache key is based on spatial shape, accel, ACS, and device.
    """
    Ny, Nx = image.shape[-2], image.shape[-1]
    key = (Ny, Nx, R, acs_lines, image.device)

    if key not in _mask_cache:
        _mask_cache[key] = make_acc_mask(
            shape=(Ny, Nx),
            accel=R,
            acs_lines=acs_lines
        ).to(image.device, non_blocking=True)

    return _mask_cache[key]

def main(args):
    """ Given argument dictionary, load data, initialize model, and fit model.
    """
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")

    model_args, train_args, paths, mri_args = [args[item] for item in ['model','train','paths', 'mri']]
    loaders = get_fit_loaders(**train_args['loaders'])
    net, opt, sched, epoch0 = init_model(args, device=device)
    
    if train_args['fit']['x_init'] is True:
        wandb.init(project="immap2p5_R"+str(mri_args['R'])+"_mri_reconstruction", config=args)
    else:
        wandb.init(project="e2e_R"+str(mri_args['R'])+"_mri_reconstruction", config=args)
    wandb.watch(net, log="gradients", log_freq=2000)

    fit(net, 
        opt, 
        loaders,
        sched       = sched,
        save_dir    = paths['save'],
        start_epoch = epoch0 + 1,
        device      = device,
        **train_args['fit'],
        **mri_args,
        epoch_fun = lambda epoch_num: save_args(args, epoch_num))

def fit(net, opt, loaders,
        sched=None,
        epochs=1,
        device=torch.device("cpu"),
        save_dir=None,
        start_epoch=1,
        clip_grad=1,
        noise_std=25,
        image_noise_std=0,
        blur = False,
        pix_width=(1, 21), # A tuple for determining the widths of random blurring operators
        loss_type = 'complex-mse',
        demosaic=False,
        verbose=True,
        val_freq=1,
        save_freq=1,
        epoch_fun=None,
        mcsure=False,
        noise_dist='uniform',
        noise_log_base=10,
        x_init=False,
        sigma_scaling=False,
        e2enet_args_path=None,
        denoiser_args_path = None,
        init_type = None,
        ssdu_masking = False,
        ssdu_acs = 10,
        ssdu_base_accel = 2,
        ssdu_rho = (0.2, 0.8), 
        R=8,
        acs_lines=24,
        kspace_type = 'measurement',
        whiten_kspace = False,
        backtrack_thresh=1,
        log_every=50,
        max_steps=None,
        val_every=2000,
        save_every=5000):

    print(f"fit: using device {device}")

    if not isinstance(noise_std, (list, tuple)):
        noise_std = (noise_std, noise_std)

    if max_steps is None:
        max_steps = epochs * len(loaders["train"])

    ckpt_path = os.path.join(save_dir, '0.ckpt')
    save_ckpt(ckpt_path, net, 0, opt, sched)

    top_psnr = {"val":0}

    step = start_epoch - 1

    train_loader = loaders["train"]
    train_iter = iter(train_loader)

    pbar = tqdm(total=max_steps, initial=step, dynamic_ncols=True, desc="TRAIN")

    # GPU running statistics (no sync)
    loss_sum = torch.zeros((), device=device)
    mse_sum  = torch.zeros((), device=device)
    window_count = 0

    # We can try using an end to end network to initialize our immap2p5 iterations
    # Load our e2enet
    nl1_nl2_loss = lambda x, y: ( ((x - y).abs() + 1e-8).mean((1,2,3)) / (x.abs() + 1e-8).mean((1,2,3)) ).mean() + ( (x - y).pow(2).mean((1,2,3)) / (x.pow(2).mean((1,2,3))+1e-8) ).sqrt().mean()
    if e2enet_args_path and init_type == 'e2e':
        e2enet = load_model(e2enet_args_path, device = device)
    if denoiser_args_path and init_type == 'denoiser':
        denoiser = load_model(denoiser_args_path, device = device)
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        kspace, smaps, image, organ_mask = batch

        kspace = kspace.to(device, non_blocking=True)
        smaps = smaps.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
        organ_mask = organ_mask.to(device, non_blocking=True)
        net.train()

        mask = get_mask(smaps, R, acs_lines)

        if kspace_type == "simulated":
            y, sigma_n = mri_awgn(image, mask, smaps, noise_std)
        elif kspace_type == "measurement":
            if whiten_kspace == True:
                # whiten with masked kspace
                kspace_w, smaps, Sigma_n, Zinv = whiten(mask*kspace, smaps)
                sigma_n = Sigma_n.max()
                y = mask*kspace_w
                # regenerate a coil combined image by using our new smaps (careful, don't use masked kspace)
                image_w = torch.sum(smaps.conj()*ifftc(kspace_w), dim = 1, keepdim = True)
            else:
                y = mask*kspace
                sigma_n = 0.01
        sigma_n = torch.as_tensor(sigma_n, device=image.device, dtype=torch.float32)
        opt.zero_grad(set_to_none = True)
        if x_init:
            with torch.no_grad():
                if init_type == 'e2e': 
                    # Follow how we do inference for default mri recon
                    e2e_recon, _ = e2enet(
                        y,
                        sigma_n,
                        mask,
                        smaps,
                        mri=True
                    )
                    x_t, sig_t = awgn(e2e_recon, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                elif init_type == 'denoiser':
                    # Denoise then feed into the network
                    x_n, sig_t = awgn(image, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                    denoised_recon, _ = denoiser(x_n, sig_t)
                    # Want set sig_t = 0. 
                    x_t = denoised_recon
                elif init_type is None and whiten_kspace is True:
                    x_t, sig_t = awgn(image_w, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                elif init_type is None and whiten_kspace is False:
                    x_t, sig_t = awgn(image, image_noise_std, dist=noise_dist, log_base=noise_log_base)
            '''
            # Blur image first
            if blur is True:
                width = random.randint(pix_width[0], pix_width[1])
                image_blur = gaussian_blur_complex(image, width)
                # Add noise to blurred image
                x_t, sig_t = awgn(image_blur, image_noise_std, dist=noise_dist)
            else:
                x_t, sig_t = awgn(image, image_noise_std, dist=noise_dist)
            )
            '''
            # Train on measured kspace, not simulated
            if ssdu_masking is True:
                # Generate a base mask 
                ssdu_base_mask = make_acc_mask(
                    [smaps.shape[-2], smaps.shape[-1]],
                    ssdu_base_accel,
                    acs_lines
                )
                # Cast to bool so we can apply bitwise operations internally
                ssdu_base_mask = ssdu_base_mask.bool()
                # Subsample on top of this mask
                ssdu_mask, _ = ssdu_mask_subsample(
                    ssdu_base_mask,
                    rho = ssdu_rho, # Keep 20% to 80% of lines
                    acs_size = ssdu_acs,
                    type = "uniform_1D"
                )
                # Push to GPU
                ssdu_mask = ssdu_mask.to(device)
                img_recon,_ = net.forward_double_noise(
                    y,
                    sigma_n,
                    mask,
                    smaps,
                    x_init=x_t,
                    mri=True,
                    sigma_t=sig_t,
                    ssdu_mask = ssdu_mask
                )
            else:
                img_recon,_ = net.forward_double_noise(
                    y,
                    sigma_n,
                    mask,
                    smaps,
                    x_init=x_t,
                    mri=True,
                    sigma_t=sig_t
                )
        else:
            img_recon,_ = net(
                y,
                sigma_n,
                mask,
                smaps,
                mri=True
            )
        # If we whitened, we need to apply Zinv
        if whiten_kspace == True:
            batch_hat = Zinv*batch_hat # Elementwise multiplication      
        mse = torch.mean((image[organ_mask].abs() - img_recon[organ_mask].abs())**2)
        batch_hat = img_recon * organ_mask
        batch = image * organ_mask
        denom = torch.sum(organ_mask)+1e-12

        if loss_type == 'complex-mse':
            loss = torch.sum((batch-batch_hat).abs()**2)/denom
        elif loss_type == 'magnitude-nl1-nl2':
            loss = nl1_nl2_loss(batch.abs(), batch_hat.abs())
        elif loss_type == 'complex-nl1-nl2':
            loss = nl1_nl2_loss(batch, batch_hat)
        elif loss_type == 'magnitude-mse':
            loss = torch.sum((batch.abs()-batch_hat.abs())**2)/denom
        
        if x_init is True and sigma_scaling is True:
            # For mri recon tasks, we actually have to force batch = 1 so this is perfectly valid
            loss = loss*torch.squeeze(sig_t)**(-2)
        
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

        opt.step()
        net.project()

        step += 1
        pbar.update(1)

        if sched is not None:
            sched.step()

        # accumulate stats on GPU
        loss_sum += loss.detach()
        mse_sum  += mse.detach()
        window_count += 1

        # -----------------------
        # Logging (rare CPU sync)
        # -----------------------
        if verbose and step % log_every == 0:
            avg_loss = (loss_sum / window_count).item()
            avg_mse  = (mse_sum  / window_count).item()
            avg_psnr = -10.0 * math.log10(avg_mse + 1e-12)

            wandb.log({
                "train/loss": avg_loss,
                "train/mse": avg_mse,
                "train/psnr": avg_psnr,
                "lr": getlr(opt)[0],
            }, step=step)

            loss_sum.zero_()
            mse_sum.zero_()
            window_count = 0

        # -----------------------
        # Validation
        # -----------------------
        if step % val_every == 0:
            net.eval()
            mse_sum_val = torch.zeros((), device=device)
            # Log Dictionary 
            if wandb.run is not None:
                wandb.log(
                    get_filter_grids_lpds(net, scale_each=False),
                    step=step
                )

            with torch.no_grad():
                for itern, batch in enumerate(
                        tqdm(loaders["val"],
                             desc=f"VAL@{step}",
                             leave=False,
                             dynamic_ncols=True)):
                    kspace, smaps, image, organ_mask = batch
                    kspace = kspace.to(device)
                    smaps = smaps.to(device)
                    image = image.to(device)

                    mask = get_mask(smaps, R, acs_lines)
                    if kspace_type == "simulated":
                        y, sigma_n = mri_awgn(image, mask, smaps, noise_std)
                    elif kspace_type == "measurement":
                        if whiten_kspace == True:
                            # whiten with masked kspace
                            kspace_w, smaps, Sigma_n, Zinv = whiten(mask*kspace, smaps)
                            sigma_n = Sigma_n.max()
                            y = mask*kspace_w
                            # Regenerate coil combined image
                            image_w = torch.sum(smaps.conj()*ifftc(kspace_w), dim = 1, keepdim = True)
                        else:
                            y = mask*kspace
                            sigma_n = 0.01
                        
                    sigma_n = torch.as_tensor(sigma_n, device=image.device)
                    if x_init is True:
                        if init_type == 'e2e':
                            # Follow how we do inference for default mri recon
                            e2e_recon, _ = e2enet(
                                y,
                                sigma_n,
                                mask,
                                smaps,
                                mri=True
                            )
                            x_t, sig_t = awgn(e2e_recon, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                        elif init_type == 'denoiser':
                            # Denoise then feed into the network
                            x_n, sig_t = awgn(image, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                            denoised_recon, _ = denoiser(x_n, sig_t)
                            # Want set sig_t = 0.
                            x_t = denoised_recon
                        elif init_type is None and whiten_kspace is True:
                            x_t, sig_t = awgn(image_w, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                        elif init_type is None and whiten_kspace is False:
                            x_t, sig_t = awgn(image, image_noise_std, dist=noise_dist, log_base=noise_log_base)
                        img_recon,_ = net.forward_double_noise(
                            y,
                            sigma_n,
                            mask,
                            smaps,
                            x_init=x_t,
                            mri=True,
                            sigma_t=sig_t
                        )
                    else:
                        img_recon, _ = net(
                            y,
                            sigma_n,
                            mask,
                            smaps,
                            mri=True,
                        )

                    # If we whitened, we need to revise the sensitivity profile
                    if whiten_kspace is True:
                        img_recon = Zinv * img_recon

                    mse = torch.mean((image[organ_mask].abs() - img_recon[organ_mask].abs())**2)
                    mse_sum_val += mse
                    if itern == 0:
                        # We end up getting black images out of the recon panel, so let's try scaling before hand to cheat
                        gt = image.abs().detach().cpu().numpy()
                        recon = img_recon.abs().detach().cpu().numpy()

                        # Not interested in error, we wanna see noisy x_t
                        # err = np.abs(gt - recon)
                        gt = np.squeeze(gt)
                        recon = np.squeeze(recon)
                        if x_init is True:
                            init = x_t.abs().detach().cpu().numpy()
                            init = np.squeeze(init)                       
                            panel = np.concatenate([gt, recon, init], axis=1)
                            panel = panel/panel.max()
                            wandb.log({
                                "reconstruction_panel": wandb.Image(
                                panel,
                                caption=f"Ground Truth | Reconstruction | Initialization (sigma = {sig_t.item():.2f})"
                            )
                            }, step=step)
                        else:
                            panel = np.concatenate([gt, recon], axis=1)    
                            panel = panel/panel.max()
                            wandb.log({
                                "reconstruction_panel": wandb.Image(
                                panel,
                                caption="Ground Truth | Reconstruction"
                            )
                            }, step=step)
                avg_mse = (mse_sum_val / (itern + 1)).item()
                psnr = -10 * math.log10(avg_mse + 1e-12)
                wandb.log({"val/psnr": psnr}, step=step)

            if psnr > top_psnr["val"]:
                top_psnr["val"] = psnr

            elif psnr + backtrack_thresh < top_psnr["val"]:
                print("Validation dropped — backtracking")
                ckpt_path = os.path.join(save_dir, "net.ckpt")
                net,_,_,_ = load_ckpt(ckpt_path, net, opt, sched)
                old_lr = np.array(getlr(opt))
                new_lr = old_lr * 0.8
                setlr(opt, new_lr)

                print("Updated LR:", new_lr)

        # -----------------------
        # Checkpoint
        # -----------------------
        if step % save_every == 0:
            ckpt_path = os.path.join(save_dir, "net.ckpt")
            print(f"\nCheckpoint: {ckpt_path} (step {step})")
            save_ckpt(ckpt_path, net, step, opt, sched)
            if epoch_fun is not None:
                epoch_fun(step)

    pbar.close()


def grad_norm(params):
    """ computes norm of mini-batch gradient
    """
    total_norm = 0
    for p in params:
        param_norm = torch.tensor(0)
        if p.grad is not None:
            param_norm = p.grad.data.abs().norm(2)
        total_norm = total_norm + param_norm.item()**2
    return total_norm**(.5)

def getlr(opt):
    return [pg['lr'] for pg in opt.param_groups]

def setlr(opt, lr):
    if not issubclass(type(lr), (list, np.ndarray)):
        lr = [lr for _ in range(len(opt.param_groups))]
    for (i, pg) in enumerate(opt.param_groups):
        pg['lr'] = lr[i]

def load_model(config, verbose = False, device = 'cpu'):
    # Load Denoiser
    model_args_file = open(config)
    model_args = json.load(model_args_file)
    model_args_file.close()
    if verbose == True:
        pprint(model_args)
    net, _, _, _ = init_model(model_args, device=device)

    return net

def init_model(args, device):
    model_type, model_args, train_args, paths = [args[item] for item in ['type','model','train','paths']]
    # If loading from checkpoint, init = False
    init = False if paths['ckpt'] is not None else True
    if model_type == "CDLNet":
        net  = CDLNet(**model_args, init=init)
    elif model_type == "LPDSNet":
        net = LPDSNet(**model_args, init = init)
    elif model_type == "LPDSNetBase":
        net = LPDSNetBase(**model_args, init = init)
    # Place model on gpu
    net.to(device)
    
    # set optimizer and learning rate schedule
    opt   = torch.optim.Adam(net.parameters(), **train_args['opt'])     
    if train_args['sched']['type'] == "exponential":
        sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched']['params'])
    elif train_args['sched']['type'] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, **train_args['sched']['params'])
    # get checkpoint path
    ckpt_path = paths['ckpt']
    

    if ckpt_path is not None:
        print(f"Initializing net from {ckpt_path} ...")
        net, opt, sched, epoch0 = load_ckpt(ckpt_path, net, opt, sched)
    else:
        epoch0 = 0

    print("Current Learning Rate(s):")
    for param_group in opt.param_groups:
        print(param_group['lr'])

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Number of Parameters: {total_params:,}")

    print(f"Using {paths['save']} ...")
    os.makedirs(paths['save'], exist_ok=True)
    return net, opt, sched, epoch0

def save_ckpt(path, net=None,epoch=None,opt=None,sched=None):
    """ Save Checkpoint.
    Saves net, optimizer, scheduler state dicts and epoch num to path.
    """
    getSD = lambda obj: obj.state_dict() if obj is not None else None
    torch.save({'epoch': epoch,
                'net_state_dict': getSD(net),
                'opt_state_dict':   getSD(opt),
                'sched_state_dict': getSD(sched)
                }, path)

def load_ckpt(path, net=None,opt=None,sched=None):
    """ Load Checkpoint.
    Loads net, optimizer, scheduler and epoch number
    from state dict stored in path.
    """
    ckpt = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    def setSD(obj, name):
        if obj is not None and name+"_state_dict" in ckpt:
            print(f"Loading {name} state-dict...")
            obj.load_state_dict(ckpt[name+"_state_dict"])
        return obj

    net = setSD(net, 'net')
    opt   = setSD(opt, 'opt')
    sched = setSD(sched, 'sched')
    return net, opt, sched, ckpt['epoch']

def save_args(args, ckpt=True):
    """ Write argument dictionary to file,
    with optionally writing the checkpoint.
    """
    save_path = args['paths']['save']
    if ckpt:
        ckpt_path = os.path.join(save_path, f"net.ckpt")
        args['paths']['ckpt'] = ckpt_path
    with open(os.path.join(save_path, "args.json"), "+w") as outfile:
        outfile.write(json.dumps(args, indent=4, sort_keys=True))

# Helper function to log dictionary. 
def get_filter_grids_lpds(net, scale_each=False):
    """
    Returns a dict:
        filters/A_stage_00
        filters/B_stage_00
        ...
        filters/D
    """
    assert hasattr(net, "A") and hasattr(net, "B")

    K = net.K
    AL, BL = [], []
    mmax = 0

    # collect filters + global max for consistent scaling
    for k in range(K):
        A = net.A[k].weight.detach().cpu()
        B = get_B_complex(net.B[k]).detach().cpu()

        AL.append(A)
        BL.append(B)

        mmax = max(
            mmax,
            A.abs().max().item(),
            B.abs().max().item()
        )

    n = int(np.ceil(np.sqrt(AL[0].shape[0])))

    logs = {}

    for k in range(K):
        Ag = complex_to_rgb_grid(
            AL[k],
            nrow=n,
            scale_each=scale_each,
            global_max=mmax
        )

        Bg = complex_to_rgb_grid(
            BL[k],
            nrow=n,
            scale_each=scale_each,
            global_max=mmax
        )

        logs[f"filters/A_stage_{k:02d}"] = wandb.Image(
            Ag.permute(1, 2, 0).numpy(),
            caption=f"A filters stage {k}"
        )

        logs[f"filters/B_stage_{k:02d}"] = wandb.Image(
            Bg.permute(1, 2, 0).numpy(),
            caption=f"B filters stage {k}"
        )

    # dictionary D
    D = get_B_complex(net.D).detach().cpu()

    Dg = complex_to_rgb_grid(
        D,
        nrow=n,
        scale_each=scale_each,
        global_max=mmax
    )

    logs["filters/D"] = wandb.Image(
        Dg.permute(1, 2, 0).numpy(),
        caption="Learned dictionary D"
    )

    return logs

if __name__ == "__main__":
    """ Load arguments dictionary from json file to pass to main.
    """
    if len(sys.argv)<2:
        print('ERROR: usage: train.py [path/to/arg_file.json]')
        sys.exit(1)
    args_file = open(sys.argv[1])
    args = json.load(args_file)
    pprint(args)
    args_file.close()
    main(args)
