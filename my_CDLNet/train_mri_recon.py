import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import math

from functools import partial
from model import CDLNet, LPDSNet
from train_utils import awgn
from kspace_data import get_fit_loaders
from mri_utils import mri_encoding, mri_decoding, make_acc_mask, mri_awgn

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
'''    
def fit(net, opt, loaders, 
        sched = None,
        epochs = 1, 
        device = torch.device("cpu"), 
        save_dir = None, 
        start_epoch = 1,
        clip_grad = 1,
        noise_std = 25,
        image_noise_std = 0,
        demosaic = False, 
        verbose = True, 
        val_freq = 1,
        save_freq = 1,
        epoch_fun = None, 
        mcsure = False,
        noise_dist='uniform',
        x_init = False,
        denoiser_args_path=None,
        R = 8, # MRI args
        acs_lines = 24,
        backtrack_thresh = 1,
        log_every = 50):
    
    # Train the model
    print(f"fit: using device {device}")
    
    # Noise standard should be prescribed as a range
    if not type(noise_std) in [list, tuple]:
        noise_std = (noise_std, noise_std)
        
    ckpt_path = os.path.join(save_dir, '0.ckpt')
    save_ckpt(ckpt_path, net, 0, opt, sched)

    top_psnr = {"train": 0, "val": 0, "test": 0} # for backtracking
    epoch = start_epoch
    
    # start at the correct epoch, iterate up until number of epochs prescribed
    while epoch < start_epoch + epochs:
        # separate based on training phase
        for phase in ['train', 'val', 'test']:
            # only update params if we are in the training phase
            net.train() if phase == 'train' else net.eval()
            if epoch != epochs and phase == 'test':
                continue
            if phase == 'val' and epoch%val_freq != 0:
                continue
            if phase in ['val', 'test']:
                phase_nstd = (noise_std[0]+noise_std[1])/2.0
            else:
                phase_nstd = noise_std
            psnr = 0
            t = tqdm(iter(loaders[phase]), desc=phase.upper()+'-E'+str(epoch), dynamic_ncols=True)
            log_every = 200  # try 50-200; if small (<10) you’ll tank perf

            # running sums on GPU, allocated once
            loss_sum = torch.zeros((), device=device)
            mse_sum  = torch.zeros((), device=device)

            for itern, batch in enumerate(t):
                _, smaps, image = batch
                # kspace = kspace.to(device, non_blocking=True)
                smaps  = smaps.to(device, non_blocking=True)
                image  = image.to(device, non_blocking=True)

                mask = get_mask(smaps, R, acs_lines)  # IMPORTANT: ensure this is already on GPU
                # kspace_masked = mask * kspace

                # kspace_masked_noisy, sigma_n = awgn(kspace_masked, phase_nstd)  # (also fixed your masked/noise mismatch)
                
                # When adding noise, we actually want to add noise in the multicoil image domain and then turn to kspace and mask
                # WE ACTUALLY DO NOT WANT TO ADD NOISE, KNEE IMAGES ARE INHERENTLY NOISY
                kspace_masked_noisy, sigma_n = mri_awgn(image, mask, smaps, 0.)
                opt.zero_grad(set_to_none=True)
                sigma_n = torch.as_tensor(sigma_n, device=image.device, dtype=torch.float32)
                with torch.set_grad_enabled(phase == 'train'):
                    if x_init is True:
                        x_t, sig_t = awgn(image, image_noise_std, dist=noise_dist)
                        img_recon, _ = net.forward_double_noise(
                            kspace_masked_noisy[0], sigma_n, mask, smaps,
                            x_init=x_t, mri=True, sigma_t=sig_t
                        )
                        loss = torch.mean(torch.pow(sig_t/255., -2) * (image - img_recon).abs()**2)
                    else:
                        img_recon, _ = net(kspace_masked_noisy[0], sigma_n, mask, smaps, mri=True)
                        loss = torch.mean((image - img_recon).abs() ** 2)

                    mse = torch.mean((image - img_recon).abs()**2)

                    if phase == 'train':
                        loss.backward()
                        if clip_grad is not None:
                            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                        opt.step()
                        net.project()

                # --- super cheap accumulation (in-place, no extra log10 kernels) ---
                loss_sum.add_(loss.detach())
                mse_sum.add_(mse.detach())

                # --- log occasionally ---
                if verbose and ((itern + 1) % log_every == 0):
                    avg_loss = (loss_sum / (itern + 1)).item()   # sync only here
                    avg_mse  = (mse_sum  / (itern + 1)).item()
                    avg_psnr = -10.0 * math.log10(avg_mse + 1e-12)  # compute PSNR on CPU

                    # grad_norm is expensive; keep it in the log block (or remove it entirely)
                    total_norm = grad_norm(net.parameters())

                    t.set_postfix_str(
                        f"loss={avg_loss:.3e}|psnr={avg_psnr:.2f}|gnorm={total_norm:.2e}",
                        refresh=False
                    )

            # epoch summary
            avg_mse  = (mse_sum  / (itern + 1)).item()
            avg_psnr = -10.0 * math.log10(avg_mse + 1e-12)
            print(f"{phase.upper()} PSNR: {avg_psnr:.3f} dB")
            if psnr > top_psnr[phase]:
                top_psnr[phase] = psnr
            # backtracking check
            elif (psnr + backtrack_thresh < top_psnr[phase]) or torch.isnan(loss) or torch.isinf(loss):
                break

            with open(os.path.join(save_dir, f'{phase}.txt'),'a') as psnr_file:
                psnr_file.write(f'{psnr:.3f}, ')

        if (psnr + backtrack_thresh < top_psnr[phase]) or torch.isnan(loss) or torch.isinf(loss):
            ckpt_path = os.path.join(save_dir, 'net.ckpt')
            if epoch <= save_freq:  
                ckpt_path = os.path.join(save_dir, '0.ckpt')
            print(f"Loss has diverged. Backtracking to {ckpt_path} ...")

            with open(os.path.join(save_dir, f'backtrack.txt'),'a') as psnr_file:
                psnr_file.write(f'{epoch}  ')

            if epoch % save_freq == 0:
                epoch = epoch - save_freq
            else:
                epoch = epoch - epoch%save_freq

            old_lr = np.array(getlr(opt))
            net, _, _, _ = load_ckpt(ckpt_path, net, opt, sched)
            new_lr = old_lr * 0.8
            setlr(opt, new_lr)
            print("Updated Learning Rate(s):", new_lr)
            epoch = epoch + 1
            continue

        if sched is not None:
            sched.step()
            if hasattr(sched, "step_size") and epoch % sched.step_size == 0:
                print("Updated Learning Rate(s): ")
                print(getlr(opt))

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(save_dir, 'net.ckpt')
            print('Checkpoint: ' + ckpt_path)
            save_ckpt(ckpt_path, net, epoch, opt, sched)

            if epoch_fun is not None:
                epoch_fun(epoch)

        epoch = epoch + 1
'''
def fit(net, opt, loaders,
        sched=None,
        epochs=1,
        device=torch.device("cpu"),
        save_dir=None,
        start_epoch=1,
        clip_grad=1,
        noise_std=25,
        image_noise_std=0,
        demosaic=False,
        verbose=True,
        val_freq=1,
        save_freq=1,
        epoch_fun=None,
        mcsure=False,
        noise_dist='uniform',
        x_init=False,
        denoiser_args_path=None,
        R=8,
        acs_lines=24,
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

    while step < max_steps:

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        _, smaps, image = batch

        smaps = smaps.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        net.train()

        mask = get_mask(smaps, R, acs_lines)

        kspace_masked_noisy, sigma_n = mri_awgn(image, mask, smaps, 0.)
        # We have one extra dimension after this
        kspace_masked_noisy = kspace_masked_noisy[0]
        sigma_n = torch.as_tensor(sigma_n, device=image.device, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        if x_init:
            x_t, sig_t = awgn(image, image_noise_std, dist=noise_dist)
            img_recon,_ = net.forward_double_noise(
                kspace_masked_noisy,
                sigma_n,
                mask,
                smaps,
                x_init=x_t,
                mri=True,
                sigma_t=sig_t
            )

            loss = torch.mean(torch.pow(sig_t/255., -2) * (image - img_recon).abs()**2)

        else:
            img_recon,_ = net(
                kspace_masked_noisy,
                sigma_n,
                mask,
                smaps,
                mri=True
            )

            loss = torch.mean((image - img_recon).abs()**2)

        mse = torch.mean((image - img_recon).abs()**2)

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

            # grad norm less frequent
            total_norm = grad_norm(net.parameters()) if step % (log_every*4) == 0 else None

            postfix = {
                "loss": f"{avg_loss:.3e}",
                "psnr": f"{avg_psnr:.2f}",
                "lr": f"{getlr(opt)[0]:.2e}"
            }

            if total_norm is not None:
                postfix["gnorm"] = f"{total_norm:.2e}"

            pbar.set_postfix(postfix)

            # reset window stats
            loss_sum.zero_()
            mse_sum.zero_()
            window_count = 0

        # -----------------------
        # Validation
        # -----------------------
        if step % val_every == 0:

            net.eval()

            mse_sum_val = torch.zeros((), device=device)

            with torch.no_grad():

                for itern, batch in enumerate(
                        tqdm(loaders["val"],
                             desc=f"VAL@{step}",
                             leave=False,
                             dynamic_ncols=True)):

                    _, smaps, image = batch

                    smaps = smaps.to(device)
                    image = image.to(device)

                    mask = get_mask(smaps, R, acs_lines)

                    kspace_masked_noisy, sigma_n = mri_awgn(image, mask, smaps, 0.)
                    kspace_masked_noisy = kspace_masked_noisy[0]
                    sigma_n = torch.as_tensor(sigma_n, device=image.device)

                    img_recon,_ = net(
                        kspace_masked_noisy,
                        sigma_n,
                        mask,
                        smaps,
                        mri=True
                    )

                    mse = torch.mean((image - img_recon).abs()**2)
                    mse_sum_val += mse

            avg_mse = (mse_sum_val / (itern + 1)).item()
            psnr = -10 * math.log10(avg_mse + 1e-12)

            print(f"\nVAL PSNR @ step {step}: {psnr:.3f} dB")

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
    
def init_model(args, device, quant_ckpt = False):
    model_type, model_args, train_args, paths = [args[item] for item in ['type','model','train','paths']]
    # If loading from checkpoint, init = False
    init = False if paths['ckpt'] is not None else True
    if model_type == "CDLNet":
        net  = CDLNet(**model_args, init=init)
    elif model_type == "LPDSNet":
        net = LPDSNet(**model_args, init = init)
    # Place model on gpu
    net.to(device)
    
    # set optimizer and learning rate schedule
    if quant_ckpt:
        opt = None
        sched = None
    else:
        opt   = torch.optim.Adam(net.parameters(), **train_args['opt'])     
        sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched'])

    # get checkpoint path
    ckpt_path = paths['ckpt']
    

    if ckpt_path is not None:
        print(f"Initializing net from {ckpt_path} ...")
        net, opt, sched, epoch0 = load_ckpt(ckpt_path, net, opt, sched)
    else:
        epoch0 = 0

    #print("Current Learning Rate(s):")
    #for param_group in opt.param_groups:
    #    print(param_group['lr'])

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
