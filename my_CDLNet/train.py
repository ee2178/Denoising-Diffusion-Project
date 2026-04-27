import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import wandb

from model import CDLNet, LPDSNet, LPDSNetBase
from train_utils import awgn
from data import get_fit_loaders

def main(args):
    """ Given argument dictionary, load data, initialize model, and fit model.
    """
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")

    model_args, train_args, paths = [args[item] for item in ['model','train','paths']]
    loaders = get_fit_loaders(**train_args['loaders'])
    net, opt, sched, epoch0 = init_model(args, device=device)
    wandb.init(project="lpds_denoiser", config=args)
    wandb.watch(net, log="gradients", log_freq=2000)
    fit(net, 
        opt, 
        loaders,
        sched       = sched,
        save_dir    = paths['save'],
        start_epoch = epoch0 + 1,
        device      = device,
        **train_args['fit'],
        epoch_fun = lambda epoch_num: save_args(args, epoch_num))
def fit(net, opt, loaders,
        sched=None,
        epochs=1,
        device=torch.device("cpu"),
        save_dir=None,
        start_epoch=1,
        clip_grad=1,
        noise_std=25,
        loss_type = 'complex-mse',
        demosaic=False,
        verbose=True,
        val_freq=1,
        save_freq=1,
        epoch_fun=None,
        mcsure=False,
        noise_dist='uniform',
        backtrack_thresh=5,
        log_every = 50,
        max_steps=None,
        val_every=1000,
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
    print("Using " + loss_type + " loss.")
    # -----------------------
    # tqdm progress bar
    # -----------------------
    pbar = tqdm(
        total=max_steps,
        initial=step,
        dynamic_ncols=True,
        desc="TRAIN"
    )

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        net.train()

        batch = batch.to(device)
        batch = batch[:,None,:,:]
        mask = 1

        noisy_batch, sigma_n = awgn(batch, noise_std, dist=noise_dist)
        obsrv_batch = mask * noisy_batch
        opt.zero_grad()
        
        batch_hat,_ = net(obsrv_batch, sigma_n, mask=mask)
        if loss_type == 'complex-mse':
            loss = torch.mean((batch-batch_hat).abs()**2)
        elif loss_type == 'magnitude-mse':
            loss = torch.mean((batch.abs()-batch_hat.abs())**2)
        elif loss_type == 'sigma-scaled-complex-mse':
            loss = torch.mean((sigma_n+1e-2)**(-2)*(batch-batch_hat).abs()**2)
        elif loss_type == 'sigma-scaled-magnitude-mse':
            loss = torch.mean((sigma_n+1e-2)**(-2)*(batch.abs()-batch_hat.abs())**2)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

        opt.step()
        net.project()

        total_norm = grad_norm(net.parameters())
        batch_abs = torch.max(torch.abs(batch))
        batch_mean = torch.abs(torch.mean(batch))
        step += 1
        # pbar.set_postfix_str(f"loss={loss.item():.1e}|gnorm={total_norm:.1e}|batch_abs={batch_abs:.1e}|batch_mean={batch_mean:.1e}|noise_levels={torch.sum(sigma_n).item():.1e}")
        pbar.update(1)

        if sched is not None:
            sched.step()

        # -----------------------
        # Update tqdm metrics
        # -----------------------
        if verbose and step % log_every == 0:
            mse = torch.mean(torch.abs(batch-batch_hat)**2)
            psnr = -10*np.log10(mse.item() + 1e-12)
            wandb.log({
                "train/loss": loss.item(),
                "train/psnr": psnr,
                "lr": getlr(opt)[0],
            }, step=step)
        # ---------------------
        # Validation
        # ---------------------
        if step % val_every == 0:
            net.eval()
            psnr = 0
            with torch.no_grad():
                for itern, batch in enumerate(
                        tqdm(loaders["val"],
                             desc=f"VAL@{step}",
                             leave=False,
                             dynamic_ncols=True)):

                    batch = batch.to(device)
                    batch = batch[:,None,:,:]

                    # phase_nstd = (noise_std[0]+noise_std[1])/2.0
                    phase_nstd = 0.05
                    # Take an average on a log scale 
                    # phase_nstd = 10**(math.log10(noise_std[0])+math.log10(noise))

                    noisy_batch, sigma_n = awgn(batch, phase_nstd, dist=noise_dist)
                    obsrv_batch = noisy_batch

                    batch_hat,_ = net(obsrv_batch, sigma_n, mask=1)

                    mse = torch.mean(torch.abs(batch-batch_hat)**2)
                    psnr += -10*np.log10(mse.item()+1e-12)

                    if itern == 0:
                        # Divide by 10 for visibility, wandb handles normalization
                        gt = batch[0].abs().detach().cpu().numpy()
                        noisy = noisy_batch[0].abs().detach().cpu().numpy()
                        recon = batch_hat[0].abs().detach().cpu().numpy()

                        # normalize for visibility
                        # gt = gt / (gt.max() + 1e-8)
                        # noisy = noisy / (noisy.max() + 1e-8)
                        # recon = recon / (recon.max() + 1e-8)
                        
                        gt = np.squeeze(gt)
                        noisy = np.squeeze(noisy)
                        recon = np.squeeze(recon)

                        err = np.abs(gt - recon)*5

                        panel = np.concatenate([noisy, recon, gt, err], axis=1)
                        panel = panel/panel.max()
                        wandb.log({
                            "denoiser_panel": wandb.Image(
                            panel,
                            caption="Input (Noisy) | Denoised | Ground Truth | Error "
                            )
                        }, step=step)
                psnr /= (itern+1)
            if psnr > top_psnr["val"]:
                top_psnr["val"] = psnr
            elif psnr + backtrack_thresh < top_psnr["val"]:
                print("Validation dropped — backtracking")
                ckpt_path = os.path.join(save_dir,"net.ckpt")
                net,_,_,_ = load_ckpt(ckpt_path,net,opt,sched)

                old_lr = np.array(getlr(opt))
                new_lr = old_lr*0.8
                setlr(opt,new_lr)

                print("Updated LR:",new_lr)

        # ---------------------
        # Checkpoint
        # ---------------------
        if step % save_every == 0:

            ckpt_path = os.path.join(save_dir,"net.ckpt")
            print(f"\nCheckpoint: {ckpt_path} (step {step})")

            save_ckpt(ckpt_path,net,step,opt,sched)

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
    elif model_type == "LPDSNetBase":
        net = LPDSNetBase(**model_args, init = init)
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

def load_model(config, verbose = False, device = 'cpu'):
    # Load Denoiser
    model_args_file = open(config)
    model_args = json.load(model_args_file)
    model_args_file.close()
    if verbose == True:
        pprint(model_args)
    net, _, _, steps = init_model(model_args, device=device)
    return net

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
