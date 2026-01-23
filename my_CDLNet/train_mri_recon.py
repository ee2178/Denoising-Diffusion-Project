import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn

from functools import partial
from model import CDLNet, LPDSNet
from train_utils import awgn
from kspace_data import get_fit_loaders
from mri_utils import mri_encoding, mri_decoding, make_acc_mask


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
        x_init = False,
        denoiser_args_path=None,
        R = 8, # MRI args
        acs_lines = 24,
        backtrack_thresh = 1):
    
    # Train the model
    print(f"fit: using device {device}")
    
    # Noise standard should be prescribed as a range
    if not type(noise_std) in [list, tuple]:
        noise_std = (noise_std, noise_std)
        
    ckpt_path = os.path.join(save_dir, '0.ckpt')
    save_ckpt(ckpt_path, net, 0, opt, sched)

    top_psnr = {"train": 0, "val": 0, "test": 0} # for backtracking
    epoch = start_epoch
    
    # Do NOT DO THIS
    '''
    if x_init:
        # if we are using a denoiser starting point, then we must load it in:
        dnsr_args_file = open(denoiser_args_path)
        dnsr_args = json.load(dnsr_args_file)
        dnsr_args_file.close()
        dnsr, _, _, _ = init_model(dnsr_args, device=device)
    '''
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
            for itern, batch in enumerate(t):
                # Unpack batch
                kspace, smaps, image = batch
                # Send to device
                kspace = kspace.to(device)
                smaps = smaps.to(device)
                image = image.to(device)

                # generate a mask 
                _, mask = make_acc_mask(shape = (kspace.shape[-2], kspace.shape[-1]), accel = R, acs_lines = acs_lines)
                
                mask = mask.to(device)

                # Generate a masked kspace sample
                kspace_masked = mask * kspace
                
                # Add noise to kspace_masked
                kspace_masked_noisy, sigma_n = awgn(kspace, phase_nstd)
                # Reset gradients
                opt.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Make predictions per batch
                    if x_init:
                        # If we want to initialize our model with some x_t, then we have to generate noisy image domain observation
                        x_t, sig_t = awgn(image, image_noise_std, dist = 'cosine')
                        # We want to add some powerful noise to x_t and then parameterize noise as affine fcn of both sig_t and sig_y
                        img_recon, _ = net.forward_double_noise(kspace_masked_noisy, sigma_n, mask, smaps, x_init = x_t, mri = True, sigma_t = sig_t)
                    else:
                        img_recon, _ = net(kspace_masked_noisy, sigma_n, mask, smaps, mri = True)
                    # supervised or unsupervised (MCSURE) loss during training
                    '''
                    if mcsure and phase == "train":
                        h = 1e-3
                        # Implementation of mcsure loss
                        b = torch.randn_like(obsrv_batch)
                        batch_hat_b, _ = net(obsrv_batch.clone() + h*b, sigma_n, mask=mask)
                        # assume you have a good estimator for sigma_n
                        div = 2.0*torch.mean(((sigma_n/255.0)**2)*b*(batch_hat_b-batch_hat)) / h
                        loss = torch.mean(torch.abs(obsrv_batch - batch_hat)**2) + div
                    '''

                    # if not mcsure then mse 
                    loss = torch.mean(torch.abs(image - img_recon).abs()**2)
                    if phase == 'train':
                        # Get gradients
                        loss.backward()
                        # Clip gradients
                        if clip_grad is not None:
                            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
                        # Apply gradient update
                        opt.step()
                        # method found in the CDLNet class, projects weights onto unit ball
                        net.project()
                loss = loss.item()

                if verbose:
                    total_norm = grad_norm(net.parameters())
                    t.set_postfix_str(f"loss={loss:.1e}|gnorm={total_norm:.1e}")
                psnr = psnr - 10*np.log10(loss)
            
            psnr = psnr/(itern+1)
            print(f"{phase.upper()} PSNR: {psnr:.3f} dB")

            if psnr > top_psnr[phase]:
                top_psnr[phase] = psnr
            # backtracking check
            elif (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
                break

            with open(os.path.join(save_dir, f'{phase}.txt'),'a') as psnr_file:
                psnr_file.write(f'{psnr:.3f}, ')

        if (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
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
