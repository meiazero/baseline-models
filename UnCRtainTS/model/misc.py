import os
import sys
import time
import json
import random
import pprint
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

S2_BANDS = 13

def plot_img(imgs, mod, plot_dir, file_id=None):
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    try:
        imgs = imgs.cpu().numpy()
        for tdx, img in enumerate(imgs): # iterate over temporal dimension
            time = '' if imgs.shape[0] == 1 else f'_t-{tdx}'
            if mod in ["pred", "in", "target", "s2"]:
                rgb = [3,2,1] if img.shape[0]==S2_BANDS else [5,4,3]
                img, val_min, val_max = img[rgb, ...], 0, 1
            elif mod == "s1":
                img, val_min, val_max = img[[0], ...], 0, 1
            elif mod == "mask":
                img, val_min, val_max = img[[0], ...], 0, 1
            elif mod == "err":
                img, val_min, val_max = img[[0], ...], 0, 0.01
            elif mod == "var":
                img, val_min, val_max = img[[0], ...], 0, 0.000025
            else: raise NotImplementedError
            if file_id is not None: # export into file name
                img = img.clip(val_min, val_max) # note: this only removes outliers, vmin/vmax below do the global rescaling (else doing instance-wise min/max scaling)
                plt.imsave(os.path.join(plot_dir, f'img-{file_id}_{mod}{time}.png'), np.moveaxis(img,0,-1).squeeze(), dpi=100, cmap='gray', vmin=val_min, vmax=val_max)
    except: 
        if isinstance(imgs, plt.Figure): # the passed argument is a pre-rendered figure
            plt.savefig(os.path.join(plot_dir, f'img-{file_id}_{mod}.png'), dpi=100)
        else: raise NotImplementedError


def export(arrs, mod, export_dir, file_id=None):
    if not os.path.exists(export_dir): os.makedirs(export_dir)
    for tdx, arr in enumerate(arrs): # iterate over temporal dimension
        num = '' if arrs.shape[0] == 1 else f'_t-{tdx}'
        np.save(os.path.join(export_dir, f'img-{file_id}_{mod}{num}.npy'), arr.cpu())


def log_aleatoric(writer, config, mode, step, var, name, img_meter=None):

    # if var is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
    if len(var.shape) > 5: 
        covar = var
        # get [B x 1 x C x H x W] variance tensor
        var   = var.diagonal(dim1=2, dim2=3).moveaxis(-1,2)

        # compute spatial-average to visualize patch-wise covariance matrices
        patch_covmat = covar.mean(dim=-1).mean(dim=-1).squeeze(dim=1)
        for bdx, img in enumerate(patch_covmat): # iterate over [B x C x C] covmats
            img = img.detach().numpy()

            max_abs = max(abs(img.min()), abs(img.max()))
            scale_rel_left, scale_rel_right = -max_abs, +max_abs
            fig = continuous_matshow(img, min=scale_rel_left, max=scale_rel_right)
            writer.add_figure(f'Img/{mode}/patch covmat relative {bdx}',fig, step)
            scale_center0_absolute = 1/4 * 1**2 # assuming covmat has been rescaled already, this is an upper bound
            fig = continuous_matshow(img, min=-scale_center0_absolute, max=scale_center0_absolute)
            writer.add_figure(f'Img/{mode}/patch covmat absolute {bdx}',fig, step)

    # aleatoric uncertainty: comput during train, val and test
    # note: the quantile statistics are computed solely over the variances (and would be much different if involving covariances, e.g. in the isotopic case)
    avg_var     = torch.mean(var, dim=2, keepdim=True) # avg over bands, note: this only considers variances (else diag COV's avg would be tiny)
    q50         = avg_var[:,0,...].view(avg_var.shape[0],-1).median(dim=-1)[0].detach().clone()
    q75         = avg_var[:,0,...].view(avg_var.shape[0],-1).quantile(0.75,dim=-1).detach().clone()
    q50, q75    = q50[0], q75[0] # take batch's first item as a summary
    binning     = 256 # see: https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram

    if config.loss in ["GNLL", 'MGNLL']:
        writer.add_image(f'Img/{mode}/{name}aleatoric [0,1]', avg_var[0,0,...].clip(0, 1), step, dataformats='CHW') # map image to [0, 1]
        writer.add_image(f'Img/{mode}/{name}aleatoric [0,q75]', avg_var[0,0,...].clip(0.0, q75)/q75, step, dataformats='CHW') # map image to [0, q75]
        writer.add_histogram(f'Hist/{mode}/{name}aleatoric', avg_var[0,0,...].flatten().clip(0,1), step, bins=binning, max_bins=binning)
    else: raise NotImplementedError

    writer.add_scalar(f'{mode}/{name}aleatoric median all', q50, step)
    writer.add_scalar(f'{mode}/{name}aleatoric q75 all', q75, step)
    if img_meter is not None: 
        writer.add_scalar(f'{mode}/{name}UCE SE', img_meter.value()['UCE SE'], step)
        writer.add_scalar(f'{mode}/{name}AUCE SE', img_meter.value()['AUCE SE'], step)


def log_train(writer, config, model, step, x, out, y, in_m, name='', var=None):
    # logged loss is before rescaling by learning rate
    _, loss = model.criterion, model.loss_G.cpu()
    if name != '': name = f'model_{name}/'     
    
    writer.add_scalar(f'train/{name}{config.loss}', loss, step)
    writer.add_scalar(f'train/{name}total', loss, step)
    # use add_images for batch-wise adding across temporal dimension
    if config.use_sar:
        writer.add_image(f'Img/train/{name}in_s1', x[0,:,[0], ...], step, dataformats='NCHW')
        writer.add_image(f'Img/train/{name}in_s2', x[0,:,[5,4,3], ...], step, dataformats='NCHW')
    else:
        writer.add_image(f'Img/train/{name}in_s2', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
    writer.add_image(f'Img/train/{name}out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
    writer.add_image(f'Img/train/{name}y', y[0,0,[3,2,1], ...], step, dataformats='CHW')
    writer.add_image(f'Img/train/{name}m', in_m[0,:,None, ...], step, dataformats='NCHW')

    # analyse cloud coverage

    # covered at ALL time points (AND) or covered at ANY time points (OR)
    #and_m, or_m = torch.prod(in_m[0,:, ...], dim=0, keepdim=True), torch.sum(in_m[0,:, ...], dim=0, keepdim=True).clip(0,1)
    and_m, or_m = torch.prod(in_m, dim=1, keepdim=True), torch.sum(in_m, dim=1, keepdim=True).clip(0,1)
    writer.add_scalar(f'train/{name}OR m %', or_m.float().mean(), step)
    writer.add_scalar(f'train/{name}AND m %', and_m.float().mean(), step)
    writer.add_image(f'Img/train/{name}AND m', and_m, step, dataformats='NCHW')
    writer.add_image(f'Img/train/{name}OR m',  or_m, step, dataformats='NCHW')

    and_m_gray = in_m.float().mean(axis=1).cpu()
    for bdx, img in enumerate(and_m_gray):
        fig = discrete_matshow(img, n_colors=config.input_t)
        writer.add_figure(f'Img/train/temp overlay m {bdx}',fig, step)

    if var is not None:  
        # log aleatoric uncertainty statistics, excluding computation of ECE
        log_aleatoric(writer, config, 'train', step, var, name, img_meter=None)

def discrete_matshow(data, n_colors=5, min=0, max=1):
    fig, ax = plt.subplots()
    # get discrete colormap
    cmap = plt.get_cmap('gray', n_colors+1)
    ax.matshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.axis('off')
    fig.tight_layout()
    return fig

def continuous_matshow(data, min=0, max=1):
    fig, ax = plt.subplots()
    # get discrete colormap
    cmap = plt.get_cmap('seismic')
    ax.matshow(data, cmap=cmap, vmin=min, vmax=max)
    ax.axis('off')
    # optionally: provide a colorbar and tick at integers
    # cax = plt.colorbar(mat, ticks=np.arange(min, max + 1))
    return fig


def plot_discard(sorted_errors, config, mode, step, is_se=True, writer=None):
    metric = 'SE' if is_se else 'AE'

    fig, ax = plt.subplots()
    x_axis  = np.arange(0.0, 1.0, 0.05)
    ax.scatter(x_axis, sorted_errors, c="b", alpha=1.0, marker=r'.', label=f"{metric}, sorted by uncertainty")

    # fit a linear regressor with slope b and intercept a
    sorted_errors[np.isnan(sorted_errors)] = np.nanmean(sorted_errors)
    b, a  = np.polyfit(x_axis, sorted_errors, deg=1)
    x_seq = np.linspace(0, 1.0, num=1000)
    ax.plot(x_seq, a + b * x_seq, c="k", lw=1.5, alpha=0.75, label=f"linear fit, {round(a, 3)} + {round(b, 3)} * x")
    plt.xlabel("Fraction of samples, sorted ascendingly by uncertainty")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    plt.grid()
    fig.tight_layout()
    writer.add_figure(f'Img/{mode}/discard_uncertain',fig, step)
    if mode=='test': # export the final test split plots for print
        path_to = os.path.join(config.res_dir, config.experiment_name)
        print(f'Logging discard plots to path {path_to}')
        fig.savefig(os.path.join(path_to, f'plot_{mode}_{metric}_discard.png'), bbox_inches='tight', dpi=int(1e3))
        fig.savefig(os.path.join(path_to, f'plot_{mode}_{metric}_discard.pdf'), bbox_inches='tight', dpi=int(1e3))


def compute_ece(vars, errors, n_samples, percent=5):
    # rank sample-averaged uncertainties ascendingly, and errors accordingly
    _, vars_indices = torch.sort(torch.Tensor(vars))
    errors = torch.Tensor(errors)
    errs_sort = errors[vars_indices]
    # incrementally remove 5% of errors, ranked by highest uncertainty
    bins = torch.linspace(0, n_samples, 100//percent+1, dtype=int)[1:]
    # get uncertainty-sorted cumulative errors, i.e. at x-tick 65% we report the average error for the 65% most certain predictions
    sorted_errors = np.array([torch.nanmean(errs_sort[:rdx]).cpu().numpy() for rdx in bins])

    return sorted_errors 


binarize   = lambda arg, n_bins, floor=0, ceil=1: np.digitize(arg, bins=np.linspace(floor, ceil, num=n_bins)[1:])

def compute_uce_auce(var, errors, n_samples, percent=5, l2=True, mode='val', step=0, writer=None):
    n_bins = 100//percent
    var, errors = torch.Tensor(var), torch.Tensor(errors)

    # metric: IN:  standard deviation & error
    #         OUT: either root mean variance & root mean squared error or mean standard deviation & mean absolute error
    metric = lambda arg: torch.sqrt(torch.mean(arg**2)) if l2 else torch.mean(torch.abs(arg))
    m_str  = 'L2' if l2 else 'L1'

    # group uncertainty values into n_bins 
    var_idx = torch.Tensor(binarize(var, n_bins, floor=var.min(), ceil=var.max()))

    # compute bin-wise statistics, defaults to nan if no data contained in bin
    bk_var, bk_err = torch.empty(n_bins), torch.empty(n_bins)
    for bin_idx in range(n_bins): # for each of the n_bins ... 
        bk_var[bin_idx] = metric(var[var_idx==bin_idx].sqrt())  # note: taking the sqrt to wrap into metric function,
        bk_err[bin_idx] = metric(errors[var_idx==bin_idx])      # apply same metric function on error

    calib_err = torch.abs(bk_err-bk_var)                        # calibration error: discrepancy of error vs uncertainty
    bk_weight = torch.histogram(var_idx, n_bins)[0]/n_samples   # fraction of total data per bin, for bin-weighting
    uce  = torch.nansum(bk_weight * calib_err)                  # calc. weighted UCE, 
    auce = torch.nanmean(calib_err)                             # calc. unweighted AUCE

    # plot bin-wise error versus bin-wise uncertainty
    fig, ax = plt.subplots()
    x_min, x_max = bk_var[~bk_var.isnan()].min(), bk_var[~bk_var.isnan()].max()
    y_min, y_max = 0, bk_err[~bk_err.isnan()].max()
    x_axis  = np.linspace(x_min, x_max, num=n_bins)

    ax.plot(x_axis, x_axis)                                     # diagonal reference line
    ax.bar(x_axis, bk_err, width=x_axis[1]-x_axis[0], alpha=0.75, edgecolor='k', color='gray')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Uncertainty")
    plt.ylabel(f"{m_str} Error")
    plt.legend(loc='upper left')
    plt.grid()
    fig.tight_layout()
    writer.add_figure(f'Img/{mode}/err_vs_var_{m_str}',fig, step)
    
    return uce, auce


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(os.path.join(config.res_dir, config.experiment_name), exist_ok=True)

def checkpoint(log, config):
    with open(
        os.path.join(config.res_dir, config.experiment_name, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)

def save_results(metrics, path, split='test'):
    with open(
        os.path.join(path, f"{split}_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)


# check for file of pre-computed statistics, e.g. indices or cloud coverage
def import_from_path(split, config):
    if os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'util', 'precomputed')):
        import_path = os.path.join(os.path.dirname(os.getcwd()), 'util', 'precomputed', f'generic_{config.input_t}_{split}_{config.region}_s2cloudless_mask.npy')
    else:
        import_path = os.path.join(config.precomputed, f'generic_{config.input_t}_{split}_{config.region}_s2cloudless_mask.npy')
    import_data_path = import_path if os.path.isfile(import_path) else None
    return import_data_path
    