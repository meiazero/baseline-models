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


dirname = os.path.dirname(os.getcwd())
sys.path.append(os.path.dirname(dirname))

from model.parse_args import create_parser
# from data.dataLoader import SEN12MSCR, SEN12MSCRTS
from data.SEN12MSCRTS import SEN12MSCRTS
from model.src.model_utils import get_model, save_model, freeze_layers, load_model, load_checkpoint
from model.src.learning.metrics import img_metrics, avg_img_metrics
from model.misc import *

import torch
import torchnet as tnt
from torch.utils.tensorboard import SummaryWriter

from model.src import utils, losses
from model.src.learning.weight_init import weight_init










S2_BANDS = 13
parser   = create_parser(mode='train')
config   = utils.str2list(parser.parse_known_args()[0], list_args=["encoder_widths", "decoder_widths", "out_conv"])

config.root1 = "/share/hariharan/cloud_removal/SEN12MSCRTS"
config.root2 = "/share/hariharan/cloud_removal/SEN12MSCRTS"
config.root3 = "/share/hariharan/cloud_removal/SEN12MSCRTS"

config.model = "uncrtaints"
config.input_t = 3
config.region = "all"
config.epochs = 20
config.lr = 0.001
config.batch_size = 4
config.gamma = 1.0
config.scale_by = 10.0
config.trained_checkp = ""
config.loss = "MGNLL"
config.covmode = "diag"
config.var_nonLinearity = "softplus"
config.display_step = 10
config.use_sar = True
config.block_type = "mbconv"
config.n_head = 16
config.device = "cuda"
config.res_dir = "./results"
config.rdm_seed = 1

config.experiment_name = "allclear_v1"

if config.model in['unet', 'utae']:
    assert len(config.encoder_widths) == len(config.decoder_widths)
    config.loss = 'l2'
    if config.model=='unet':
        # train U-Net from scratch
        config.pretrain=True
        config.trained_checkp = ''

if config.pretrain:  # pre-training is on a single time point
    config.input_t = config.n_head = 1
    config.sample_type = 'pretrain'
    if config.model=='unet': config.batch_size = 32
    config.positional_encoding = False

if config.loss in ['GNLL', 'MGNLL']:
    # for univariate losses, default to univariate mode (batched across channels)
    if config.loss in ['GNLL']: config.covmode = 'uni' 

    if config.covmode == 'iso':
        config.out_conv[-1] += 1
    elif config.covmode in ['uni', 'diag']:
        config.out_conv[-1] += S2_BANDS
        config.var_nonLinearity = 'softplus'

# grab the PID so we can look it up in the logged config for server-side process management
config.pid = os.getpid()

# import & re-load a previous configuration, e.g. to resume training
if config.resume_from:
    load_conf = os.path.join(config.res_dir, config.experiment_name, 'conf.json')
    if config.experiment_name != config.trained_checkp.split('/')[-2]: 
        raise ValueError("Mismatch of loaded config file and checkpoints")
    with open(load_conf, 'rt') as f:
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = ['pid', 'num_workers', 'root1', 'root2', 'root3', 'resume_from', 'trained_checkp', 'epochs', 'encoder_widths', 'decoder_widths', 'lr']
        conf_dict = {key:val for key,val in json.load(f).items() if key not in no_overwrite}
        for key, val in vars(config).items(): 
            if key in no_overwrite: conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
config = utils.str2list(config, list_args=["encoder_widths", "decoder_widths", "out_conv"])

# resume at a specified epoch and update optimizer accordingly
if config.resume_at >= 0:
    config.lr = config.lr * config.gamma**config.resume_at


# fix all RNG seeds,
# throw the whole bunch at 'em
def seed_packages(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# seed everything
seed_packages(config.rdm_seed)
# seed generators for train & val/test dataloaders
f, g = torch.Generator(), torch.Generator()
f.manual_seed(config.rdm_seed + 0)  # note:  this may get re-seeded each epoch
g.manual_seed(config.rdm_seed)      #        keep this one fixed

if __name__ == "__main__": pprint.pprint(config)

# instantiate tensorboard logger
writer = SummaryWriter(os.path.join(os.path.dirname(config.res_dir), "logs", config.experiment_name))



def prepare_data(batch, device, config):
    if config.pretrain: return prepare_data_mono(batch, device, config)
    else: return prepare_data_multi(batch, device, config)

def prepare_data_mono(batch, device, config):
    x = batch['input']['S2'].to(device).unsqueeze(1)
    if config.use_sar: 
        x = torch.cat((batch['input']['S1'].to(device).unsqueeze(1), x), dim=2)
    m = batch['input']['masks'].to(device).unsqueeze(1)
    y = batch['target']['S2'].to(device).unsqueeze(1)
    return x, y, m

def prepare_data_multi(batch, device, config):
    in_S2       = recursive_todevice(batch['input']['S2'], device)
    in_S2_td    = recursive_todevice(batch['input']['S2 TD'], device)
    if config.batch_size>1: in_S2_td = torch.stack((in_S2_td)).T
    in_m        = torch.stack(recursive_todevice(batch['input']['masks'], device)).swapaxes(0,1)
    target_S2   = recursive_todevice(batch['target']['S2'], device)
    y           = torch.cat(target_S2,dim=0).unsqueeze(1)

    if config.use_sar: 
        in_S1 = recursive_todevice(batch['input']['S1'], device)
        in_S1_td = recursive_todevice(batch['input']['S1 TD'], device)
        if config.batch_size>1: in_S1_td = torch.stack((in_S1_td)).T
        x     = torch.cat((torch.stack(in_S1,dim=1), torch.stack(in_S2,dim=1)),dim=2)
        # dates = torch.stack((torch.tensor(in_S1_td),torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
        # print(in_S1_td)
        # dates = torch.stack((torch.tensor(in_S1_td),torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
        dates = torch.stack((in_S1_td.clone().detach(),in_S2_td.clone().detach())).float().mean(dim=0).to(device)
    else:
        x     = torch.stack(in_S2,dim=1)
        print(in_S2_td)
        dates = torch.tensor(in_S2_td).float().to(device)
    
    return x, y, in_m, dates

def iterate(model, data_loader, config, writer, mode="train", epoch=None, device=None):
    if len(data_loader) == 0: raise ValueError("Received data loader with zero samples!")
    # loss meter, needs 1 meter per scalar (see https://tnt.readthedocs.io/en/latest/_modules/torchnet/meter/averagevaluemeter.html);
    loss_meter = tnt.meter.AverageValueMeter()
    img_meter  = avg_img_metrics()

    # collect sample-averaged uncertainties and errors
    errs, errs_se, errs_ae,  vars_aleatoric= [], [], [], []

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        step = (epoch-1)*len(data_loader)+i

        if config.dataset == "ALLCLEAR":
            x, y, in_m, dates = batch
            x, y, in_m, dates = x.to(device), y.to(device), in_m.to(device), dates.to(device)
        elif config.sample_type == 'cloudy_cloudfree':
            x, y, in_m, dates = prepare_data(batch, device, config)
        elif config.sample_type == 'pretrain':
            x, y, in_m = prepare_data(batch, device, config)
            dates = None
        else:
            raise NotImplementedError
        inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m}


        if mode != "train": # val or test
            with torch.no_grad():
                # compute single-model mean and variance predictions
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                model.rescale()
                out = model.fake_B
                if hasattr(model.netG, 'variance') and model.netG.variance is not None:
                    var = model.netG.variance
                    model.netG.variance = None
                else:
                    var = out[:, :, S2_BANDS:, ...]
                out = out[:, :, :S2_BANDS, ...]
                batch_size = y.size()[0]

                for bdx in range(batch_size):
                    # only compute statistics on variance estimates if using e.g. NLL loss or combinations thereof
                    
                    if config.loss in ['GNLL', 'MGNLL']:
                        
                        # if the variance variable is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
                        if len(var.shape) > 5: 
                            covar = var
                            # get [B x 1 x C x H x W] variance tensor
                            var   = var.diagonal(dim1=2, dim2=3).moveaxis(-1,2)

                        extended_metrics = img_metrics(y[bdx], out[bdx], var=var[bdx])
                        vars_aleatoric.append(extended_metrics['mean var']) 
                        errs.append(extended_metrics['error'])
                        errs_se.append(extended_metrics['mean se'])
                        errs_ae.append(extended_metrics['mean ae'])
                    else:
                        extended_metrics = img_metrics(y[bdx], out[bdx])
                    
                    img_meter.add(extended_metrics)
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if config.plot_every>0 and idx % config.plot_every == 0:
                        plot_dir = os.path.join(config.res_dir, config.experiment_name, 'plots', f'epoch_{epoch}', f'{mode}')
                        plot_img(x[bdx], 'in', plot_dir, file_id=idx)
                        plot_img(out[bdx], 'pred', plot_dir, file_id=idx)
                        plot_img(y[bdx], 'target', plot_dir, file_id=idx)
                        plot_img(((out[bdx]-y[bdx])**2).mean(1, keepdims=True), 'err', plot_dir, file_id=idx)
                        plot_img(discrete_matshow(in_m.float().mean(axis=1).cpu()[bdx], n_colors=config.input_t), 'mask', plot_dir, file_id=idx)
                        if var is not None: plot_img(var.mean(2, keepdims=True)[bdx], 'var', plot_dir, file_id=idx)
                    if config.export_every>0 and idx % config.export_every == 0:
                        export_dir = os.path.join(config.res_dir, config.experiment_name, 'export', f'epoch_{epoch}', f'{mode}')
                        export(out[bdx], 'pred', export_dir, file_id=idx)
                        export(y[bdx], 'target', export_dir, file_id=idx)
                        if var is not None: 
                            try: export(covar[bdx], 'covar', export_dir, file_id=idx)
                            except: export(var[bdx], 'var', export_dir, file_id=idx)
        else: # training
            
            # compute single-model mean and variance predictions
            model.set_input(inputs)
            model.optimize_parameters() # not using model.forward() directly
            out    = model.fake_B.detach().cpu()

            # read variance predictions stored on generator
            if hasattr(model.netG, 'variance') and model.netG.variance is not None:
                var = model.netG.variance.cpu()
            else:
                var = out[:, :, S2_BANDS:, ...]
            out = out[:, :, :S2_BANDS, ...]

            if config.plot_every>0:
                plot_out = out.detach().clone()
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if idx % config.plot_every == 0:
                        plot_dir = os.path.join(config.res_dir, config.experiment_name, 'plots', f'epoch_{epoch}', f'{mode}')
                        plot_img(x[bdx], 'in', plot_dir, file_id=i)
                        plot_img(plot_out[bdx], 'pred', plot_dir, file_id=i)
                        plot_img(y[bdx], 'target', plot_dir, file_id=i)

        if mode == "train":
            # periodically log stats
            if step%config.display_step==0:
                out, x, y, in_m = out.cpu(), x.cpu(), y.cpu(), in_m.cpu()
                if config.loss in ['GNLL', 'MGNLL']:
                    var = var.cpu()
                    log_train(writer, config, model, step, x, out, y, in_m, var=var)
                else:
                    log_train(writer, config, model, step, x, out, y, in_m)
        
        # log the loss, computed via model.backward_G() at train time & via model.get_loss_G() at val/test time
        loss_meter.add(model.loss_G.item())

        # after each batch, close any leftover figures
        plt.close('all')

    # --- end of epoch ---
    # after each epoch, log the loss metrics
    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {f"{mode}_epoch_time": total_time}
    # log the loss, only computed within model.backward_G() at train time
    metrics[f"{mode}_loss"] = loss_meter.value()[0]

    if mode == "train": # after each epoch, update lr acc. to scheduler
        current_lr = model.optimizer_G.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('Etc/train/lr', current_lr, step)
        model.scheduler_G.step()

    if mode == "test" or mode == "val":
        # log the metrics

        # log image metrics
        for key, val in img_meter.value().items(): writer.add_scalar(f'{mode}/{key}', val, step)

        # any loss is currently only computed within model.backward_G() at train time
        writer.add_scalar(f'{mode}/loss', metrics[f"{mode}_loss"], step)

        # use add_images for batch-wise adding across temporal dimension
        if config.use_sar:
            writer.add_image(f'Img/{mode}/in_s1', x[0,:,[0], ...], step, dataformats='NCHW')
            writer.add_image(f'Img/{mode}/in_s2', x[0,:,[5,4,3], ...], step, dataformats='NCHW')
        else:
            writer.add_image(f'Img/{mode}/in_s2', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
        writer.add_image(f'Img/{mode}/out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
        writer.add_image(f'Img/{mode}/y', y[0,0,[3,2,1], ...], step, dataformats='CHW')
        writer.add_image(f'Img/{mode}/m', in_m[0,:,None, ...], step, dataformats='NCHW')


        # compute Expected Calibration Error (ECE)
        if config.loss in ['GNLL', 'MGNLL']:
            sorted_errors_se   = compute_ece(vars_aleatoric, errs_se, len(data_loader.dataset), percent=5)
            sorted_errors      = {'se_sortAleatoric': sorted_errors_se}
            plot_discard(sorted_errors['se_sortAleatoric'], config, mode, step, is_se=True,writer=writer)

            # compute ECE 
            uce_l2, auce_l2 = compute_uce_auce(vars_aleatoric, errs, len(data_loader.dataset), percent=5, l2=True, mode=mode, step=step, writer=writer)

            # no need for a running mean here
            img_meter.value()['UCE SE']  = uce_l2.cpu().numpy().item()
            img_meter.value()['AUCE SE'] = auce_l2.cpu().numpy().item()

        if config.loss in ['GNLL', 'MGNLL']:
            log_aleatoric(writer, config, mode, step, var,  f'model/', img_meter)

        return metrics, img_meter.value()
    else:
        return metrics
    
    
    
    
    
    
    
    
    
    
    
prepare_output(config)
device = torch.device(config.device)

# define data sets
if config.pretrain: # pretrain / training on mono-temporal data
    dt_train    = SEN12MSCR(os.path.expanduser(config.root3), split='train', region=config.region, sample_type=config.sample_type)
    dt_val      = SEN12MSCR(os.path.expanduser(config.root3), split='val', region=config.region, sample_type=config.sample_type) 
    dt_test     = SEN12MSCR(os.path.expanduser(config.root3), split='test', region=config.region, sample_type=config.sample_type)
else:
    if config.dataset == "SEN12MSCRTS":
        dt_train    = SEN12MSCRTS(split='train', region=config.region, sample_type=config.sample_type, sampler = 'random' if config.vary_samples else 'fixed', n_input_samples=config.input_t, import_data_path=import_from_path('train', config), min_cov=config.min_cov, max_cov=config.max_cov)
        dt_val      = SEN12MSCRTS(split='val', region='all', sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=import_from_path('val', config)) 
        dt_test     = SEN12MSCRTS(split='test', region='all', sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=import_from_path('test', config))
        
    if config.dataset == "ALLCLEAR":
        from data.dataloader_v46 import CogDataset_v46

        dt_train = CogDataset_v46(max_num_frames=config.input_t, image_size=256, mode="train")
        # train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        dt_val = CogDataset_v46(max_num_frames=config.input_t, image_size=256, mode="val")
        # test_dataloader = DataLoader(test_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        dt_test = CogDataset_v46(max_num_frames=config.input_t, image_size=256, mode="test")
        # test_dataloader = DataLoader(test_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)


# wrap to allow for subsampling, e.g. for test runs etc
dt_train    = torch.utils.data.Subset(dt_train, range(0, min(config.max_samples_count, len(dt_train), int(len(dt_train)*config.max_samples_frac))))
dt_val      = torch.utils.data.Subset(dt_val, range(0, min(config.max_samples_count, len(dt_val), int(len(dt_train)*config.max_samples_frac))))
dt_test     = torch.utils.data.Subset(dt_test, range(0, min(config.max_samples_count, len(dt_test), int(len(dt_train)*config.max_samples_frac))))

# instantiate dataloaders, note: worker_init_fn is needed to get reproducible random samples across runs if vary_samples=True
train_loader = torch.utils.data.DataLoader(
    dt_train,
    batch_size=config.batch_size,
    shuffle=True,
    worker_init_fn=seed_worker, generator=f,
    num_workers=config.num_workers,
)
val_loader = torch.utils.data.DataLoader(
    dt_val,
    batch_size=config.batch_size,
    shuffle=False,
    worker_init_fn=seed_worker, generator=g,
    num_workers=config.num_workers,
)
test_loader = torch.utils.data.DataLoader(
    dt_test,
    batch_size=config.batch_size,
    shuffle=False,
    worker_init_fn=seed_worker, generator=g,
    #num_workers=config.num_workers,
)

print("Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test)))

# model definition
# (compiled model hangs up in validation step on some systems, retry in the future for pytorch > 2.0)
model = get_model(config) #torch.compile(get_model(config))

# set model properties
model.len_epoch = len(train_loader)

config.N_params = utils.get_ntrainparams(model)
# print("\n\nTrainable layers:")
# for name, p in model.named_parameters():
#     if p.requires_grad: print(f"\t{name}")
model = model.to(device)
# do random weight initialization
print('\nInitializing weights randomly.')
model.netG.apply(weight_init)

if config.trained_checkp and len(config.trained_checkp)>0:
    # load weights from the indicated checkpoint
    print(f'Loading weights from (pre-)trained checkpoint {config.trained_checkp}')
    load_model(config, model, train_out_layer=True, load_out_partly=config.model in ['uncrtaints'])

with open(os.path.join(config.res_dir, config.experiment_name, "conf.json"), "w") as file:
    file.write(json.dumps(vars(config), indent=4))
print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
# print(model)

# Optimizer and Loss
model.criterion = losses.get_loss(config)

# track best loss, checkpoint at best validation performance
is_better, best_loss = lambda new, prev: new <= prev, float("inf")

# Training loop
trainlog = {}

# resume training at scheduler's latest epoch, != 0 if --resume_from
begin_at = config.resume_at if config.resume_at >= 0 else model.scheduler_G.state_dict()['last_epoch']
for epoch in range(begin_at+1, config.epochs + 1):
    print("\nEPOCH {}/{}".format(epoch, config.epochs))

    # put all networks in training mode again
    model.train()
    model.netG.train()

    # unfreeze all layers after specified epoch
    if epoch>config.unfreeze_after and hasattr(model, 'frozen') and model.frozen:
        print('Unfreezing all network layers')
        model.frozen = False
        freeze_layers(model.netG, grad=True)

    # re-seed train generator for each epoch anew, depending on seed choice plus current epoch number
    #   ~ else, dataloader provides same samples no matter what epoch training starts/resumes from
    #   ~ note: only re-seed train split dataloader (if config.vary_samples), but keep all others consistent
    #   ~ if desiring different runs, then the seeds must at least be config.epochs numbers apart
    if config.vary_samples:
        # condition dataloader samples on current epoch count
        f.manual_seed(config.rdm_seed + epoch)
        train_loader = torch.utils.data.DataLoader(
                        dt_train,
                        batch_size=config.batch_size,
                        shuffle=True,
                        worker_init_fn=seed_worker, generator=f,
                        num_workers=config.num_workers,
                        )

    train_metrics = iterate(
        model,
        data_loader=train_loader,
        config=config,
        writer=writer,
        mode="train",
        epoch=epoch,
        device=device,
    )

    # do regular validation steps at the end of each training epoch
    if epoch % config.val_every == 0 and epoch > config.val_after:
        print("Validation . . . ")

        model.eval()
        model.netG.eval()

        val_metrics, val_img_metrics = iterate(
                                        model,
                                        data_loader=val_loader,
                                        config=config,
                                        writer=writer,
                                        mode="val",
                                        epoch=epoch,
                                        device=device,
                                    )
        # use the training loss for validation
        print('Using training loss as validation loss')
        if "val_loss" in val_metrics: val_loss = val_metrics["val_loss"]
        else: val_loss = val_metrics['val_loss_ensembleAverage']


        print(f'Validation Loss {val_loss}')
        print(f'validation image metrics: {val_img_metrics}')
        save_results(val_img_metrics, os.path.join(config.res_dir, config.experiment_name), split=f'val_epoch_{epoch}')
        print(f'\nLogged validation epoch {epoch} metrics to path {os.path.join(config.res_dir, config.experiment_name)}')   

        # checkpoint best model
        trainlog[epoch] = {**train_metrics, **val_metrics}
        checkpoint(trainlog, config)
        if is_better(val_loss, best_loss):
            best_loss = val_loss
            save_model(config, epoch, model, "model")
    else:
        trainlog[epoch] = {**train_metrics}
        checkpoint(trainlog, config)

    # always checkpoint the current epoch's model
    save_model(config, epoch, model, f"model_epoch_{epoch}")

    print(f'Completed current epoch of experiment {config.experiment_name}.')

# following training, test on hold-out data
print("Testing best epoch . . .")
load_checkpoint(config, config.res_dir, model, "model")

model.eval()
model.netG.eval()

test_metrics, test_img_metrics = iterate(
                                model,
                                data_loader=test_loader,
                                config=config,
                                writer=writer,
                                mode="test",
                                epoch=epoch,
                                device=device,
                            )

if "test_loss" in test_metrics: test_loss = test_metrics["test_loss"]
else: test_loss = test_metrics['test_loss_ensembleAverage']
print(f'Test Loss {test_loss}')
print(f'\nTest image metrics: {test_img_metrics}')
save_results(test_img_metrics, os.path.join(config.res_dir, config.experiment_name), split='test')
print(f'\nLogged test metrics to path {os.path.join(config.res_dir, config.experiment_name)}')   

# close tensorboard logging
writer.close()

print(f'Finished training experiment {config.experiment_name}.')