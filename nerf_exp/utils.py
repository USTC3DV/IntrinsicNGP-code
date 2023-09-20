import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time


import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver




def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)





def sample(mask,patch_size=1,N_rays=-1,in_rio=0.5,in_face=False):
    in_rays=int(N_rays*in_rio)
    around_mask=False
   
    if around_mask:
        out_rays=N_rays-in_rays
        device = mask.device
        
        all_mask=mask[0].cpu().numpy()

        kernel = np.ones((4, 4), np.uint8)
        all_mask = cv2.dilate(all_mask, kernel, iterations=10)
        all_mask[all_mask!=0]=1
        all_mask=torch.from_numpy(all_mask).unsqueeze(0).to(device)
        out=all_mask-mask
        prefix = mask.shape[:-3]
        mask=mask[0,:,:,0].reshape(-1)
        out=out[0,:,:,0].reshape(-1)
        index=mask.nonzero().squeeze(-1)
        out_index=out.nonzero().squeeze(-1)

        msize=index.shape[0]
        osize=out_index.shape[0]
        select_ind=torch.randint(0,msize,size=[in_rays]).to(device)
        select_indout=torch.randint(0,osize,size=[out_rays]).to(device)
        result_in=torch.index_select(index,0,select_ind).to(device)
        result_out=torch.index_select(out_index,0,select_indout).to(device)
        result=torch.cat((result_in,result_out))
        return result.expand([*prefix, N_rays]),N_rays
    else:
        if patch_size==1:    
            out_rays=N_rays-in_rays
            #print(out_rays,N_rays,in_rays)
            prefix = mask.shape[:-3]
            device = mask.device

            H,W=mask.shape[1],mask.shape[2]
            mask=mask[0,:,:,0].reshape(-1)
            out=1-mask
            # print(mask.shape)
            index=mask.nonzero().squeeze(-1)
            out_index=out.nonzero().squeeze(-1)
            msize=index.shape[0]
            osize=out_index.shape[0]

            if in_face:
                in_rays_face=int(in_rays/2)
                # select_ind_face=torch.randint(0,int(msize/10),size=[in_rays_face]).to(device)
                select_ind_face=torch.randint(int(msize/20*19),int(msize),size=[in_rays_face]).to(device)
                select_ind_out_face=torch.randint(0,msize,size=[in_rays_face]).to(device)
                select_ind=torch.cat([select_ind_face,select_ind_out_face],dim=-1)
            select_ind=torch.randint(0,msize,size=[in_rays]).to(device)
            select_indout=torch.randint(0,osize,size=[out_rays]).to(device)
            # print(select_ind.shape)
            # print(index.shape)
            #print(index[select_ind].shape)
            result_in=torch.index_select(index,0,select_ind).to(device)
            result_out=torch.index_select(out_index,0,select_indout).to(device)
            result=torch.cat((result_in,result_out))
        else:
            prefix = mask.shape[:-3] 
            device = mask.device
            H,W=mask.shape[1],mask.shape[2]     
            num_patch = N_rays // (patch_size ** 2)
            mask=mask[0,:,:,0]
            index=mask.nonzero().permute(1,0)
            x_min,x_max=torch.min(index[0],0)[0],torch.max(index[0],0)[0]
            y_min,y_max=torch.min(index[1],0)[0],torch.max(index[1],0)[0]
            inds_x = torch.randint(x_min, x_max - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(y_min, y_max - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten
            result=inds
            in_rays=N_rays
    return result.expand([*prefix, N_rays]),in_rays

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True





class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'



class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 cfg, # config
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.cfg=cfg
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion


        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
  


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):
        images = data['image'] # [B, H, W ,3]
        mask = data["mask"]   #[B, H, W, 3]
        mask_clone=mask.clone().detach()
        B, H, W, C = images.shape
        t_index=data["t_index"].long()
        c_index=data["c_index"].long()

        verts=data["verts"][t_index]#[B,6890,3] vertice
        smpl_params=data["smpl_params"]
        
        face_centers=data["face_centers"][t_index]
        face_normals=data["face_normals"][t_index]
        rays_o,rays_d=data["rays_o"][c_index].reshape(B,-1,3),data["rays_d"][c_index].reshape(B,-1,3) #[B,N ,3]
        bg=self.cfg.train.bg        
        change_step=self.cfg.train.change_step 
        if self.global_step>change_step:
            in_rio=self.cfg.loss.in_rio
            para_dist=self.cfg.loss.para_dist
            para_weight=self.cfg.loss.para_weight
            patch_size=self.cfg.loss.patch_size
            para_dfm=self.cfg.loss.para_dfm
            para_smooth=self.cfg.loss.para_smooth
            para_hard=self.cfg.loss.para_hard
            para_sharp=self.cfg.loss.para_sharp
        else:
            in_rio=0.5
            para_dist=0.1
            para_weight=0.5
            patch_size=1
            para_dfm=0 
            para_smooth=0
            para_hard=0.0
            para_sharp=0.0
        inds,in_N=sample(mask_clone,patch_size,self.cfg.train.num_rays,in_rio,in_face=self.cfg.train.in_face)
        rays_o=torch.gather(rays_o.reshape(B, -1, C),1,torch.stack(C*[inds],-1))
        rays_d=torch.gather(rays_d.reshape(B, -1, C),1,torch.stack(C*[inds],-1))
        gt_rgb = torch.gather(images.reshape(B, -1, C), 1, torch.stack(C*[inds],-1)) # [B, N, 3]   
        
  
 


        outputs = self.model.render(self.cfg, self.global_step, rays_o, rays_d,t_index,verts,face_normals,face_centers, smpl_params,staged=False, bg_color=bg, perturb=True, force_all_rays=False if self.cfg.train.patch_size == 1 else True)
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
    
        pred_rgb = outputs['image'].reshape(B,-1,C)
 
        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]


        loss = loss.mean()

        # extra loss
        if patch_size==1:
            pred_weights_sum = outputs['weights_sum'] 
            loss_ws=torch.mean(1.-pred_weights_sum[:in_N])+10*torch.mean(pred_weights_sum[in_N:])
            
        else:
            loss_ws=0
       
        loss_dist=outputs['dist_loss']
        deplace_loss=outputs['deplace_loss']
        time_smooth_loss=outputs['time_smooth_loss']
        hard_loss=outputs['hard_loss']
        loss_sharp = -pred_weights_sum * torch.log(pred_weights_sum+1e-7) # entropy to encourage weights_sum to be 0 or 1.
        loss_sharp=loss_sharp.mean()
      
    
        loss = loss + para_weight*loss_ws+para_dist*loss_dist+para_dfm*deplace_loss+para_smooth*time_smooth_loss+loss_sharp*para_sharp+hard_loss*para_hard

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        images = data['image'] # [B, H, W ,3]
        mask = data["mask"]   #[B, H, W, 3]
      

        B, H, W, C = images.shape
        t_index=data["t_index"].long()
        c_index=data["c_index"].long()
        verts=data["verts"][t_index]#[B,6890,3] vertice
        face_centers=data["face_centers"][t_index]
        face_normals=data["face_normals"][t_index]
        rays_o,rays_d=data["rays_o"][c_index].reshape(B,-1,3),data["rays_d"][c_index].reshape(B,-1,3) #[B, H, W ,3]

        smpl_params=data["smpl_params"]

        bg=self.cfg.train.bg  
   

        # eval with fixed background color
        bg_color = self.cfg.train.bg

        gt_rgb = images

        with torch.no_grad():
            outputs = self.model.render(self.cfg, self.global_step, rays_o, rays_d,t_index,verts,face_normals,face_centers, smpl_params,staged=True, max_ray_batch=self.cfg.train.max_ray_batch)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  
        t_index=data["t_index"].long()
        device=t_index.device
        index=data["index"]
        rough_mask=data["rough_mask"]
        B,H,W=rough_mask.shape[:3]
        device=t_index.device

        verts=data["verts"][t_index]
        face_centers=data["face_centers"][t_index]
        face_normals=data["face_normals"][t_index]
        rays_o,rays_d=data["rays_o"].reshape(B,-1,3), data["rays_d"].reshape(B,-1,3) #[B, H*W ,3]
        smpl_params=data["smpl_params"]
        
        fast_mode=self.cfg.test.fast_mode
       
        if fast_mode:
            # breakpoint()
            rough_mask=rough_mask.reshape(B,-1)#[B,H*W]
            # rough_mask=torch.sum(rough_mask,dim=-1)
            rough_mask=rough_mask[0,:].cpu().numpy()#[H*W]
            true_index=np.argwhere(rough_mask>0)
            true_index=torch.from_numpy(true_index).squeeze(-1).to(device) 
            rays_o=torch.index_select(rays_o,1,true_index)
            rays_d=torch.index_select(rays_d,1,true_index)
            if self.cfg.train.bg ==0:
                pred_rgb=torch.zeros(B, H,W, 3).reshape(B,-1,3).to(device) 
            else:
                pred_rgb=torch.ones(B, H, W, 3).reshape(B,-1,3).to(device) 
            pred_depth=torch.zeros(B, H, W).reshape(B,-1).to(device)
        with torch.no_grad():
            outputs = self.model.render(self.cfg, self.global_step, rays_o, rays_d,t_index,verts,face_normals,face_centers, smpl_params,staged=True, max_ray_batch=self.cfg.train.max_ray_batch)
        if fast_mode:
            true_pred_rgb=outputs['image'].reshape(B,-1, 3)
            true_pred_depth = outputs['depth'].reshape(B,-1)
            pred_rgb[:,true_index,:]=true_pred_rgb
            pred_depth[:,true_index]=true_pred_depth
            pred_rgb=pred_rgb.reshape(B, H, W, 3)
            pred_depth=pred_depth.reshape(B, H, W)
        else:
            pred_rgb = outputs['image'].reshape(-1, H, W, 3)
            pred_depth = outputs['depth'].reshape(-1, H, W)
        return pred_rgb, pred_depth
       


    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

       
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader,test_view1,test_view2, save_path=None, name=None, write_video=False):

        if save_path is None:
            # save_path = os.path.join(self.workspace, 'results')
            if self.cfg.test.novel_shape:
                save_path = os.path.join(self.workspace, 'novel_shape/Camera ({})'.format(view))
            elif test_view1==test_view2:
                save_path = os.path.join(self.workspace, 'results/Camera ({})'.format(test_view1))
            else:
                save_path = os.path.join(self.workspace, 'results/Inter({}_{})'.format(test_view1,test_view2))
            
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)


                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{i:04d}.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    #cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        


        self.log(f"==> Finished Test.")
    
   

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            

                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)


                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

 
        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])


        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")