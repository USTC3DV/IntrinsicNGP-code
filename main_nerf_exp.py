import torch
import argparse
from nerf_exp.provider import NeRFDataset

from nerf_exp.utils import *

from functools import partial

from config import cfg, opt
#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    print(cfg)
 

    from nerf_exp.network_dfm import NeRFNetwork

 
    seed_everything(cfg.seed)
    
    model = NeRFNetwork(
        cfg=cfg,
        encoding="hashgrid",
        data_root=cfg.data.data_root
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if cfg.test_mode:
        
        metrics = [PSNRMeter()]
        trainer = Trainer('ngp',cfg, model, device=device, workspace=cfg.workspace, criterion=criterion, fp16=False, metrics=metrics, use_checkpoint=cfg.ckpt)

        i,j=cfg.test_view1, cfg.test_view2

        test_loader = NeRFDataset(cfg, device=device, type='test',test_view1=i,test_view2=j).dataloader()
    
        trainer.test(test_loader, test_view1=i,test_view2=j) # test and save video

    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(cfg.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(cfg, device=device, type='train').dataloader()

        print("load data complete")
        # decay to 0.1 * init_lr at last iter step
  
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter /cfg.iters, 1))

        metrics = [PSNRMeter()]
        trainer = Trainer('ngp',  cfg,model, device=device, workspace=cfg.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=False, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=cfg.ckpt, eval_interval=1)
  
        valid_loader = NeRFDataset(cfg, device=device, type='valid').dataloader()

        max_epoch = np.ceil(cfg.iters / len(train_loader)).astype(np.int32)

        trainer.train(train_loader, valid_loader, max_epoch)

        # # also test
        for i in cfg.test.test_view:
    
            test_loader = NeRFDataset(cfg, device=device, type='test',test_view1=i,test_view2=i).dataloader()
        
            trainer.test(test_loader, test_view1=i,test_view2=i) 
        