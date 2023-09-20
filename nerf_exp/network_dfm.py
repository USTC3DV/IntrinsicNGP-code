import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
import numpy as np
from .network_util import initseq
def DCTBasis(k,N):
	assert(k<N)
	basis=torch.tensor([np.pi*(float(n)+0.5)*k/float(N) for n in range(N)]).float()
	basis=torch.cos(basis)*(1./np.sqrt(float(N)) if k==0 else np.sqrt(2./float(N)))
	return basis

def DCTSpace(k,N):
	return torch.stack([DCTBasis(ind,N) for ind in range(0,k)])





class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 cfg=None,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.cfg=cfg

        self.num_frames=cfg.data.n_train

        #dfm_network
        desired_resolution=cfg.data.H/cfg.data.downscale
        self.num_layers_dfm = cfg.dfm_net.num_layers        
        self.hidden_dim_dfm = cfg.dfm_net.hidden_dim
        # self.encoder_dfm, self.in_dim_dfm=get_encoder(encoding, input_dim=3, log2_hashmap_size=17,desired_resolution=desired_resolution)
        
        self.encoder_dfm, self.in_dim_dfm=get_encoder('frequency', input_dim=3)
        self.skips = cfg.dfm_net.skip
        
        block_mlps = [nn.Linear(self.in_dim_dfm+69, 
                                self.hidden_dim_dfm), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, self.num_layers_dfm):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(self.hidden_dim_dfm+self.in_dim_dfm, self.hidden_dim_dfm), 
                               nn.ReLU()]
            else:
                block_mlps += [nn.Linear(self.hidden_dim_dfm, self.hidden_dim_dfm), nn.ReLU()]

        block_mlps += [nn.Linear(self.hidden_dim_dfm, 3)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()


        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, log2_hashmap_size=17,desired_resolution=desired_resolution)
   
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                if self.cfg.NeRF.use_direction:
                    in_dim = self.in_dim_dir + self.geo_feat_dim
                else:
                    in_dim=self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
     
        self.bg_net = None


    def forward(self, x, d):

        print(x.requires_grad)
        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color
    
    def deform(self,x,cond_pose,t_index,test=False,global_step=-1):
        # x[N*T,3]
        N,C=x.shape
        # if self.cfg.dfm_net.delay_opt:
        #     t_start=self.cfg.dfm_net.change_step
        #     t_end=self.cfg.dfm_net.end_delay_step
        #     alpha=16*np.maximum(0,global_step-t_start)/(t_end-t_start)
        #     weight=torch.arange(16)
        #     weight=weight.unsqueeze(-1).expand(16,2).reshape(-1)
        #     weight=(1-torch.cos(torch.clamp(alpha-weight,min=0,max=1)*torch.pi))/2
        #     weight=weight.to(x.device)

        if self.cfg.dfm_net.delay_opt:
            t_start=self.cfg.dfm_net.change_step
            t_end=self.cfg.dfm_net.end_delay_step
            alpha=6*np.maximum(0,global_step-t_start)/(t_end-t_start)
            weight=torch.arange(6)
            weight=weight.unsqueeze(-1).expand(6,6).reshape(-1)
            weight=(1-torch.cos(torch.clamp(alpha-weight,min=0,max=1)*torch.pi))/2
            weight_input=torch.ones(3)
            weight=torch.cat([weight_input,weight],dim=0)
            weight=weight.to(x.device)
        cond_pose=cond_pose[:,:,3:]+1e-2

        pos_embed = self.encoder_dfm(x)#[N*T,32]
        if self.cfg.dfm_net.delay_opt:
            weight=weight.unsqueeze(0).expand_as(pos_embed)#[N*T,32]
            pos_embed=weight*pos_embed
        pose_feature=cond_pose[t_index].squeeze(0).expand(N,69)#[N*T,69]
        h=torch.cat([pose_feature,pos_embed],dim=-1)#[N*T,101]
        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h
        # trans=torch.clamp(trans,min=-0.01,max=0.01)
        return trans

    def density(self, rel_pts):

        # rel_pts:[N*T,3]
   

        # print(x.shape)
        # if global_step>self.cfg.dfm_net.change_step and self.cfg.dfm_net.use_dfm:
        #     x=self.deform(x,t_index)
        # print('density',x.requires_grad)

        x = self.encoder(rel_pts)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # print(torch.max(torch.abs(delta_x)))
        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }


    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d,t_index=-1, mask=None, geo_feat=None, **kwargs):
  

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
        if self.cfg.NeRF.use_appcode:
            latent=self.latent(t_index)
            latent=latent.reshape(1,16).expand(x.shape[0],16)#[N,16]
            features = torch.cat((geo_feat, latent), dim=-1).permute(1,0).unsqueeze(0)#[B,31,N]
            features = self.latent_fc(features).squeeze(0).permute(1,0)#[N,15]
        if self.cfg.NeRF.use_direction:
            d = self.encoder_dir(d)
            h = torch.cat([d, geo_feat], dim=-1)
        else:
            h=geo_feat
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.cfg.dfm_net.use_dfm:
            params.append({'params': self.encoder_dfm.parameters(), 'lr': lr/5})
            params.append({'params': self.block_mlps.parameters(), 'lr': lr/5})
        if self.cfg.NeRF.use_appcode:
            params.append({'params': self.latent.parameters(), 'lr': lr})
            params.append({'params': self.latent_fc.parameters(), 'lr': lr})
           

        
        return params
