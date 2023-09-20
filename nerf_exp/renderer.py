import math
import trimesh
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
import raymarching
from .utils import custom_meshgrid

def get_barycentric(p,verts,mesh_indx):
    '''
    p:query [N*T,3]
    verts[6890,3]
    mesh_index[N*T,3]
    return:
    proj:[N*T,3]
    barycentric_coord[N*T,3]
    distance[N*T]
    '''
    flag=torch.ones_like(p)[:,0]#[(N*T)]
    proj=torch.zeros_like(p).float()
    a=verts[mesh_indx[:,0],:].float()#[(N*T),3]
    # print("a",a.shape)
    b=verts[mesh_indx[:,1],:].float()#[(N*T),3]
    c=verts[mesh_indx[:,2],:].float()#[(N*T),3]
    barycentric_coord=torch.zeros_like(p).float()#[(N*T),3]
    ab = b - a
    ac = c - a
    ap = p - a
    dot1 =torch.sum(ab*ap,dim=-1)#[(N*T)]
    dot2 =torch.sum(ac*ap,dim=-1)#[(N*T)]
    d1=dot1.cpu().numpy() #[(N*T)]
    d2=dot2.cpu().numpy() #[(N*T)]
    index1=np.argwhere((d1<=0)*(d2<=0))
    index1=torch.from_numpy(index1).squeeze(-1)
    proj[index1,:]=a[index1,:]
    flag[index1]=0
    barycentric_coord[index1,0]=1.

    bp = p - b
    dot3 =torch.sum(ab*bp,dim=-1)#[(N*T)]
    dot4 =torch.sum(ac*bp,dim=-1)#[(N*T)
    d3=dot3.cpu().numpy() #[(N*T)]
    d4=dot4.cpu().numpy() #[(N*T)]
    index2=np.argwhere((d3>=0) * (d4<=d3))
    index2=torch.from_numpy(index2).squeeze(-1)
    proj[index2,:]=b[index2,:]
    flag[index2]=0
    barycentric_coord[index2,1]=1.

    cp = p - c

    dot5 =torch.sum(ab*cp,dim=-1)#[(N*T)]
    dot6 =torch.sum(ac*cp,dim=-1)#[(N*T)]
    d5=dot5.cpu().numpy() #[(N*T)]
    d6=dot6.cpu().numpy() #[(N*T)]
    index3=np.argwhere((d6>=0)*(d5<d6))
    index3=torch.from_numpy(index3).squeeze(-1)
    proj[index3,:]=c[index3,:]
    flag[index3]=0
    barycentric_coord[index3,2]=1.

    vc = dot1 * dot4 - dot3 * dot2#[(N*T)]
    nc=vc.cpu().numpy()
    index4=np.argwhere((nc <= 0) * (d1 >= 0) *( d3 <= 0))
    index4=torch.from_numpy(index4).squeeze(-1)
    v1 = dot1 / (dot1 - dot3)#[(N*T)]
    proj[index4,:]=a[index4,:] + v1[index4].unsqueeze(-1) * ab[index4,:]
    flag[index4]=0
    barycentric_coord[index4,0],barycentric_coord[index4,1]=1.-v1[index4], v1[index4]
    vb = dot5 * dot2 - dot1 * dot6#[(N*T)]
    nb=vb.cpu().numpy()
    index5=np.argwhere((nb <= 0) * (d2 >= 0)*  (d6 <= 0))
    index5=torch.from_numpy(index5).squeeze(-1)
    w1 = dot2 / (dot2 - dot6)#[(N*T)]
    proj[index5,:]=a[index5,:] + w1[index5].unsqueeze(-1)  * ac[index5,:]
    flag[index5]=0
    barycentric_coord[index5,0],barycentric_coord[index5,2]=1.-w1[index5], w1[index5]
    bc=c-b
    va = dot3 * dot6 - dot5 * dot4
    na=va.cpu().numpy()
    index6=np.argwhere((na <= 0) * (d4 >= d3) * (d6 <= d5))
    index6=torch.from_numpy(index6).squeeze(-1)
    w1 = (dot4 - dot3) / ((dot4 - dot3) + (dot5 - dot6))
    proj[index6,:]=b[index6,:] + w1[index6].unsqueeze(-1)  * bc[index6,:]
    flag[index6]=0
    barycentric_coord[index6,1],barycentric_coord[index6,2]=1.-w1[index6], w1[index6]
    index7=np.argwhere(flag.cpu().numpy())
    index7=torch.from_numpy(index7).squeeze(-1)
    denom = 1. / (va + vb + vc)#[(N*T)]
    v = vb * denom#[(N*T)]
    w = vc * denom#[(N*T)]
    proj[index7,:]=a[index7,:] +ab[index7,:]*v[index7].unsqueeze(-1)  +w[index7].unsqueeze(-1)  * ac[index7,:]
    barycentric_coord[index7,0],barycentric_coord[index7,1],barycentric_coord[index7,2]=1.-v[index7]-w[index7], v[index7],w[index7]
    distance=torch.norm(proj-p,dim=-1)
    return proj, barycentric_coord,distance


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class NeRFRenderer(nn.Module):
    
    def __init__(self,
                data_root=None,
                density_scale=1.1
    ):
        super().__init__()

        uv=np.load(os.path.join(data_root,'mesh_properties/smpl/basicModel_vt.npy'))
        self.uv=torch.from_numpy(uv).cuda()
        uv_f=np.load(os.path.join(data_root,'mesh_properties/smpl/basicModel_ft.npy'))
        self.uv_f=torch.from_numpy(uv_f).cuda()
        mesh_f=np.load(os.path.join(data_root,'mesh_properties/smpl/mesh_f.npy'))
        vf=torch.from_numpy(np.load(os.path.join(data_root,'mesh_properties/smpl/vf.npy'))).cuda()
        self.vf=vf[:,:6]
        self.mesh_f=torch.from_numpy(mesh_f).cuda()
        self.knn = KNN(k=1, transpose_mode=True)
        self.density_scale = density_scale
  

    

    def IntrinsicCoord(self, cfg, pts, verts, face_normals,face_centers,t_index,smpl_params,global_step=-1,test=False):
  
        device=pts.device
        N, T, C = pts.shape
        query=pts.reshape(-1,C)

        ref=verts.to(device)
        face_normals,face_centers=face_normals[0],face_centers[0]
    
        _, vert_idx=self.knn(ref,query.unsqueeze(0))

        vert_idx=vert_idx.squeeze(-1)
        face_idx=self.vf[vert_idx,:]#[B,N*T,6]
        face_idx=face_idx[0]#[N*T,6]
        face_dist=torch.norm((query.unsqueeze(1).expand(N*T,6,3)-face_centers[face_idx,:]),dim=-1)#[n*t,6]
        min_idx=torch.min(face_dist,dim=-1)[1].unsqueeze(-1)#[n*t,1]
 
        face_idx=torch.gather(face_idx,dim=-1,index=min_idx)#[n*t,1]
      
   
        face_idx=face_idx.squeeze(-1)#[(N*T)]
        mesh=verts[0].to(device)#[6890,3]
        
        uv_indx=self.uv_f[face_idx]#[(N*T),3]
        mesh_indx=self.mesh_f[face_idx]#[(N*T),3]
        proj,barycentric_coord,distance=get_barycentric(query,mesh,mesh_indx)
        signed_distance=torch.sign(torch.sum(face_normals[face_idx,:]*(query-proj),dim=-1))*distance#[N*T]


        rel_uv=self.uv[uv_indx,:] #[N*T,3,2] 
       

        rel_uv=torch.sum(barycentric_coord.unsqueeze(-1)*rel_uv,dim=1)#[N*T,2]      

        dist=signed_distance#[N*T]
        rel_uv= rel_uv.float()

        dist=dist*10
        dist=torch.sigmoid(dist)
        inputs=torch.cat((rel_uv,dist.unsqueeze(-1)),dim=-1).float()#[N*T,3]

        if global_step>cfg.dfm_net.change_step and cfg.dfm_net.use_dfm:
            deplace=self.deform(inputs,smpl_params,t_index,test=test,global_step=global_step)
        else:
            deplace=torch.zeros_like(inputs)
        time_smooth_loss=0
        inputs=deplace+inputs
        inputs=torch.clamp(inputs,min = 0,max = 1)
        return inputs,signed_distance.clone().detach(),deplace,time_smooth_loss
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(self, cfg, global_step,rays_o, rays_d, t_index,verts,face_normals,face_centers,testing,smpl_params,num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        bg=cfg.train.bg
        num_steps, upsample_steps=cfg.render.num_steps, cfg.render.upsample_steps
        k=cfg.render.k
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
   
        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        use_depth=True
        disturb=True
        dist_loss=0

        if cfg.render.use_aabb:
            vert=verts[0].permute(1,0)
            x_min,x_max=torch.min(vert[0],0)[0],torch.max(vert[0],0)[0]
            y_min,y_max=torch.min(vert[1],0)[0],torch.max(vert[1],0)[0]
            z_min,z_max=torch.min(vert[2],0)[0],torch.max(vert[2],0)[0]
    
            aabb = torch.FloatTensor([x_min, y_min, z_min, x_max, y_max, z_max]).to(device)
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, 0)
            nears.unsqueeze_(-1)
            fars.unsqueeze_(-1)
            nears[nears>100]=cfg.NeRF.near
            fars[fars>100]=cfg.NeRF.far
            # print(nears.shape)
        else:
            near_ori=cfg.NeRF.near
            far_ori=cfg.NeRF.far


            if global_step!=-1:
                mid=(near_ori+far_ori)/2
                para=min(1,max(0.5,global_step/400))
                near=mid-(far_ori-near_ori)*para
                far=mid+(far_ori-near_ori)*para
            else:
                near=near_ori
                far=far_ori
            nears=(torch.ones(N,1)*near).to(device)
            fars=(torch.ones(N,1)*far).to(device)
        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist


        # generate pts
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [B, N, T, 3]
        # print(pts.shape)
        rel_pts,dist,deplace,time_smooth_loss=self.IntrinsicCoord(cfg,pts,verts, face_normals,face_centers,t_index,smpl_params,global_step=global_step,test=testing)#[B,N*T,k,4],[B,N*T,k,4], [B,N*T]

        deplace_loss=torch.mean(torch.norm(deplace,dim=-1))

            #print(torch.mean(dist))
        fast_mode=cfg.NeRF.fast_mode
        if fast_mode:
            #print('fast_mode')
            fine_dist=dist.cpu().numpy() 
            true_index=np.argwhere(fine_dist<cfg.render.max_dist)#除去距离大于阈值的点
            true_index=torch.from_numpy(true_index).squeeze(-1).to(device) 
            #print(true_index)           
            if true_index.shape[0]==0:
                if bg==1:
                    image=torch.ones(N,3).to(device)
                else:
                    image=torch.zeros(N,3).to(device)
                depth=torch.zeros(N).to(device)
                weights_sum=torch.zeros(N).to(device)
                dist_loss=0
                hard_loss=0
                time_smooth_loss=0
                return {
                        'depth': depth,
                        'image': image,
                        'weights_sum': weights_sum,
                        'dist_loss':dist_loss,
                        'hard_loss': hard_loss,
                        'deplace_loss':deplace_loss,
                        'time_smooth_loss':time_smooth_loss,
                        }
            rel_pts=torch.index_select(rel_pts,0,true_index)

            # rays_d_=torch.index_select(rays_d_,1,true_index)


            sigma=torch.zeros(N*num_steps).to(device)#[N*T]
            geo_feat=torch.zeros(N*num_steps,15).to(device)#[N*T, 15]          
            density_outputs = self.density(rel_pts)
            # print(torch.max(torch.abs(self.offset)))
            true_sigma=density_outputs['sigma']
            true_geo_feat=density_outputs['geo_feat']
  
            sigma[true_index]=true_sigma
            geo_feat[true_index,:]=true_geo_feat

        from math import e
        beta=20
        dist_loss=torch.mean((pow(e,beta*F.relu(dist,inplace=False))-1)*sigma)
        # torch.mean((pow(e,beta*F.relu(dist,inplace=False))-1)*sigma)
        geo_feat = geo_feat.reshape(N, num_steps, -1) # [N, T, 15]
        sigma = sigma.reshape(N, num_steps,-1) # [N, T, 1]

  
        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * sigma.squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
       
            
            new_rel_pts,new_dist,new_deplace,new_time_smooth_loss=self.IntrinsicCoord(cfg,new_pts,verts, face_normals,face_centers,t_index,smpl_params,global_step,test=testing)#[B,N*T,4]

            new_geo_feat=torch.zeros(N*num_steps, 15).to(device)
            new_sigma=torch.zeros(N*num_steps).to(device)
            if fast_mode:
                fine_dist=new_dist.cpu().numpy() 
                true_index=np.argwhere(fine_dist<cfg.render.max_dist)#除去距离大于阈值的点
                true_index=torch.from_numpy(true_index).squeeze(-1).to(device)            
                if true_index.shape[0]==0:
                    if bg==1:
                        image=torch.ones(N,3).to(device)
                    else:
                        image=torch.zeros(N,3).to(device)
                    depth=torch.zeros(N).to(device)
                    weights_sum=torch.zeros(N).to(device)
                    dist_loss=0
                    deplace_loss=0
                    time_smooth_loss=0
                    hard_loss=0
                    return {
                            'depth': depth,
                            'image': image,
                            'weights_sum': weights_sum,
                            'dist_loss':dist_loss,
                            'hard_loss': hard_loss,
                            'deplace_loss':deplace_loss,
                            'time_smooth_loss':time_smooth_loss,
                            }
                new_rel_pts=torch.index_select(new_rel_pts,0,true_index)


            new_density_outputs = self.density(new_rel_pts)
            new_true_sigma=new_density_outputs['sigma']
            new_true_geo_feat=new_density_outputs['geo_feat']
            new_geo_feat[true_index]=new_true_geo_feat
            new_sigma[true_index]=new_true_sigma
            # dist_loss+=torch.mean((pow(e,beta*F.relu(new_dist,inplace=False))-1)*new_sigma)
            # dist_loss/=2
            deplace_loss+=torch.mean(torch.norm(new_deplace,dim=-1))
            deplace_loss/=2
            new_geo_feat = new_geo_feat.reshape(N, num_steps, -1) # [N, T, 15]
            new_sigma = new_sigma.reshape(N, num_steps, -1) # [N, T, 1]

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            pts = torch.cat([pts, new_pts], dim=1) # [N, T+t, 3]
            pts = torch.gather(pts, dim=1, index=z_index.unsqueeze(-1).expand_as(pts))
            dist = torch.cat([dist.reshape(N,-1,1), new_dist.reshape(N,-1,1),], dim=1) # [N, T+t]
            dist = torch.gather(dist, dim=1, index=z_index.unsqueeze(-1).expand_as(dist))
   
            tmp_sigma = torch.cat([sigma, new_sigma], dim=1)
            sigma= torch.gather(tmp_sigma, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_sigma))
            tmp_geo_feat = torch.cat([geo_feat, new_geo_feat], dim=1)
            geo_feat= torch.gather(tmp_geo_feat, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_geo_feat))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
     
        alphas = 1 - torch.exp(-deltas * self.density_scale * sigma.squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(pts)# [N, T+t, 3]
        geo_feat=geo_feat.reshape(-1,geo_feat.shape[-1])

        mask = weights > 1e-4 # hard coded
        rgbs = self.color(pts.reshape(-1, 3), dirs.reshape(-1, 3), t_index,mask=mask.reshape(-1), geo_feat=geo_feat)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        #print(pts.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color


            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        
        # dist_loss=F.relu(dist,inplace=False)*weights+F.relu(-dist,inplace=False)*(1-weights)
        from math import e
        beta=8
        # dist_loss=torch.mean((pow(e,beta*F.relu(dist,inplace=False))-1)*sigma)+torch.mean(torch.sign(F.relu(-dist,inplace=False))*pow((10-sigma),2))
        # dist_loss=dist_loss.mean()
        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()
        hard_loss=-weights * torch.log(weights+1e-7)
        hard_loss=hard_loss.mean()
        # breakpoint()
        return {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
            'dist_loss':dist_loss,
            'hard_loss': hard_loss,
            'deplace_loss':deplace_loss,
            'time_smooth_loss':(time_smooth_loss)
        }
  
    def render(self,cfg, global_step, rays_o, rays_d,t_index,verts,face_normals,face_centers, smpl_params,staged=False, max_ray_batch=2048, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

 
        _run = self.run
        testing=False
        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged:
            testing=True
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            # print(max_ray_batch)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(cfg,global_step,rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail],t_index,verts,face_normals,face_centers, testing,smpl_params,**kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(cfg,global_step,rays_o, rays_d,t_index,verts,face_normals,face_centers,testing,smpl_params, **kwargs)

        return results