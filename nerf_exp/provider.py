import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .render_nvdiff import MeshRenderer


def batch_to_tensor(array):
    return torch.from_numpy(array).unsqueeze(0).cuda()

def get_rays(Pose, K, H, W):

    R=Pose[ :3, :3]
    T=Pose[ :3, 3]
    K=K
    rays_o = -np.dot(R.T, T).ravel()

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),np.arange(H, dtype=np.float32),indexing='xy')

    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)

    rays_d = pixel_world - rays_o[None, None]
    rays_d=torch.tensor(rays_d).cuda()
    rays_d=F.normalize(rays_d, dim=-1)
    rays_d=rays_d.cpu().numpy()
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o,rays_d


def load_cam(camera_file,downscale=1):
    cams = np.load(camera_file, allow_pickle=True).item()
    K = []
    K_ori=[]
    RT = []
    D=cams['D']
    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        k_ori=np.array(cams['K'][i]).copy()
        K_ori.append(k_ori)
        K[i][:2] = K[i][:2] /downscale
        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i])
        r_t = np.concatenate([r, t], 1)
        RT.append(r_t)  
    return K, K_ori,RT,D

def load_data(type,cfg,test_view1=-1,test_view2=-1):
    start_frame=cfg.data.start_frame
    n_train=cfg.data.n_train
    n_test=cfg.data.n_test
    downscale=cfg.data.downscale
    if type=='valid':
        downscale*=2
    path=cfg.data.data_root
    bg=cfg.train.bg
    camera_file=os.path.join(path+'/camera.npy')
    K,_,RT,_=load_cam(camera_file,downscale)
    D=None
    imgs = []
    poses = []
    Ks=[]
    c_index=[]
    t_index=[]
    verts=[]
    masks=[]
    rays_o_,rays_d_=[],[]
    if type=='train':
        view=cfg.train.train_view
        ntime=n_train
    elif type=='valid':
        view=[1]
        ntime=1
    else:
        assert test_view1!=-1 and test_view2!=-1 
        ntime=n_test
        view=[test_view1,test_view2]


    for c in range(len(view)):
        i=view[c]
        for j in range(start_frame,ntime+start_frame): 

            if type!='test':
                fname=os.path.join(path+'/{0}/{1:06d}.jpg'.format(i,j))
                mname=os.path.join(path+'/mask/{0}/{1:06d}.png'.format(i,j))
                mask=cv2.imread(mname)
                mask[mask!=0]=1
                if bg==1:
                    background=np.ones_like(mask)*255
                else:
                    background=np.zeros_like(mask)
                image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)*mask+(1-mask)*background 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255 # [H, W, 3]
                H,W=image.shape[:2]
                image=cv2.resize(image,(W//downscale, H//downscale), interpolation=cv2.INTER_AREA)
                mask=cv2.resize(mask,(W//downscale, H//downscale), interpolation=cv2.INTER_AREA)
                H, W = image.shape[:2]
                imgs.append(image)
                masks.append(mask)
            poses.append(np.array(RT[i-1]))
            Ks.append(K[i-1])
            c_index.append(c)
            t_index.append(j-start_frame)
        c2w=RT[i-1]
        intrinsics=K[i-1]

        R=c2w[0:3,0:3]
        T=c2w[0:3,3]
        if type!='test':
            H,W=imgs[0].shape[:2]
            rays_o,rays_d=get_rays(c2w, intrinsics, H,W)
            rays_o_.append(rays_o.astype(np.float32))
            rays_d_.append(rays_d.astype(np.float32))  
    poses = np.stack(poses, axis=0).astype(np.float32)
    Ks = np.stack(Ks, axis=0).astype(np.float32)
    c_index = np.stack(c_index, axis=0).astype(np.float32)
    t_index = np.stack(t_index, axis=0).astype(np.float32)
    if type!='test':
        imgs = np.stack(imgs, axis=0).astype(np.float32)
        masks = np.stack(masks, axis=0).astype(np.float32)
        rays_o_ = np.stack(rays_o_, axis=0).astype(np.float32)
        rays_d_ = np.stack(rays_d_, axis=0).astype(np.float32)


    face_normals=[]
    face_centers=[]
    smpl_paras=[]
    for j in range(start_frame,start_frame+ntime):
        if cfg.test.novel_shape and test_view1!=-1:
            vert_path=os.path.join(path+'/shape_verts/{}.npy'.format(j))
            face_normal_path=os.path.join(path+'/shape_face_normal/{}.npy'.format(j))
            face_center_path=os.path.join(path+'/shape_face_center/{}.npy'.format(j))
        else:
            vert_path=os.path.join(path+'/verts/{}.npy'.format(j))
            face_normal_path=os.path.join(path+'/face_normal/{}.npy'.format(j))
            face_center_path=os.path.join(path+'/face_center/{}.npy'.format(j))
            smpl_para_path=os.path.join(path+'/new_params/{}.npy'.format(j))
        vert=np.load(vert_path)
        face_normal=np.load(face_normal_path)
        face_center=np.load(face_center_path)
        smpl_para=np.load(smpl_para_path,allow_pickle=True).item()
        smpl_para=smpl_para["poses"]
        face_normals.append(face_normal)
        face_centers.append(face_center)
        verts.append(vert)
        smpl_paras.append(smpl_para)
    verts = np.stack(verts, axis=0).astype(np.float32)
    face_normals = np.stack(face_normals, axis=0).astype(np.float32)
    face_centers = np.stack(face_centers, axis=0).astype(np.float32)
    smpl_paras = np.stack(smpl_paras, axis=0).astype(np.float32)
    
    return imgs, masks,poses, Ks,t_index,c_index,verts,face_normals,face_centers,smpl_paras,rays_o_,rays_d_



class NeRFDataset:
    def __init__(self,cfg, device, type='train',test_view1=-1,test_view2=-1):
        super().__init__()
        
        self.cfg = cfg

        self.device = device
        self.type = type # train, valid, test



        self.training = self.type in ['train', 'all', 'trainval']

        data_type=cfg.data.data_type
        n_test=cfg.data.n_test
        start_frame=cfg.data.start_frame
        path=cfg.data.data_root
        H,W=cfg.data.H,cfg.data.W
        downscale=cfg.data.downscale
        if type=='valid':
            downscale*=2
        H,W=H//downscale,W//downscale


        # self.renderer=MeshRenderer(img_size=(H,W))
        
        if type=='train':
            # print('init',cfg.data.n_train)
            self.images,self.masks,self.poses,self.intrinsics,self.t_index,self.c_index,self.verts,self.face_normals,self.face_centers,self.para,self.rays_o,self.rays_d=load_data('train',cfg)

        elif type=='valid':

            self.images,self.masks,self.poses,self.intrinsics,self.t_index,self.c_index,self.verts,self.face_normals,self.face_centers,self.para,self.rays_o,self.rays_d=load_data('valid',cfg)

        else:
            _,_,self.poses,self.intrinsics,self.t_index,self.c_index,self.verts,self.face_normals,self.face_centers,self.para, _ , _ = load_data('test', cfg,test_view1=test_view1,test_view2=test_view2)
            
            #get test pose and rays
            poses=self.poses
            self.H,self.W=H,W
        
            static=False
 
            pose0 = poses[0]
            pose1 = poses[-1]
            #print(pose0,pose1)
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.rays_o=[]
            self.rays_d=[]
            for i in range(n_test):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                rays_o,rays_d=get_rays(pose, self.intrinsics[0], self.H, self.W)
                self.poses.append(pose)
                self.rays_o.append(rays_o)
                self.rays_d.append(rays_d)
            
            self.poses = np.stack(self.poses, axis=0).astype(np.float32)
            self.rays_o = np.stack(self.rays_o, axis=0).astype(np.float32)
            self.rays_d = np.stack(self.rays_d, axis=0).astype(np.float32)
            msks=[]
            # breakpoint()

            self.renderer=MeshRenderer(img_size=(H,W))
            tris=np.load(os.path.join(path,'mesh_properties/smpl/mesh_f.npy')).astype(np.int32)
            tris=batch_to_tensor(tris)
            
            fx,fy,cx,cy=self.intrinsics[0][0,0],self.intrinsics[0][1,1],self.intrinsics[0][0,2],self.intrinsics[0][1,2]
            self.renderer.set_cam_params(fx, fy, cx, cy)
            for j in range(0,n_test):
                verts=batch_to_tensor(self.verts[j])
                cam_pose=batch_to_tensor(self.poses[j])
                rot=cam_pose[:,:3,:3]
                trans=cam_pose[:, :3, 3:].reshape(-1, 1, 3)
                rot[:,-1,:]*=-1
                trans[:,:,-1]*=-1
                verts = torch.bmm(verts, rot.permute(0,2,1))+trans
                mask = self.renderer(verts.float(), tris)
                mask=255.*mask[0].cpu().numpy()
                mask=cv2.flip(mask,0)

                dilate_ratio=0.02
                kernel = np.ones((int(dilate_ratio*self.H), int(dilate_ratio*self.W)), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
                mask[mask!=0]=1
           
                msks.append(mask)
                
            msks = np.stack(msks, axis=0).astype(np.float32)
            self.rough_mask=msks
            self.images=None    




  


    def collate(self, index):
        B = len(index) # a list of length 1
        results = {
            'pose':  torch.from_numpy(self.poses[index]).to(self.device) ,
            'smpl_params':torch.tensor(self.para).to(self.device) ,
            'verts': torch.from_numpy(self.verts).to(self.device) ,
            'face_normals': torch.from_numpy(self.face_normals).to(self.device),
            'face_centers': torch.from_numpy(self.face_centers).to(self.device),
            'c_index': torch.tensor(self.c_index[index]).to(self.device),
            't_index': torch.tensor(self.t_index[index]).to(self.device),
            'index': index,
        }
        if self.type == 'test':
            results['rough_mask']=torch.from_numpy(self.rough_mask[index]).to(self.device)
            results['rays_o']= torch.from_numpy(self.rays_o[index]).to(self.device)
            results['rays_d']= torch.from_numpy(self.rays_d[index]).to(self.device)

        else:
            results['image'] = torch.from_numpy(self.images[index]).to(self.device)
            results['mask'] = torch.from_numpy(self.masks[index]).to(self.device)
            results['rays_o']= torch.from_numpy(self.rays_o).to(self.device)
            results['rays_d']= torch.from_numpy(self.rays_d).to(self.device)



  
        return results

    def dataloader(self):
        size = len(self.poses)
    
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader