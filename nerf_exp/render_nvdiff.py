import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr


def ndc_projection(fx,fy,cx,cy, img_size, n=0.1, f=5.0):
    """n -- near, f -- far"""
    height, width = img_size
    # fmt: off
    res = [
        [
            [2 * fx / (width - 1),                      0, 1 - 2 * cx / (width - 1) ,                      0],
            [                   0, -2 * fy / (height - 1), 1 - 2 * cy / (height - 1),                      0],
            [                   0,                      0,        -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [                   0,                      0,                        -1,                      0],
        ]
    ]
    # fmt: on
    return np.array(res, dtype=np.float32)


def compute_vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    ex_faces = (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    faces = faces + ex_faces # expanded faces
    # print('exfaces : ', ex_faces.shape)
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)
    # print(faces, vertices_faces)

    normals.index_add_(0, faces[:, 1].long(), torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(), torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

class MeshRenderer(nn.Module):
    def __init__(self, img_size=[512, 512]):
        super(MeshRenderer, self).__init__()

        self.img_size = img_size
        self.glctx = None

    def set_cam_params(self, fx,fy,cx,cy):
        self.ndc_proj = torch.as_tensor(ndc_projection(fx,fy,cx,cy, self.img_size))
        self.ndc_proj.requires_grad_(False)

    def forward(self, verts, tris):
        """
        Parameters:
            verts           -- torch.tensor, size (B, N, 3)
            tris            -- torch.tensor, size (B, M, 3) or (M, 3), triangles
        """
        batch_size=verts.shape[0]
        device=verts.device
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d verts, the direction of y is the same as v
        if verts.shape[-1] == 3:
            verts = torch.cat([verts, torch.ones([*verts.shape[:2], 1]).to(device)], dim=-1)
        
        verts_ndc = verts @ ndc_proj.permute(0, 2, 1)
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext(device=device)
        # breakpoint()
        rast_out, _rast_db = dr.rasterize(self.glctx, verts_ndc.contiguous(), tris[0], resolution=self.img_size)
        # breakpoint()
        ## rast_out.shape (bs, height, width, 4), 4 means (u, v, z/w, triangle_id)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)

    
        return  mask
