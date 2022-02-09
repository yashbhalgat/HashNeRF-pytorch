import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import get_voxel_vertices, ngp_hash


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512, 
                num_hashes=1, pool_over_hashes=False, which_hash='yash'):
        super().__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        
        # Hashing parameters
        self.num_hashes = num_hashes
        self.pool_over_hashes = pool_over_hashes
        self.which_hash = which_hash

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        # Initialization for embeddings
        init_debug_vector = torch.zeros((8388608, 2)).uniform_(-0.0001, 0.0001)

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level, sparse=True) 
            for i in range(n_levels)
        ])

        # # Custom initialization
        init_debug_vector = init_debug_vector.to(self.embeddings[0].weight.device)
        for i in range(n_levels):
            hashmap_size = 2**self.log2_hashmap_size
            self.embeddings[i].weight.data = init_debug_vector[i * hashmap_size: (i+1) * hashmap_size]

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i).int().item()

            # Get the indices using our hash function
            # `all_hashed_voxel_indices` has size (N_rays, 8, num_hashes)
            voxel_min_vertex, voxel_max_vertex, all_hashed_voxel_indices = get_voxel_vertices(
                x, self.bounding_box, resolution, self.log2_hashmap_size, self.num_hashes, self.which_hash)

            # Get the embeddings from the hash table
            voxel_embedds = self.embeddings[i](all_hashed_voxel_indices)  # (N_rays, 8, num_hashes, D)

            # Pool over embeddings
            if self.pool_over_hashes:
                voxel_embedds = torch.max(voxel_embedds, dim=-2, keepdim=True).values  # (N_rays, 8, 1, D)

            # Flatten all the hashed embeddings together
            voxel_embedds = torch.flatten(voxel_embedds, start_dim=2, end_dim=3)  # (N_rays, 8, num_hashes * D)

            # Trilinear interpolation
            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        x_embedded_all = torch.cat(x_embedded_all, dim=-1)

        return x_embedded_all


class ParallelHashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512, 
                num_hashes=1, pool_over_hashes=False, which_hash='yash'):
        super().__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        
        # Offsets for boxes
        self.box_offsets = torch.tensor(
            [[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
            dtype=torch.int32, device='cuda'
        ).reshape(1, 1, 8, 1, 3)
        
        # Hashing
        self.num_hashes = num_hashes
        self.pool_over_hashes = pool_over_hashes
        self.hash_offsets = torch.arange(num_hashes).reshape(1, 1, num_hashes, 1) * 53  # MAGIC NUMBER  # TODO: Change this! It doesn't make sense!
        self.which_hash = which_hash

        # Resolutions
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        # Initialization
        init_debug_vector = torch.zeros((8388608, 2)).uniform_(-0.0001, 0.0001)

        # Embedding table: a single huge table
        self.hashmap_size = 2 ** self.log2_hashmap_size
        self.embeddings = nn.Embedding(self.hashmap_size * n_levels, self.n_features_per_level, sparse=True)

        # # Custom uniform initialization
        self.embeddings.weight.data = init_debug_vector.to(self.embeddings.weight.device)

        
    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: (N_levels, N_rays, 3)
        voxel_min_vertex: (N_levels, N_rays, 3)
        voxel_max_vertex: (N_levels, N_rays, 3)
        voxel_embedds: (N_levels, N_rays, 8, 2)
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # (N_levels, N_rays, 3)

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,:,0]*(1-weights[:,:,0][:,:,None]) + voxel_embedds[:,:,4]*weights[:,:,0][:,:,None]
        c01 = voxel_embedds[:,:,1]*(1-weights[:,:,0][:,:,None]) + voxel_embedds[:,:,5]*weights[:,:,0][:,:,None]
        c10 = voxel_embedds[:,:,2]*(1-weights[:,:,0][:,:,None]) + voxel_embedds[:,:,6]*weights[:,:,0][:,:,None]
        c11 = voxel_embedds[:,:,3]*(1-weights[:,:,0][:,:,None]) + voxel_embedds[:,:,7]*weights[:,:,0][:,:,None]

        # step 2
        c0 = c00*(1-weights[:,:,1][:,:,None]) + c10*weights[:,:,1][:,:,None]
        c1 = c01*(1-weights[:,:,1][:,:,None]) + c11*weights[:,:,1][:,:,None]

        # step 3
        c = c0*(1-weights[:,:,2][:,:,None]) + c1*weights[:,:,2][:,:,None]

        return c

    def forward(self, x):
        """The input `x` is a batch of 3D point positions: (B, 3)"""
        box_min, box_max = self.bounding_box  # (B, 3), (B, 3)
        x = x[None]  # (1, N_rays, 3)

        # Check
        if not torch.all(x < box_max) or not torch.all(x > box_min):
            print("ALERT: some points are outside bounding box. Clipping them!")
            import pdb; pdb.set_trace()

        # Parallel
        resolution = torch.floor(self.base_resolution * self.b ** torch.arange(self.n_levels))  # (N_levels, )

        ### Get voxel vertex indices

        # Get grid info
        resolution = resolution[:, None, None]  # (N_levels, 1, 1)
        box_max = box_max[None, None]  # (1, 1, 3)
        box_min = box_min[None, None]  # (1, 1, 3)
        grid_size = (box_max - box_min) / resolution  # (N_levels, 1, 3)
        bottom_left_idx = ((x - box_min) / grid_size).int()  # (N_levels, N_rays, 3)
        voxel_min_vertex = bottom_left_idx * grid_size + box_min  # (N_levels, N_rays, 3)
        voxel_max_vertex = voxel_min_vertex + grid_size  # (N_levels, N_rays, 3) -- offset by a single grid coordinate

        # Parallel code with multiple hashes
        bottom_left_idxs = bottom_left_idx[:, :, None, None, :]  # (N_levels, N_rays, 1, 1, 3)
        bottom_left_idxs = bottom_left_idxs + self.box_offsets  # (N_levels, N_rays, 8, 1, 3) box coordinates
        bottom_left_idxs = bottom_left_idxs + self.hash_offsets  # (N_levels, N_rays, 8, num_hashes, 3) box "coordinates"
        all_hashed_voxel_indices = ngp_hash(bottom_left_idxs, self.log2_hashmap_size)  # N_levels, N_rays, 8, num_hashes) integer embedding indices 

        # Get the embeddings from the hash table
        all_hashed_voxel_indices = all_hashed_voxel_indices + torch.arange(self.n_levels).reshape(self.n_levels, 1, 1, 1) * self.hashmap_size  # offset for different levels
        voxel_embedds = self.embeddings(all_hashed_voxel_indices)  # (N_levels, N_rays, 8, num_hashes, D)
        
        # Pool over embeddings
        if self.pool_over_hashes:
            voxel_embedds = torch.max(voxel_embedds, dim=-2, keepdim=True).values  # (N_levels, N_rays, 8, 1, D)

        # Flatten all the hashed embeddings together
        voxel_embedds = torch.flatten(voxel_embedds, start_dim=-2, end_dim=-1)  # (N_levels, N_rays, 8, num_hashes * D)

        # Trilinear interpolation
        x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)  # (N_levels, N_rays, num_hashes * D)
        
        # Reshape and return
        x_embedded = torch.flatten(x_embedded.permute(1, 2, 0), -2, -1)
        return x_embedded
