import argparse
import numpy as np
import torch
import plyfile
import skimage.measure
from tqdm import tqdm 
import yaml
import os.path as osp
import skimage
import time 

def convert_sigma_samples_to_ply(
    input_3d_sigma_array: np.ndarray,
    voxel_grid_origin,
    volume_size,
    ply_filename_out,
    level=5.0,
    offset=None,
    scale=None,):
    """
    Convert density samples to .ply
    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :volume_size: a list of three floats
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        input_3d_sigma_array, level=level, spacing=volume_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % str(ply_filename_out))
    ply_data.write(ply_filename_out)

    print(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def generate_and_write_mesh(bounding_box, num_pts, levels, chunk, device, ply_root, **render_kwargs):
    """
    Generate density grid for marching cubes
    :bounding_box: bounding box for meshing 
    :num_pts: Number of grid elements on each axis
    :levels: list of levels to write meshes for 
    :ply_root: string, path of the folder to save meshes to
    """

    near = render_kwargs['near']
    bb_min = (*(bounding_box[0] + near).cpu().numpy(),)
    bb_max = (*(bounding_box[1] - near).cpu().numpy(),)

    x_vals = torch.tensor(np.linspace(bb_min[0], bb_max[0], num_pts))
    y_vals = torch.tensor(np.linspace(bb_min[1], bb_max[1], num_pts))
    z_vals = torch.tensor(np.linspace(bb_min[2], bb_max[2], num_pts))

    xs, ys, zs = torch.meshgrid(x_vals, y_vals, z_vals, indexing = 'ij')
    coords = torch.stack((xs, ys, zs), dim = -1)

    coords = coords.view(1, -1, 3).type(torch.FloatTensor).to(device)
    dummy_viewdirs = torch.tensor([0, 0, 1]).view(-1, 3).type(torch.FloatTensor).to(device)

    nerf_model = render_kwargs['network_fine']
    radiance_field = render_kwargs['network_query_fn']

    chunk_outs = []

    for k in tqdm(range(coords.shape[1] // chunk), desc = "Retrieving densities at grid points"):
        chunk_out = radiance_field(coords[:, k * chunk: (k + 1) * chunk, :], dummy_viewdirs, nerf_model)
        chunk_outs.append(chunk_out.detach().cpu().numpy()[:, :, -1])

    if not coords.shape[1] % chunk == 0:
        chunk_out = radiance_field(coords[:, (k+1) * chunk: , :], dummy_viewdirs, nerf_model)
        chunk_outs.append(chunk_out.detach().cpu().numpy()[:, :, -1])

    input_sigma_arr = np.concatenate(chunk_outs, axis = -1).reshape(num_pts, num_pts, num_pts)

    for level in levels:
        try:
            sizes = (abs(bounding_box[1] - bounding_box[0]).cpu()).tolist()
            convert_sigma_samples_to_ply(input_sigma_arr, list(bb_min), sizes, osp.join(ply_root, f"test_mesh_{level}.ply"), level = level)
        except ValueError:
            print(f"Density field does not seem to have an isosurface at level {level} yet")
