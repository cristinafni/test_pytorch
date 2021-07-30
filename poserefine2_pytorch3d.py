#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:59:30 2021

Original file: viewer_pytorch3d (author: Florian Vogl)

This enables to create images that have the correct postion (ground truth)
obtained from manual pose.
It also allows to create blurred/noisy images based on ground truth.

Using tutorial: https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering


@author: Cristina
"""

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from skimage.color import rgb2gray
from stl.mesh import Mesh

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.io import load_obj
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
)
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from scipy.ndimage.filters import gaussian_filter
from rotation import rotation_matrix_from_euler
from losses import JaccardLoss

class Pytorch3dViewer():
    """creates render images from stls and poses that can be backpropagated"""

    def __init__(self, img_size: int = 200, device=None):
        if device is None:
            self.device = torch.device("cuda:0")
        else:
            self.device = device
        self.img_size = img_size

        # standard values from `deepautomatch.viewer`
        # as we are only using calibration-corrected poses here, these are the important ones
        std_img_size = 1000 # change?
        std_cal_mm_per_pxl = 0.29 
        std_cal_focal_length = 972

        fov_angle = np.degrees(np.arctan(std_img_size * std_cal_mm_per_pxl / std_cal_focal_length))
        # for render images
        cameras = OpenGLPerspectiveCameras(device=self.device, fov=fov_angle)
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))
        raster_settings = RasterizationSettings(
            image_size=self.img_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = PointLights(
            device=self.device,
            location=((0.0, 1.0, 0.0),),
            ambient_color=((1.0, 0.0, 0.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
        )
        materials = Materials(
            ambient_color=((1, 1, 1),),
            diffuse_color=((1, 1, 1),),
            specular_color=((1, 1, 1),),
            shininess=0,
            device=self.device,
        )
        shader = HardPhongShader(
            lights=lights,
            cameras=cameras,
            materials=materials,
            blend_params=blend_params,
        )
        self.phong_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        # for silhouette images
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        ) # parameters to change!!!

    def mesh_from_stl(self, stl):
        # Load the stl and ignore the textures and materials.
        stl = Mesh.from_file(Path(stl))
        verts = torch.from_numpy(np.reshape(stl.vectors, (-1, 3)))
        assert verts.shape[0] % 3 == 0
        faces = torch.Tensor([[i * 3, i * 3 + 1, i * 3 + 2] for i in range(verts.shape[0] // 3)])
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        implant_mesh = Meshes(verts=[verts.to(self.device)], faces=[faces.to(self.device)], textures=textures.to(self.device))
        return implant_mesh
    
    def join_meshes(self, femur_stl, tibia_stl):
        # Load the stl and ignore the textures and materials.
        femur_stl = Mesh.from_file(Path(femur_stl))
        femur_verts = torch.from_numpy(np.reshape(femur_stl.vectors, (-1, 3)))
        assert femur_verts.shape[0] % 3 == 0
        femur_faces = torch.Tensor([[i * 3, i * 3 + 1, i * 3 + 2] for i in range(femur_verts.shape[0] // 3)])
        # Initialize each vertex to be white in color.
        femur_verts_rgb = torch.ones_like(femur_verts)[None]  # (1, V, 3)
        #femur_textures = TexturesVertex(femur_verts_features=femur_verts_rgb)
        
        # Load the stl and ignore the textures and materials.
        tibia_stl = Mesh.from_file(Path(tibia_stl))
        tibia_verts = torch.from_numpy(np.reshape(tibia_stl.vectors, (-1, 3)))
        assert femur_verts.shape[0] % 3 == 0
        tibia_faces = torch.Tensor([[i * 3, i * 3 + 1, i * 3 + 2] for i in range(tibia_verts.shape[0] // 3)])
        # Initialize each vertex to be white in color.
        tibia_verts_rgb = torch.ones_like(tibia_verts)[None]  # (1, V, 3)
        #tibia_textures = TexturesVertex(tibia_verts_features=tibia_verts_rgb)
        
        # verts = torch.Tensor(np.concatenate((femur_verts, tibia_verts), axis=0))
        # faces = torch.Tensor(np.concatenate((femur_verts, tibia_verts), axis=0))
        # verts_rgb = np.concatenate((femur_verts_rgb, tibia_verts_rgb), axis=1)
        verts = torch.cat((femur_verts, tibia_verts), axis=0)
        faces = torch.cat((femur_verts, tibia_verts), axis=0)
        verts_rgb = torch.cat((femur_verts_rgb, tibia_verts_rgb), axis=1)
        textures = TexturesVertex(verts_features= torch.Tensor(verts_rgb))
        
        implant_mesh_joined = Meshes(verts=[verts.to(self.device)], faces=[faces.to(self.device)], textures=textures.to(self.device))
        return implant_mesh_joined

    def scene_snapshot(
        self,
        femur_stls,
        tibia_stls,
        femur_R: torch.Tensor,
        femur_T: torch.Tensor,
        tibia_R: torch.Tensor,
        tibia_T: torch.Tensor,
    ) -> torch.Tensor:
        """for batch_size N
        femur_stl, tibia_stl: string/path list of length N
        femur_r, tibia_r: (N, 3, 3), same as `deepautomatch.viewer´
        femur_T, tibia_T: (N, 3), same as `deepautomatch.viewer´

        Note that pytorch3D has a different coordinate convention from openGl
        (and thus flumatch and viewer.py) so that we need to change some things
        """
        if not isinstance(femur_stls, list):
            # just one string given, for would loop over characters
            femur_stls = [femur_stls]
        if not isinstance(tibia_stls, list):
            tibia_stls = [tibia_stls]
        femur_meshes = []
        tibia_meshes = []
        for stl in femur_stls:
            femur_meshes.append(self.mesh_from_stl(stl))
        for stl in tibia_stls:
            tibia_meshes.append(self.mesh_from_stl(stl))

        femur_meshes = join_meshes_as_batch(femur_meshes)
        tibia_meshes = join_meshes_as_batch(tibia_meshes)

        # adjust for differences in coordinate conventions
        m = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(femur_R.device)
        femur_T = torch.matmul(femur_T, m)
        tibia_T = torch.matmul(tibia_T, m)
        femur_R = femur_R.permute(0, 2, 1)
        tibia_R = tibia_R.permute(0, 2, 1)
        femur_R = torch.matmul(femur_R, m)
        tibia_R = torch.matmul(tibia_R, m)

        femur_imgs = self.phong_renderer(
            meshes_world=femur_meshes, R=femur_R.to(self.device), T=femur_T.to(self.device)
        )
        # get rid of alpha channel
        femur_imgs = femur_imgs[..., :3]
        tibia_imgs = self.phong_renderer(
            meshes_world=tibia_meshes, R=tibia_R.to(self.device), T=tibia_T.to(self.device)
        )
        tibia_imgs = tibia_imgs[..., :3]
        # combine the femur_img with its tibia_img
        combined = torch.max(femur_imgs, tibia_imgs)
        if torch.isnan(combined).any():
            logging.error("NaN encountered during creating scene snapshot")
        return combined

    def scene_snapshot_shadow(
        self,
        femur_stls,
        tibia_stls,
        femur_R: torch.Tensor,
        femur_T: torch.Tensor,
        tibia_R: torch.Tensor,
        tibia_T: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """same as scene snapshot, but collapse all channels
        returns (batch_nr, img_size, img_size, 1)"""
        # returns (batch_nr, img_size, img_size)
        img = self.scene_snapshot(femur_stls, tibia_stls, femur_R, femur_T, tibia_R, tibia_T)
        # collapse all 3 channels
        img = img.sum(dim=3)
        assert img.shape == (femur_R.shape[0], self.img_size, self.img_size)
        img = img / img.max() 
        return img

    def scene_snapshot_one_component(
        self,
        stls,
        R: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """for batch_size N
        femur_stl, tibia_stl: string/path list of length N
        femur_r, tibia_r: (N, 3, 3), same as `deepautomatch.viewer´
        femur_T, tibia_T: (N, 3), same as `deepautomatch.viewer´

        Note that pytorch3D has a different coordinate convention from openGl
        (and thus flumatch and viewer.py) so that we need to change some things
        """
        if not isinstance(stls, list):
            # just one string given, for would loop over characters
            stls = [stls]
    
        meshes = []
   
        for stl in stls:
            meshes.append(self.mesh_from_stl(stl))

        meshes = join_meshes_as_batch(meshes)
        
        # adjust for differences in coordinate conventions
        m = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(R.device)
        T = torch.matmul(T, m)
        R = R.permute(0, 2, 1)
        R = torch.matmul(R, m)

        imgs = self.phong_renderer(
            meshes_world=meshes, R=R.to(self.device), T=T.to(self.device)
        )
        # get rid of alpha channel
        imgs1 = imgs[..., :3]
        
        return imgs1
    
    def scene_snapshot_shadow_one_component(
        self,
        stls,
        R: torch.Tensor,
        T: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """same as scene snapshot, but collapse all channels
        returns (batch_nr, img_size, img_size, 1)"""
        # returns (batch_nr, img_size, img_size)
        img = self.scene_snapshot_one_component(stls, R, T)
        # collapse all 3 channels
        img = img.sum(dim=3)
        assert img.shape == (R.shape[0], self.img_size, self.img_size)
        img = img / img.max() 
        return img

def create_ground_truth(index):
    device = torch.device("cpu")
    viewer = Pytorch3dViewer(200, device=device)
    d = Path("./data_files/stl_3D_models")
    femur_stls = [d / "cr_fem_4_r_narrow_mm.stl"]
    tibia_stls = [d / "cr_tib_modular_3_r_narrow.stl"]
    # create list 
    data_folder = "./data_files/csv_files/flumatch_data.csv"
    f = open(data_folder)
    load_data_info = pd.read_csv(f)
    
    reference_list = []
    
    for i in range(index):
        
        poses = {
            "femur_rx": [load_data_info['femur_rx'][i]],
            "femur_ry": [load_data_info['femur_ry'][i]],
            "femur_rz": [load_data_info['femur_rz'][i]],
            "femur_tx": [load_data_info['femur_tx'][i]],
            "femur_ty": [load_data_info['femur_ty'][i]],
            "femur_tz": [load_data_info['femur_tz'][i]],
            "tibia_rx": [load_data_info['tibia_rx'][i]],
            "tibia_ry": [load_data_info['tibia_ry'][i]],
            "tibia_rz": [load_data_info['tibia_ry'][i]],
            "tibia_tx": [load_data_info['tibia_tx'][i]],
            "tibia_ty": [load_data_info['tibia_ty'][i]],
            "tibia_tz": [load_data_info['tibia_tz'][i]],
        }
    
        for k, v in poses.items():
            poses[k] = torch.Tensor(v)
        mm_per_pxl = 0.2876
        femur_R = rotation_matrix_from_euler(
            poses["femur_rx"], poses["femur_ry"], poses["femur_rz"]
        )
        tibia_R = rotation_matrix_from_euler(
            poses["tibia_rx"], poses["tibia_ry"], poses["tibia_rz"]
        )
        """ for later refactoring of rotation.py
        eulers = torch.cat([poses["femur_rz"], poses["femur_rx"], poses["femur_ry"]]).T * np.pi/180
        femur_R_2 = transforms.euler_angles_to_matrix(eulers, "ZXY")
        print(femur_R)
        print(femur_R_2)
        """
        
        femur_T = torch.cat(
            [torch.unsqueeze(v, dim=0) for v in [poses["femur_tx"], poses["femur_ty"], poses["femur_tz"]]]
        ).T
        tibia_T = torch.cat(
            [torch.unsqueeze(v, dim=0) for v in [poses["tibia_tx"], poses["tibia_ty"], poses["tibia_tz"]]]
        ).T
        py3d_imgs_shadow = viewer.scene_snapshot_shadow(
            femur_stls, tibia_stls, femur_R, femur_T, tibia_R, tibia_T
        )
        py3d_imgs_shadow = py3d_imgs_shadow.cpu().numpy()
        reference_list.append(py3d_imgs_shadow[0].squeeze())
    
    return reference_list

def create_ground_truth_one_component(index, component ='tibia'):
    device = torch.device("cpu")
    viewer = Pytorch3dViewer(200, device=device)
    d = Path("./data_files/stl_3D_models")
    femur_stls = [d / "cr_fem_4_r_narrow_mm.stl"]
    tibia_stls = [d / "cr_tib_modular_3_r_narrow.stl"]

    # create list 
    data_folder = "./data_files/csv_files/flumatch_data.csv"
    f = open(data_folder)
    load_data_info = pd.read_csv(f)
    
    reference_list = []
    
    if component == 'tibia':
        for i in range(index):
            
            poses = {
                "tibia_rx": [load_data_info['tibia_rx'][i]],
                "tibia_ry": [load_data_info['tibia_ry'][i]],
                "tibia_rz": [load_data_info['tibia_ry'][i]],
                "tibia_tx": [load_data_info['tibia_tx'][i]],
                "tibia_ty": [load_data_info['tibia_ty'][i]],
                "tibia_tz": [load_data_info['tibia_tz'][i]],
            }
        
            for k, v in poses.items():
                poses[k] = torch.Tensor(v)
            
            mm_per_pxl = 0.2876
            
            tibia_R = rotation_matrix_from_euler(
                poses["tibia_rx"], poses["tibia_ry"], poses["tibia_rz"]
            )
            """ for later refactoring of rotation.py
            eulers = torch.cat([poses["femur_rz"], poses["femur_rx"], poses["femur_ry"]]).T * np.pi/180
            femur_R_2 = transforms.euler_angles_to_matrix(eulers, "ZXY")
            print(femur_R)
            print(femur_R_2)
            """
            
            tibia_T = torch.cat(
                [torch.unsqueeze(v, dim=0) for v in [poses["tibia_tx"], poses["tibia_ty"], poses["tibia_tz"]]]
            ).T
            py3d_imgs_shadow = viewer.scene_snapshot_shadow_one_component(tibia_stls, tibia_R, tibia_T)
            py3d_imgs_shadow = py3d_imgs_shadow.cpu().numpy()
            reference_list.append(py3d_imgs_shadow[0].squeeze())
        
    elif component == 'femur':
        for i in range(index):
            
            poses = {
                "femur_rx": [load_data_info['femur_rx'][i]],
                "femur_ry": [load_data_info['femur_ry'][i]],
                "femur_rz": [load_data_info['femur_rz'][i]],
                "femur_tx": [load_data_info['femur_tx'][i]],
                "femur_ty": [load_data_info['femur_ty'][i]],
                "femur_tz": [load_data_info['femur_tz'][i]],
                }
            
            for k, v in poses.items():
                poses[k] = torch.Tensor(v)
                
            mm_per_pxl = 0.2876
            femur_R = rotation_matrix_from_euler(
                poses["femur_rx"], poses["femur_ry"], poses["femur_rz"]
            )
            femur_T = torch.cat(
                [torch.unsqueeze(v, dim=0) for v in [poses["femur_tx"], poses["femur_ty"], poses["femur_tz"]]]
            ).T
            py3d_imgs_shadow = viewer.scene_snapshot_shadow_one_component(femur_stls, femur_R, femur_T)
            py3d_imgs_shadow = py3d_imgs_shadow.cpu().numpy()
            reference_list.append(py3d_imgs_shadow[0].squeeze())
    
    return reference_list 

def create_blurred_shadows(index):
    device = torch.device("cpu")
    # reference_list = create_ground_truth(index=1)
    blurred_list = []
    for a in reference_list:
        blurred = gaussian_filter(a, sigma=0.5)
        blurred_list.append(blurred)
    return blurred_list

class Model(nn.Module):
    def __init__(self, meshes, renderer, phong_renderer, image_ref, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.meshes = meshes
        self.renderer = renderer
        self.phong_renderer = phong_renderer
        # self.image_ref = image_ref
        
        #Get the silhouette of the reference image by finding all non-black pixel values. 
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        # self.register_buffer('image_ref', image_ref)
        
        data_folder = "./data_files/csv_files/flumatch_data.csv"
        f = open(data_folder)
        load_data_info = pd.read_csv(f)
        viewer = Pytorch3dViewer(200, device=device)

        poses = {
                "femur_rx": [load_data_info['femur_rx'][0]],
                "femur_ry": [load_data_info['femur_ry'][0]],
                "femur_rz": [load_data_info['femur_rz'][0]],
                "femur_tx": [load_data_info['femur_tx'][0]+5],
                "femur_ty": [load_data_info['femur_ty'][0]+5],
                "femur_tz": [load_data_info['femur_tz'][0]+5],
                }
            
        for k, v in poses.items():
            poses[k] = torch.Tensor(v) 
        #mm_per_pxl = 0.2876
        self.R = rotation_matrix_from_euler(
            poses["femur_rx"], poses["femur_ry"], poses["femur_rz"])
        self.T = torch.cat(
            [torch.unsqueeze(v, dim=0) for v in [poses["femur_tx"], poses["femur_ty"], poses["femur_tz"]]]
        ).T

        self.m = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(self.R.device)
        self.T = torch.matmul(self.T, self.m)
        self.R = self.R.permute(0, 2, 1)
        self.R = torch.matmul(self.R, self.m)
       
        # Parameters to optimise
        self.R = nn.Parameter(self.R.to(self.device))
        self.T = nn.Parameter(self.T.to(self.device))
        
        # Reference 
        
        poses_ref = {
             "femur_rx": [load_data_info['femur_rx'][0]],
             "femur_ry": [load_data_info['femur_ry'][0]],
             "femur_rz": [load_data_info['femur_rz'][0]],
             "femur_tx": [load_data_info['femur_tx'][0]],
             "femur_ty": [load_data_info['femur_ty'][0]],
             "femur_tz": [load_data_info['femur_tz'][0]],
             }
            
        for k, v in poses_ref.items():
            poses_ref[k] = torch.Tensor(v) 
        #mm_per_pxl = 0.2876
        self.R_ref = rotation_matrix_from_euler(
            poses_ref["femur_rx"], poses_ref["femur_ry"], poses_ref["femur_rz"])
        self.T_ref = torch.cat(
            [torch.unsqueeze(v, dim=0) for v in [poses_ref["femur_tx"], poses_ref["femur_ty"], poses_ref["femur_tz"]]]
        ).T
        
        # py3d_imgs_shadow = viewer.scene_snapshot_shadow_one_component(femur_stls, femur_R, femur_T)
        # py3d_imgs_shadow = py3d_imgs_shadow.cpu().numpy()
        self.m_ref = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(self.R_ref.device)
        self.T_ref = torch.matmul(self.T_ref, self.m_ref)
        self.R_ref = self.R_ref.permute(0, 2, 1)
        self.R_ref = torch.matmul(self.R_ref, self.m_ref)
        self.image_ref = self.renderer(meshes_world=self.meshes, R=self.R_ref, T=self.T_ref)
    
    def forward(self):

        # print(type(self.offset_R))
        image1 = self.renderer(meshes_world=self.meshes, R=self.R, T=self.T)
        # image1 = image1.detach().squeeze().cpu().numpy()
        # image1[image1!=0] = 1
        image_ref = self.image_ref
        
        # Calculate the silhouette loss
        loss_j = JaccardLoss()
        loss = loss_j.forward(pred = image1, target = self.image_ref)
        
        # loss = torch.sum((image1 - self.image_ref) ** 2)
        # loss = torch.sum((image1[...,3] - self.image_ref) ** 2)
        
        return loss, image1, image_ref
    
def rescale_array(image):
    rescale = 2.*(image - np.min(image))/np.ptp(image)-1
    change = np.int8(rescale)
    return rescale    
    
def optimisation():
    device = torch.device("cpu")
    # blurred_list = create_blurred_shadows(index=1)
    reference_list =  create_ground_truth_one_component(index=1, component ='femur')
    d = Path("./data_files/stl_3D_models")
    femur_stls = [d / "cr_fem_4_r_narrow_mm.stl"]
    tibia_stls = [d / "cr_tib_modular_3_r_narrow.stl"]
    viewer = Pytorch3dViewer(200, device=device)
    
    #for b in blurred_list:
    for ref_img in reference_list:
        plt.figure()
        plt.grid(False)
        plt.imshow(ref_img)
       
        # # Initialize a model using the renderer, mesh and reference image
        ref_img[ref_img!=1] = 0
        model1 = Model(viewer.mesh_from_stl(femur_stls[0]),viewer.silhouette_renderer, viewer.phong_renderer, ref_img, device=device)
        optimizer = torch.optim.Adam(model1.parameters(), lr=0.05) #change?
        
        plt.figure(figsize=(10, 10))
        _, image_init, image_ref = model1()
        image_start = image_init.detach().squeeze().cpu().numpy()[..., 3]
        image_start[image_start!=0] = 1
       
        plt.subplot(1, 2, 1)
        # plt.imshow(image_init, cmap="gray")
        plt.imshow(image_start, cmap="gray")
        plt.grid(False)
        plt.title("Starting position")
        
        image_new_ref = model1.image_ref
        image_new_ref = image_new_ref.detach().squeeze().cpu().numpy()[..., 3]
        image_new_ref[image_new_ref!=0] = 1
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_new_ref, cmap="gray")
        plt.grid(False)
        plt.title("Reference silhouette");
       
        # Optimisation loop
        loss_list = []
        iterations = range(1000)
        
        for i in iterations: # change number
            print(i)
            optimizer.zero_grad()
            loss, _, _ = model1()  
            loss.backward()
            optimizer.step()
            
            # if loss.item() < 100: 
            #     break
            # if i % 10 == 0:
            image = viewer.phong_renderer(meshes_world=model1.meshes.clone(), R=model1.R, T=model1.T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            # image = rescale_array(image)
            image[image!=0] = 1
            # image = img_as_ubyte(image)
            loss_list.append(loss.data)
            
        plt.figure()
        plt.imshow(rgb2gray(image), cmap='gray')
        plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
        plt.grid("off")
        plt.axis("off")
            
        plt.figure()
        plt.plot(iterations, loss_list)
        plt.xlabel('Iterations')
        plt.ylabel('Jaccard Loss')
        plt.title("Loss VS iterations")
       

def main():
    device = torch.device("cpu")
    optimisation()
     

if __name__ == "__main__":
    main()