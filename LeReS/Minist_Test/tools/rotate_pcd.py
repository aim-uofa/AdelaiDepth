
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading point clouds|
import numpy as np
from plyfile import PlyData, PlyElement

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    look_at_rotation,
    FoVPerspectiveCameras,
)
import imageio


device = torch.device("cuda:0")


def load_pcd_color(path):
    #pointcloud = np.load(path)
    pointcloud = PlyData.read(path).elements[0].data

    verts = np.array([[data[0], data[1], data[2]] for data in pointcloud])
    rgb = np.array([[data[3], data[4], data[5]] for data in pointcloud]) / 256.0
    verts[:, 1] = -verts[:, 1]
    verts[:, 0] = -verts[:, 0]

    verts /= verts.max()
    verts = verts - verts.mean(axis=0)

    verts = torch.Tensor(verts).to(device)
    rgb = torch.Tensor(rgb).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])
    return point_cloud

def load_pcd_color2(path):
    #pointcloud = np.load(path)
    # Load point cloud
    pointcloud = np.load(path)
    verts = torch.Tensor(pointcloud['verts']).to(device)

    rgb = torch.Tensor(pointcloud['rgb']).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])
    return point_cloud

def rotate_pcd(point_cloud, angle):
    rotate_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0]
                              [-np.sin(angle), 0, np.cos(angle)]])
    #cnt = point_cloud.mean(axis=)

def render_single(point_cloud, azim=0):
    # Initialize a camera.
    R, T = look_at_view_transform(0.5, 10, azim=azim)
    #cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
    cameras = FoVPerspectiveCameras(fov=50.0, device=device, R=R, T=T)
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.008,
        points_per_pixel=50
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
        )
    images = renderer(point_cloud)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.grid("off")
    # plt.axis("off")
    # plt.show()
    img_out = images[0, ...].cpu().numpy()
    img_out = (img_out * 256).astype(np.uint8)
    return img_out

def render_batches(point_cloud):
    # Initialize a camera.
    batch_size = 10
    point_clouds = point_cloud.extend(batch_size)
    azim = torch.linspace(-180, 180, batch_size)

    R, T = look_at_view_transform(-1, 0, azim=azim)
    #cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
    cameras = FoVPerspectiveCameras(fov=50.0, device=device, R=R, T=T)
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.008,
        points_per_pixel=50
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
        )
    images = renderer(point_clouds)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.grid("off")
    plt.axis("off")
    plt.show()

def render_create_GIF(point_cloud, filename):
    # We will save images periodically and compose them into a GIF.
    filename_output = filename + ".gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.01)
    for i in range(-180, 180, 2):
        image = render_single(point_cloud, i)
        writer.append_data(image)
    writer.close()

if __name__ == '__main__':
    # path = '/mnt/data/WeiYin/publications/cvpr2021/first_page_fig/first_page_fig_results/8-rgb-recovershift.ply'
    path = '/home/gk-ai/code/mesh710.ply'
    point_cloud = load_pcd_color(path)
    name = path.split('/')[-1][:-4]
    #point_cloud = load_pcd_color2('/home/yvan/DeepLearning/Depth/MultiDepth/A-othertools/Render_PCD/data/PittsburghBridge/pointcloud.npz')
    #render_single(point_cloud)
    render_create_GIF(point_cloud, name)