from transforms3d.euler import euler2mat
import torch
import numpy as np
from math import radians

def project_point_cloud(pc, xrot=0, yrot=-90, zrot=0, width=512, space=250, diameter=10, normalize=True):
    image = torch.zeros(width, width)
    m = torch.tensor(euler2mat(radians(zrot), radians(yrot), radians(xrot)))
    pc = (m @ pc.T).T

    # We normalize scale to fit points in a unit sphere
    if normalize:
        pc -= pc.mean(dim=0)
        pc /= pc.norm(dim=-1).max()

    # Pre-compute the Gaussian disk
    rad = (diameter-1) / 2.0
    disk = torch.zeros(diameter, diameter)
    for i in range(diameter):
        for j in range(diameter):
            if (i-rad) * (i-rad) + (j-rad) * (j-rad) <= rad**2:
                disk[i, j] = np.exp((-(i-rad)**2 - (j-rad)**2)/(rad**2))
    mask = torch.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer
    zorder = torch.argsort(pc[:, 2])
    pc = pc[zorder, :]
    pc[:, 2] = (pc[:, 2] - torch.min(pc[:, 2])) / (torch.max(pc[:, 2] - torch.min(pc[:, 2])))
    max_depth = torch.max(pc[:, 2])
       
    for i in range(len(pc)):
        j = len(pc) - i - 1
        x = pc[j, 0]
        y = pc[j, 1]
        xc = width/2 + (x*space)
        yc = width/2 + (y*space)
        xc = int(torch.round(xc))
        yc = int(torch.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        try:
            image[px, py] = image[px, py] * 0.7 + dv * (max_depth - pc[j, 2]) * 0.3
        except IndexError:
            continue
    
    image = image / torch.max(image)
    return image

# from transforms3d.euler import euler2mat
# import numpy as np
# import torch

# def draw_point_cloud(pc, canvas_size=500, space=200, diameter=25,
#                      xrot=0, yrot=0, zrot=0, normalize=True):
#     """ Render point cloud to image with alpha channel.
#         Input:
#             points: Nx3 numpy array (+y is up direction)
#         Output:
#             gray image as numpy array of size canvasSizexcanvasSize
#     """
#     image = torch.zeros(canvas_size, canvas_size)
#     pc = euler2mat(zrot, yrot, xrot)
#     pc = (np.dot(M @ points.transpose())).transpose()

#     # Normalize the point cloud
#     # We normalize scale to fit points in a unit sphere
#     if normalize:
#         centroid = np.mean(points, axis=0)
#         points -= centroid
#         furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
#         points /= furthest_distance

#     # Pre-compute the Gaussian disk
#     radius = (diameter-1)/2.0
#     disk = np.zeros((diameter, diameter))
#     for i in range(diameter):
#         for j in range(diameter):
#             if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
#                 disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
#     mask = np.argwhere(disk > 0)
#     dx = mask[:, 0]
#     dy = mask[:, 1]
#     dv = disk[disk > 0]
    
#     # Order points by z-buffer
#     zorder = np.argsort(points[:, 2])
#     points = points[zorder, :]
#     points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
#     max_depth = np.max(points[:, 2])
       
#     for i in range(points.shape[0]):
#         j = points.shape[0] - i - 1
#         x = points[j, 0]
#         y = points[j, 1]
#         xc = canvasSize/2 + (x*space)
#         yc = canvasSize/2 + (y*space)
#         xc = int(np.round(xc))
#         yc = int(np.round(yc))
        
#         px = dx + xc
#         py = dy + yc
        
#         image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
#     image = image / np.max(image)
#     return image

