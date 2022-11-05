from transforms3d.euler import euler2mat
import numpy as np

def draw_point_cloud(points, canvasSize=1028, space=500, diameter=20, xrot=0, yrot=180, zrot=90, normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if points is None or points.shape[0] == 0:
        return image

    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])
       
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
    image = image / np.max(image)
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

