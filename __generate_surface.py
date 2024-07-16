import numpy as np
from scipy.spatial import Delaunay
import trimesh
from reservoir_grid import Stratigraphy_Grid
def generate_structure_of_anticline_example(): 


    ## basic settings
    epsilon = 1E-5
    x_0, x_1 = 0.0, 10_000
    y_0, y_1 = 0.0, 10_000
    bin_size = (x_1-x_0)/50
    ## generate surfaces
    x_bins = np.arange(x_0,x_1,bin_size)
    y_bins = np.arange(y_0,y_1,bin_size)
    mesh_x, mesh_y = np.meshgrid(x_bins, y_bins)


    ## Version #3 - anticline (24.7.16)
    depth = 3020
    thickness = 150
    diff = 250
    radius = 5000
    center = 5000
    power = 1

    beta = (np.pi/radius)
    mesh_r = np.sqrt((mesh_x-center)**2 + (mesh_y-center)**2)
    mesh_r_scaled = beta*mesh_r
    mesh_r_scaled [mesh_r_scaled>np.pi] = np.pi
    mesh_z = -(depth + thickness - (thickness*0.5 + thickness*0.5*np.cos(mesh_r_scaled)))
    mesh_z_2 = -(depth + thickness - (thickness*0.5 + thickness*0.5*np.cos(mesh_r_scaled) - diff))
    ## Triangulation
    pts = np.concatenate([mesh_x.reshape(-1,1), mesh_y.reshape(-1,1)],axis = 1)
    pts_3d = np.concatenate([mesh_x.reshape(-1,1), mesh_y.reshape(-1,1), mesh_z.reshape(-1,1)],axis = 1)
    pts_3d_2 = np.concatenate([mesh_x.reshape(-1,1), mesh_y.reshape(-1,1), mesh_z_2.reshape(-1,1)],axis = 1)

    tri = Delaunay(pts)
    top = trimesh.Trimesh(vertices=pts_3d,
                        faces=tri.simplices,
                        process=True)
    bottom = trimesh.Trimesh(vertices=pts_3d_2,
                        faces=tri.simplices,
                        process=True)
    nx, ny, nz = 32, 32, 16
    x0, x1, y0, y1 = 500, 9500, 500, 9500

    grid = Stratigraphy_Grid(num_grid = [nx, ny, nz], extent = [x0, x1, y0, y1], positive_depth = False)
    grid.load_horizons(top,'top')
    grid.load_horizons(bottom,'bottom')
    grid.cmg_corner_point_generate('top','bottom')
    grid.load_xx_yy_zz('top','bottom')

    return grid