import numpy as np
import os

def _load_corner_inc_files(cmg_folder = './', corner_files = ['xcorn.inc','ycorn.inc','zcorn.inc'], skiprows = 1):
    corner_files = sorted(corner_files)
    loaded_object = []
    for file in corner_files:
        loaded_object.append(np.loadtxt(os.path.join(cmg_folder, file), skiprows = skiprows))
    return loaded_object

def _reshape_corner(corner, nx, ny, nz):
    return np.reshape(corner, (nz+1, ny+1, nx+1))
def _reshape_corner_z(corner, nx, ny, nz):
    return np.reshape(corner, (nz,2,ny,2,nx,2))

def _arrange_z_corn(zcorn, nx, ny, nz):
    zcorn_ = np.zeros((nz+1, ny+1, nx+1))
    for k in range(0, nz):
        for j in range(0,ny):
            for i in range(0,nx):
                zcorn_[k,j,i]       = zcorn[k,0,j,0,i,0]
                zcorn_[k,j,i+1]     = zcorn[k,0,j,0,i,1]
                zcorn_[k,j+1,i]     = zcorn[k,0,j,1,i,0]
                zcorn_[k,j+1,i+1]   = zcorn[k,0,j,1,i,1]
                zcorn_[k+1,j,i]     = zcorn[k,1,j,0,i,0]
                zcorn_[k+1,j,i+1]   = zcorn[k,1,j,0,i,1]
                zcorn_[k+1,j+1,i]   = zcorn[k,1,j,1,i,0]
                zcorn_[k+1,j+1,i+1] = zcorn[k,1,j,1,i,1]
    return zcorn_

def load_cmg_structure_to_python(cmg_folder = 'example_cmg_structure', 
                                 corner_files = ['xcorn.inc','ycorn.inc','zcorn.inc'], 
                                 nx = 32, 
                                 ny = 32, 
                                 nz = 16, 
                                 skiprows = 1,
                                 vertical_exaggeration = False,
                                 positive_z = True):
    """
    Load CMG structure from a folder into python
    Parameters
    ----------
    cmg_folder : str
        folder name of CMG structure
    corner_files : list
        list of strings of corner file names
    nx, ny, nz : int
        number of grid points in x, y, z direction
    skiprows : int
        number of rows to skip when reading corner files
    Returns
    -------
    xcorn, ycorn, zcorn : 3D numpy array
        x, y, and z coordinates of all corner points in CMG structure
    """
    
    xcorn, ycorn, zcorn = _load_corner_inc_files(cmg_folder, corner_files = corner_files, skiprows = skiprows)
    xcorn = _reshape_corner(xcorn, nx, ny, nz)
    ycorn = _reshape_corner(ycorn, nx, ny, nz)
    zcorn = _reshape_corner_z(zcorn, nx, ny, nz)
    zcorn = _arrange_z_corn(zcorn, nx, ny, nz)
    if vertical_exaggeration:
        zcorn *= vertical_exaggeration
    if positive_z:
        zcorn *= -1
    return xcorn, ycorn, zcorn
