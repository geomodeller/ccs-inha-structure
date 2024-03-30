import numpy as np
from scipy.spatial import Delaunay
## referenced to: https://stackoverflow.com/questions/70366109/how-to-calculate-volume-of-irregular-shapes-in-python
def compute_volume_of_formation(meta_grid):
    """
    Compute the volume of the formation based on the given meta_grid.

    Parameters:
    - meta_grid: numpy array representing the meta grid

    Returns:
    - volume: numpy array containing the computed volume for each grid point
    """
    
    _, _, nz, ny, nx = meta_grid.shape
    volume = np.zeros((nz,ny,nx))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                corner = meta_grid[:, :, k, j, i]
                corner_new = corner
                corner_new [2,:] = corner [3,:]
                corner_new [3,:] = corner [2,:]
                corner_new [6,:] = corner [7,:]
                corner_new [7,:] = corner [6,:]
                
                tri = Delaunay(corner_new)
                tetrahedra = corner_new[tri.simplices]
                volume[k,j,i] = np.array([volume_tetrahedron(t) for t in tetrahedra]).sum()

    return volume

def volume_tetrahedron(tetrahedron):
    """
    Calculate the volume of a tetrahedron using the provided vertices.

    Parameters:
    tetrahedron (list): List of 4 vertices of the tetrahedron.

    Returns:
    float: Volume of the tetrahedron.
    """
    matrix = np.array([
        tetrahedron[0] - tetrahedron[3],
        tetrahedron[1] - tetrahedron[3],
        tetrahedron[2] - tetrahedron[3]
    ])
    return abs(np.linalg.det(matrix))/6