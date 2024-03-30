import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

a = np.random.normal(size = (32,32,16))
nz, ny, nx = a.shape
X_index = np.linspace(0,nx-1,nx)
Y_index = np.linspace(0,ny-1,ny)
Z_index = np.linspace(0,nz-1,nz)
yy, zz, xx = np.meshgrid(Y_index,Z_index, X_index)
points = np.concatenate([zz.reshape(-1,1), yy.reshape(-1,1), xx.reshape(-1,1)], axis = 1)
values = a.reshape(-1,1)

X_index_new = np.linspace(-0.5,nx-0.5,nx+1)
Y_index_new = np.linspace(-0.5,ny-0.5,ny+1)
Z_index_new = np.linspace(-0.5,nz-0.5,nz+1)
yy_new, zz_new, xx_new = np.meshgrid(Y_index_new,Z_index_new, X_index_new)

grid_value = griddata(points, values, (zz_new,yy_new, xx_new), method='linear').squeeze()
flag = np.isnan(grid_value)
grid_value[flag] = griddata(points, values, (zz_new,yy_new, xx_new), method='nearest').squeeze()[flag]
plt.imshow(a[0], vmin = -3, vmax = 3);
plt.scatter(xx_new[0].flatten(), yy_new[0].flatten(), c= grid_value[0].flatten(), vmin = -3, vmax = 3, edgecolors = 'k');
plt.show()