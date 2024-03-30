import numpy as np
import os
import pyvista as pv
class Stratigraphy_Grid:
    
    ## reservoir description
    nx, ny, nz = 32, 32, 16
    x0, x1, y0, y1 = 2000, 8000, 2000, 8000
    __version__ = '0.0.2'
    def __init__(self, num_grid = None, extent = None, positive_depth = False):
        if num_grid is not None:
            [self.nx, self.ny, self.nz] = num_grid
        if num_grid is not None:
            [self.x0, self.x1, self.y0, self.y1] = extent 
        
        self.horizons = {}
        self.formations = {} # for cmg
        self.formation_grids = {} # for pyvista
        self.formation_meta_grid = {} # for meta-data

        self._meshing_extent()
        
        if positive_depth == False:
            self.base = - 99999
            self.normal_vector = np.array([0,0,1], dtype = np.float64).reshape(-1,3)
            
        else:
            self.base = + 99999
            self.normal_vector = np.array([0,0,-1], dtype = np.float64).reshape(-1,3)
            pass

    def __version__(self):
        print(f'The current version is {self.__version__}')

    def __repr__(self):
        return f'{self.__class__.__name__}(nx={self.nx}, ny={self.ny}, nz={self.nz})'
    def __str__(self):
        return f'{self.__class__.__name__}(nx={self.nx}, ny={self.ny}, nz={self.nz})'
    
    def load_xx_yy_zz(self, top_surface_name, bottom_surface_name, require_return = False):
        top =self.horizons[top_surface_name]
        bottom = self.horizons[bottom_surface_name]
        zcorn_upper = [] 
        zcorn_lower = [] 
        for x_, y_ in zip(self.coord_xy[0].flatten(), self.coord_xy[1].flatten()):
            _, _, location = top.ray.intersects_id(np.array([x_,y_,self.base]).reshape(-1,3),
                                                    self.normal_vector,
                                                    return_locations = True,
                                                    multiple_hits = False)
            assert len(location) != 0, 'no intersect found'
            assert len(location) <2, 'more than two intersects'
            zcorn_upper.append(location[0][-1])
            _, _, location = bottom.ray.intersects_id(np.array([x_,y_,self.base]).reshape(-1,3),
                                                    self.normal_vector,
                                                    return_locations = True,
                                                    multiple_hits = False)
            assert len(location) != 0, 'no intersect found'
            assert len(location) <2, 'more than two intersects'   
            zcorn_lower.append(location[0][-1])

        zcorns = []
        for z_0, z_1 in zip(zcorn_upper, zcorn_lower):
            zcorns.append(np.linspace(z_0, z_1, self.nz+1))

        zz = np.array(zcorns).T.reshape(-1, self.ny+1, self.nx +1).T
        xx = self.coord_xy[0].reshape(1, self.ny+1, self.nx+1).T
        yy = self.coord_xy[1].reshape(1, self.ny+1, self.nx+1).T

        xx = np.repeat(xx, zz.shape[-1], axis=-1)
        yy = np.repeat(yy, zz.shape[-1], axis=-1)
        if require_return:
            return xx, yy, zz
        else:
            formation_name = top_surface_name + '_to_' + bottom_surface_name + '_formation'
            self.formation_grids[formation_name] = {'xx': xx, 'yy': yy, 'zz': zz}
    
    def visual_3D_from_formation_grid(self,formation_name, aspect_ratio= 10):
        xx = self.formation_grids[formation_name]['xx']
        yy = self.formation_grids[formation_name]['yy']
        zz = self.formation_grids[formation_name]['zz']
        mesh = pv.StructuredGrid(xx, yy, zz)
        mesh["depth"] = zz.ravel(order="F")
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars=mesh.points[:, -1], show_edges=False,
                        scalar_bar_args={'vertical': True})
        plotter.show_grid()
        plotter.set_scale(xscale=1, yscale=1, zscale=aspect_ratio)
        plotter.show()

    def print_horizons(self):
        print(f'Horizon are ...: ')
        for key, value in self.horizons.items():
            print(f'- {key} is at depth of {value.triangles_center[:,-1].mean():.2f}')

    def print_formations(self):
        print(f'Formations are ...: ')
        for key, value in self.formations.items():
            print(f'- {key}')
            
    def print_formation_grids(self):
        print(f'formation_grids are ...: ')
        for key, value in self.formation_grids.items():
            print(f'- {key}')
    
    def print_formation_meta_grid(self):
        print(f'formation_meta_grid are ...: ')
        for key, value in self.formation_meta_grid.items():
            print(f'- {key}')

    def load_horizons(self, surface, name='random_surface'):
        self.horizons[name] = surface

    
    def meta_corner_point_generate(self, top_surface_name, bottom_surface_name, deposition_pattern = 'proportional'):

        # dimension goes by [eight corners] x [X,Y,Z] x [K] x [J] x [I]
        meta_corners = np.zeros((8, 3, self.nz,self.ny,self.nx)) * np.nan
        formation_name = top_surface_name + '_to_' + bottom_surface_name + '_formation'
        zcorns = self._compute_zcorns(top_surface_name, bottom_surface_name,deposition_pattern = 'proportional')
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # x and y corners
                    for xy in range(2):
                        meta_corners[0, xy, k, j, i] = self.coord_xy[xy][j,i]
                        meta_corners[1, xy, k, j, i] = self.coord_xy[xy][j,i+1]
                        meta_corners[2, xy, k, j, i] = self.coord_xy[xy][j+1,i]
                        meta_corners[3, xy, k, j, i] = self.coord_xy[xy][j+1,i+1]
                        meta_corners[4, xy, k, j, i] = self.coord_xy[xy][j,i]
                        meta_corners[5, xy, k, j, i] = self.coord_xy[xy][j,i+1]
                        meta_corners[6, xy, k, j, i] = self.coord_xy[xy][j+1,i]
                        meta_corners[7, xy, k, j, i] = self.coord_xy[xy][j+1,i+1]
                    # z corner
                    meta_corners[0,2,k,j,i] = zcorns[k,j,i]
                    meta_corners[1,2,k,j,i] = zcorns[k,j,i+1]
                    meta_corners[2,2,k,j,i] = zcorns[k,j+1,i]
                    meta_corners[3,2,k,j,i] = zcorns[k,j+1,i+1]
                    meta_corners[4,2,k,j,i] = zcorns[k+1,j,i]
                    meta_corners[5,2,k,j,i] = zcorns[k+1,j,i+1]
                    meta_corners[6,2,k,j,i] = zcorns[k+1,j+1,i]
                    meta_corners[7,2,k,j,i] = zcorns[k+1,j+1,i+1]
                    
        self.formation_meta_grid[formation_name] = meta_corners


    
    def cmg_corner_point_generate(self, top_surface_name, bottom_surface_name, 
                                  require_return = False, deposition_pattern = 'proportional'):
        top =self.horizons[top_surface_name]
        bottom = self.horizons[bottom_surface_name]

        zcorn_upper = [] 
        zcorn_lower = [] 
        for x_, y_ in zip(self.coord_xy[0].flatten(), self.coord_xy[1].flatten()):
            _, _, location = top.ray.intersects_id(np.array([x_,y_,self.base]).reshape(-1,3),
                                                    self.normal_vector,
                                                    return_locations = True,
                                                    multiple_hits = False)
            assert len(location) != 0, 'no intersect found'
            assert len(location) <2, 'more than two intersects'
            zcorn_upper.append(location[0][-1])
            _, _, location = bottom.ray.intersects_id(np.array([x_,y_,self.base]).reshape(-1,3),
                                                    self.normal_vector,
                                                    return_locations = True,
                                                    multiple_hits = False)
            assert len(location) != 0, 'no intersect found'
            assert len(location) <2, 'more than two intersects'   
            zcorn_lower.append(location[0][-1])

        zcorns = []
        for z_0, z_1 in zip(zcorn_upper, zcorn_lower):
            zcorns.append(np.linspace(z_0, z_1, self.nz+1))

        zcorns = np.array(zcorns).T

        xcorn, ycorn, zcorn = [], [], []
        for z_ in zcorns:
            for i, [x_, y_] in enumerate(zip(self.coord_xy[0].flatten(), self.coord_xy[1].flatten(),)):
                xcorn.append(x_)
                ycorn.append(y_)

        zcorns = zcorns.reshape(self.nz + 1, self.ny + 1, self.nx + 1)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    zcorn.append(zcorns[k, j, i])    
                    zcorn.append(zcorns[k, j, i+1])            
                for i in range(self.nx):
                    zcorn.append(zcorns[k, j+1, i])    
                    zcorn.append(zcorns[k, j+1, i+1])

            for j in range(self.ny):        
                for i in range(self.nx):
                    zcorn.append(zcorns[k+1, j, i])    
                    zcorn.append(zcorns[k+1, j, i+1])
                for i in range(self.nx):
                    zcorn.append(zcorns[k+1, j+1, i])    
                    zcorn.append(zcorns[k+1, j+1, i+1])

        if require_return:
            corns = {}
            corns['xcorn'] = xcorn
            corns['ycorn'] = ycorn
            corns['zcorn'] = zcorn
            return xcorn, ycorn, zcorn
        else:
            formation_name = top_surface_name + '_to_' + bottom_surface_name + '_formation'
            self.load_cmg_corner_points(xcorn, ycorn, zcorn, formation_name = formation_name)
            
    def load_cmg_corner_points(self, xcorn, ycorn, zcorn, formation_name = 'Random_formation'):
        self.formations[formation_name] = {'xcorn': xcorn, 'ycorn': ycorn, 'zcorn': zcorn}
    
    def write_cmg_corner_points_input_file(self, directory, formation_name):
        xcorn = self.formations[formation_name]['xcorn']
        ycorn = self.formations[formation_name]['ycorn']
        zcorn = self.formations[formation_name]['zcorn']
        
        with open(os.path.join(directory,'xcorn.inc'), 'w') as f:
            f.write('*XCORN \n')
            for val in xcorn:
                f.write(f'{val:0.2f} \n')
            f.close()

        with open(os.path.join(directory,'ycorn.inc'), 'w') as f:
            f.write('*YCORN \n')
            for val in ycorn:
                f.write(f'{val:0.2f} \n')
            f.close()

        with open(os.path.join(directory,'zcorn.inc'), 'w') as f:
            f.write('*ZCORN \n')
            for val in zcorn:
                f.write(f'{-val:0.2f} \n')
            f.close()
        pass
    
    def _meshing_extent(self):
        mesh_x_, mesh_y_ = np.linspace(self.x0, self.x1, self.nx+1), np.linspace(self.y0, self.y1, self.ny+1)
        self.coord_xy = np.meshgrid(mesh_x_,mesh_y_)

    def _compute_zcorns(self, top_surface_name, bottom_surface_name,  deposition_pattern = 'proportional'):
        top =self.horizons[top_surface_name]
        bottom = self.horizons[bottom_surface_name]

        zcorn_upper = [] 
        zcorn_lower = [] 
        for x_, y_ in zip(self.coord_xy[0].flatten(), self.coord_xy[1].flatten()):
            _, _, location = top.ray.intersects_id(np.array([x_,y_,self.base]).reshape(-1,3),
                                                    self.normal_vector,
                                                    return_locations = True,
                                                    multiple_hits = False)
            assert len(location) != 0, 'no intersect found'
            assert len(location) <2, 'more than two intersects'
            zcorn_upper.append(location[0][-1])
            _, _, location = bottom.ray.intersects_id(np.array([x_,y_,self.base]).reshape(-1,3),
                                                    self.normal_vector,
                                                    return_locations = True,
                                                    multiple_hits = False)
            assert len(location) != 0, 'no intersect found'
            assert len(location) <2, 'more than two intersects'   
            zcorn_lower.append(location[0][-1])

        zcorns = []
        for z_0, z_1 in zip(zcorn_upper, zcorn_lower):
            zcorns.append(np.linspace(z_0, z_1, self.nz+1))

        zcorns = np.array(zcorns).reshape(self.nz + 1, self.ny + 1, self.nx +1)
        return zcorns


if __name__ == '__main__':
    
    import numpy as np
    from scipy.spatial import Delaunay
    import trimesh
    from reservoir_grid import Stratigraphy_Grid
    ## basic settings

    epsilon = 1E-5
    x_0, x_1 = 0.0, 10_000
    y_0, y_1 = 0.0, 10_000
    bin_size = (x_1-x_0)/50
    ## generate surfaces
    x_bins = np.arange(x_0,x_1,bin_size)
    y_bins = np.arange(y_0,y_1,bin_size)
    mesh_x, mesh_y = np.meshgrid(x_bins, y_bins)


    # ## Version #1 - anticline
    # thickness = 50
    # radius = 1000
    # power = 1
    # alpha = (thickness/radius**power)
    # mesh_z = 50 - np.abs(alpha*(mesh_x**power + mesh_y**power))
    # plt.plot(mesh_x[40],mesh_z[40])

    ## Version #2 - anticline
    depth = 3020
    thickness = 150
    diff = 100
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
    x0, x1, y0, y1 = 2000, 8000, 2000, 8000

    grid = Stratigraphy_Grid(num_grid = [nx, ny, nz], extent = [x0, x1, y0, y1], positive_depth = False)
    grid.load_horizons(top,'top')
    grid.load_horizons(bottom,'bottom')
    grid.cmg_corner_point_generate('top','bottom')
    grid.load_xx_yy_zz('top','bottom')
    grid.visual_3D_from_formation_grid('top_to_bottom_formation')