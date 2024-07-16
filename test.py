import numpy as np
import os
import pyvista as pv
from scipy.interpolate import griddata
import vtk

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
    

    def _add_value_to_grid(self, value, formation_name,value_name = 'Value'):
        assert formation_name in self.formation_grids.keys(), 'No such formation grid is found. Please make sure the formation name is correct.'

        X_index = np.linspace(0,self.nx-1,self.nx)
        Y_index = np.linspace(0,self.ny-1,self.ny)
        Z_index = np.linspace(0,self.nz-1,self.nz)
        yy, zz, xx = np.meshgrid(Y_index,Z_index, X_index)
        points = np.concatenate([zz.reshape(-1,1), yy.reshape(-1,1), xx.reshape(-1,1)], axis = 1)
        values = value.reshape(-1,1)

        X_index_new = np.linspace(-0.5,self.nx-0.5,self.nx+1)
        Y_index_new = np.linspace(-0.5,self.ny-0.5,self.ny+1)
        Z_index_new = np.linspace(-0.5,self.nz-0.5,self.nz+1)
        yy_new, zz_new, xx_new = np.meshgrid(Y_index_new,Z_index_new, X_index_new)

        grid_value = griddata(points, values, (zz_new,yy_new, xx_new), method='linear').squeeze()
        flag = np.isnan(grid_value)
        grid_value[flag] = griddata(points, values, (zz_new,yy_new, xx_new), method='nearest').squeeze()[flag]
        
        self.formation_grids[formation_name][value_name] = grid_value
        print(f'{value_name} succesfully added to {formation_name}')
        print(f'formation_grids[formation_name].keys are {list(self.formation_grids[formation_name].keys())}')

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
    
    def visual_3D_from_formation_grid(self,formation_name, value_name = None, aspect_ratio= 10, show_edges=True, vertical_colorbar = True, add_observer = False):
        xx = self.formation_grids[formation_name]['xx']
        yy = self.formation_grids[formation_name]['yy']
        zz = self.formation_grids[formation_name]['zz']
        mesh = pv.StructuredGrid(xx, yy, zz)       
        plotter = pv.Plotter()
        mesh.points[:,-1] *=aspect_ratio
        if value_name is None: 
            mesh["depth"] = zz.ravel(order="F")
            actor = plotter.add_mesh(mesh, 
                            scalars=mesh.points[:, -1], 
                            show_edges=show_edges,
                            scalar_bar_args={'vertical': vertical_colorbar,
                                             'title': "Depth"},
                            )
        else:
            value = self.formation_grids[formation_name][value_name]
            mesh[value_name] = value.ravel(order="C")
            actor = plotter.add_mesh(mesh, 
                            scalars=mesh[value_name], 
                            show_edges=show_edges,
                            scalar_bar_args={'vertical': vertical_colorbar,
                                             'title': value_name},
                            )
        
        well_loc_1 = [[3781.25,3781.25,-30000],[3781.25,3781.25,-32000]]
        well_1 = pv.Line(well_loc_1[0], well_loc_1[1])
        well_loc_2 = [[3781.25,6218.75,-30000],[3781.25,6218.75,-32000]]
        well_2 = pv.Line(well_loc_2[0], well_loc_2[1])
        well_loc_3 = [[6218.75,6218.75,-30000],[6218.75,6218.75,-32000]]
        well_3 = pv.Line(well_loc_3[0], well_loc_3[1])
        well_loc_4 = [[6218.75,3781.25,-30000],[6218.75,3781.25,-32000]]
        well_4 = pv.Line(well_loc_4[0], well_loc_4[1])

        actor = plotter.add_point_labels([well_loc_1[0], well_loc_1[1]], ["Injector 1", ""], font_size=15, point_color="red", text_color="red")
        actor = plotter.add_point_labels([well_loc_2[0], well_loc_2[1]], ["Injector 2", ""], font_size=15, point_color="red", text_color="red")
        actor = plotter.add_point_labels([well_loc_3[0], well_loc_3[1]], ["Injector 3", ""], font_size=15, point_color="red", text_color="red")
        actor = plotter.add_point_labels([well_loc_4[0], well_loc_4[1]], ["Injector 4", ""], font_size=15, point_color="red", text_color="red")
        actor = plotter.add_mesh(well_1, color="b", line_width=6)
        actor = plotter.add_mesh(well_2, color="b", line_width=6)
        actor = plotter.add_mesh(well_3, color="b", line_width=6)
        actor = plotter.add_mesh(well_4, color="b", line_width=6)
        
        if add_observer:
            def my_cpos_callback(*args):
                """
                Adds the current camera position to the plotter as text.

                Parameters:
                    *args: Variable length argument list.

                Returns:
                    None
                """
                plotter.add_text(str(plotter.camera_position), name="cpos")
                return
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
        else:
            pos_cam = [(-6830, -6208, -25970), (4859, 4970, -31683), (0.2470, 0.2232, 0.9429)]
            def my_cpos_callback(*args):
                """
                Adds the current camera position to the plotter as text.

                Parameters:
                    *args: Variable length argument list.

                Returns:
                    None
                """
                plotter.add_text(str(plotter.camera_position), name="cpos")
                return
            plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
            actor = plotter.camera_position = pos_cam

        actor = plotter.show_grid()
        actor = plotter.show_bounds(
                    grid='back',
                    location='origin',
                    ticks='both',
                    n_xlabels = 4,
                    n_ylabels = 4,
                    n_zlabels = 1,
                    show_xlabels=True,
                    show_ylabels=True,
                    show_zlabels=True,
                    xtitle='Easting',
                    ytitle='Northing',
                    ztitle='Depth',
                    font_size=13,
                    bold=False
                    )
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
    # from reservoir_grid import Stratigraphy_Grid
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
    grid.visual_3D_from_formation_grid('top_to_bottom_formation',vertical_colorbar =False)