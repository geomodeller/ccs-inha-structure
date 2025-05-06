import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# --- Original Helper Functions (kept for reference, but not used in the fast loop) ---
def create_box_from_vertices_original(vertices):
    """
    Creates an Open3D TriangleMesh representing a 3D box from its vertices.
    (Original version for comparison)

    This function creates a 3D box mesh from 8 input vertices using 
    Open3D library. It first checks if the input is valid (a list or 
    numpy array of 8 vertices with numerical values), then defines 
    the triangles that make up the box's faces, and finally returns 
    the resulting TriangleMesh object.
    """
    if not isinstance(vertices, (list, np.ndarray)) or len(vertices) != 8:
        print("Error: Input must be a list or numpy array of 8 vertices.")
        return None
    try:
        vertices_np = np.asarray(vertices, dtype=np.float64)
    except ValueError:
        print("Error: Vertices must contain numerical values.")
        return None

    triangles = [
        [0, 1, 2], [0, 2, 3],  # Bottom face (indices relative to box's 8 vertices)
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 4, 7], [0, 7, 3],  # Front face
        [1, 5, 6], [1, 6, 2],  # Back face
        [3, 2, 6], [3, 6, 7],  # Right face
        [0, 1, 5], [0, 5, 4]   # Left face
    ]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_np),
        o3d.utility.Vector3iVector(triangles)
    )
    return mesh

def color_mesh_original(mesh, color):
    """
    Colors all faces of an Open3D TriangleMesh with a given color.
    (Original version for comparison)

    This function, color_mesh_original, takes an Open3D 
    TriangleMesh and a color as input, and sets the color 
    of all faces of the mesh to the specified color. 
    The color should be a list or numpy array of 3 floats 
    representing the RGB values. If the color is invalid, 
    it prints an error message and returns without modifying the mesh.
    """
    if not isinstance(color, (list, np.ndarray)) or len(color) != 3:
        print("Error: Color must be a list or numpy array of 3 floats (RGB).")
        return
    try:
        color_np = np.asarray(color, dtype=np.float64)
    except ValueError:
        print("Error: Color values must be numerical.")
        return
    mesh.paint_uniform_color(color_np)

def visual_3d_with_structure(property, 
                             xcorn, 
                             ycorn, 
                             zcorn, 
                             cm_type = plt.cm.viridis, 
                             improved_ver = True,
                             add_surrounding_box = True,
                             vis_opt_json = None,
                             vis_camera_angle_json = None):
    """
    Visualizes a 3D grid with color representing property values at each cell,
    with structure defined by xcorn, ycorn, zcorn. The grid is rendered as a
    collection of boxes, with each box's color determined by the corresponding
    property value. The grid is oriented so that the x-axis points to the right,
    the y-axis points up, and the z-axis points into the screen.

    Args:
        property (ndarray): A 3D array of shape (nz, ny, nx) containing the
            property values to be visualized.
        xcorn (ndarray): A 3D array of shape (nz+1, ny+1, nx+1) containing the
            x-coordinates of the cell corners.
        ycorn (ndarray): A 3D array of shape (nz+1, ny+1, nx+1) containing the
            y-coordinates of the cell corners.
        zcorn (ndarray): A 3D array of shape (nz+1, ny+1, nx+1) containing the
            z-coordinates of the cell corners.
        cm_type (matplotlib colormap): The colormap to use for the visualization.
            Defaults to viridis.
        improved_ver (bool): Flag to use the improved visualization method.
        add_surrounding_box (bool): Flag to add an axis-aligned bounding box.
        vis_opt_json (str, optional): Path to a JSON file with visualization options.
        vis_camera_angle_json (str, optional): Path to a JSON file with camera parameters.


    Returns:
        None
    """
    all_vertices = []
    all_triangles = []
    all_vertex_colors = []
    vertex_offset = 0  

    # compute color values
    normalized_facies = (property - np.min(property)) / (np.max(property) - np.min(property))
    colors = cm_type(normalized_facies)[:, :, :, :3].astype(np.float64)

    # get the dimension of properties
    nz, ny, nx = property.shape

    # Check consistency between properties and structure grids
    assert xcorn.shape == ycorn.shape, 'inconsistent among structure'
    assert xcorn.shape == zcorn.shape, 'inconsistent among structure'
    assert zcorn.shape == ycorn.shape, 'inconsistent among structure'
    assert nz == xcorn.shape[0] - 1, 'inconsistent btw property and structure [z-grid]'
    assert ny == xcorn.shape[1] - 1, 'inconsistent btw property and structure [x-grid]'
    assert nx == xcorn.shape[2] - 1, 'inconsistent btw property and structure [y-grid]'

    # Define the triangle indices *once* relative to a box's 8 vertices (0-7)
    box_triangles_template = np.array([
            # Bottom face (z=k+1, normal points -z): Vertices 0, 1, 2, 3
            # View from -z: CCW order is 0 -> 3 -> 2 -> 1
            [0, 3, 2], [0, 2, 1],

            # Top face (z=k, normal points +z): Vertices 4, 5, 6, 7
            # View from +z: CCW order is 4 -> 5 -> 6 -> 7
            [4, 5, 6], [4, 6, 7],

            # Front face (y=0, normal points -y): Vertices 0, 1, 5, 4
            # View from -y: CCW order is 0 -> 4 -> 5 -> 1
            [0, 4, 5], [0, 5, 1],

            # Back face (y=1, normal points +y): Vertices 3, 2, 6, 7
            # View from +y: CCW order is 3 -> 2 -> 6 -> 7
            [3, 2, 6], [3, 6, 7],

            # Left face (x=0, normal points -x): Vertices 0, 3, 7, 4
            # View from -x: CCW order is 0 -> 3 -> 7 -> 4
            [0, 3, 7], [0, 7, 4],

            # Right face (x=1, normal points +x): Vertices 1, 2, 6, 5
            # View from +x: CCW order is 1 -> 2 -> 6 -> 5
            [1, 2, 6], [1, 6, 5],

            ], dtype=np.int32)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Select only boundary boxes
                if (k in [0, nz-1]) or (i in [0, nx-1]) or (j in [0, ny-1]):
                    # Define vertices for the current box (k, j, i) more directly
                    # Order: bottom face counter-clockwise, then corresponding top face vertices
                    #  3----2
                    # /|   /|      z
                    # 7----6 |      | y
                    # | 0--|-1      |/
                    # |/   |/       +--x
                    # 4----5
                    # Vertex order assumed by create_box_from_vertices_original was potentially different.
                    # Let's redefine explicitly based on common corner numbering:
                    # Order: (0,0,0), (1,0,0), (1,1,0), (0,1,0) -> bottom face (z=k)
                    #        (0,0,1), (1,0,1), (1,1,1), (0,1,1) -> top face    (z=k+1)
                    # Ensure this matches your `xcorn`, `ycorn`, `zcorn` indexing logic.
                    # The original vertex list seemed to have bottom face at z=k+1 and top at z=k.
                    # Sticking to the *original* vertex definition from the prompt for consistency:
                    vertices = np.array([
                        [xcorn[k+1, j,   i],   ycorn[k+1, j,   i],   zcorn[k+1, j,   i]],   # 0 (Original Bottom)
                        [xcorn[k+1, j,   i+1], ycorn[k+1, j,   i+1], zcorn[k+1, j,   i+1]], # 1
                        [xcorn[k+1, j+1, i+1], ycorn[k+1, j+1, i+1], zcorn[k+1, j+1, i+1]], # 2
                        [xcorn[k+1, j+1, i],   ycorn[k+1, j+1, i],   zcorn[k+1, j+1, i]],   # 3
                        [xcorn[k,   j,   i],   ycorn[k,   j,   i],   zcorn[k,   j,   i]],   # 4 (Original Top)
                        [xcorn[k,   j,   i+1], ycorn[k,   j,   i+1], zcorn[k,   j,   i+1]], # 5
                        [xcorn[k,   j+1, i+1], ycorn[k,   j+1, i+1], zcorn[k,   j+1, i+1]], # 6
                        [xcorn[k,   j+1, i],   ycorn[k,   j+1, i],   zcorn[k,   j+1, i]],   # 7
                    ], dtype=np.float64)

                    # Add these 8 vertices to the main list
                    all_vertices.append(vertices)

                    # Add the 12 triangles, offsetting indices by the current vertex count
                    all_triangles.append(box_triangles_template + vertex_offset)

                    # Get the color for this box
                    color = colors[k, j, i]
                    # Add the color for each of the 8 vertices
                    all_vertex_colors.append(np.tile(color, (8, 1)))

                    # Increment the vertex offset for the next box
                    vertex_offset += 8 # We added 8 vertices

    # Concatenate lists into large NumPy arrays
    if all_vertices: # Check if any boxes were selected
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_triangles = np.concatenate(all_triangles, axis=0)
        final_vertex_colors = np.concatenate(all_vertex_colors, axis=0)

        # Create the single, combined mesh
        combined_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(final_vertices),
            o3d.utility.Vector3iVector(final_triangles)
        )

        # Assign vertex colors
        combined_mesh.vertex_colors = o3d.utility.Vector3dVector(final_vertex_colors)

        # Optional: Compute normals for better shading
        combined_mesh.compute_vertex_normals()

        # Visualize the combined mesh
        if improved_ver:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(combined_mesh)
            if add_surrounding_box:
                # Create a bounding box from the combined_mesh
                bbox = combined_mesh.get_axis_aligned_bounding_box()
                bbox.color = (0, 0, 0) # <--- ADD THIS LINE, make bbox black!

                # Add the bounding box geometry to the visualizer
                vis.add_geometry(bbox)
            opt = vis.get_render_option()
            opt.mesh_show_back_face = True # <--- Add this line
            if vis_opt_json is not None:
                opt.load_from_json(vis_opt_json)
            view_control = vis.get_view_control()
            if vis_camera_angle_json is not None:
                param = o3d.io.read_pinhole_camera_parameters(vis_camera_angle_json)
                view_control.convert_from_pinhole_camera_parameters(param)

            vis.run()
            vis.destroy_window()
        else:
            ## NO LONGER USED... DEPRECIATED
            print('## NO LONGER USED... DEPRECIATED ##')
            o3d.visualization.draw_geometries([combined_mesh])
    else:
        assert False, 'no boundary voxels are found'