import open3d as o3d
import trimesh

def visualize_surface(surface: trimesh.Trimesh | list[trimesh.Trimesh], save_html: str = None) -> None:
    assert isinstance(surface, (trimesh.Trimesh, list)), 'surface must be a trimesh or a list of trimesh'
    
    if isinstance(surface, trimesh.Trimesh):
        surface = [surface]

    surface_o3d = [_plot_surface(s) for s in surface]
    
    if save_html:
        o3d.visualization.draw(surface_o3d)
        vis = o3d.visualization.WebVisualizer()
        vis.set(meshes=surface_o3d)
        vis.export(save_html)
        print(f"Visualization saved as {save_html}")
    else:
        o3d.visualization.draw_geometries(surface_o3d)

def _plot_surface(surface: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert a trimesh surface to an o3d mesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(surface.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(surface.faces)
    return mesh