import matplotlib.tri as tri
import matplotlib.pyplot as plt
import FEMOL


# Load mesh number 1
mesh_1 = FEMOL.mesh.load_vtk('Results/Soundboard_Article/Plate_case_study/TOM_results/TOM_modal_90_90_mode0.vtk')
mesh_1.cell_to_point_data('zc')

# Load the mesh 2
mesh_2 = FEMOL.mesh.load_vtk('TOM_modal_0_6_90_2_iter4.vtk')
mesh_2.cell_to_point_data('zc')

pts_old = mesh_1.point_data['zc']
pts_new = mesh_2.point_data['zc']
pts_old = pts_old.reshape((int(np.sqrt(mesh_1.N_ele)) +1, int(np.sqrt(mesh_1.N_ele)) + 1))
pts_new = pts_new.reshape((int(np.sqrt(mesh_2.N_ele)) +1, int(np.sqrt(mesh_2.N_ele)) + 1))*10

mesh_2.plot._empty_mesh_2D()
triangulation = tri.Triangulation(*mesh_2.points.T[:2], mesh_2.plot.all_tris)
ax = plt.gca()
ax.tricontourf(triangulation, (pts_new - pts_old).flatten(), cmap='jet')
ax.set_title('Difference in zc for mesh 1 and 2')
