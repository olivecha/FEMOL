import FEMOL
import matplotlib.pyplot as plt

# Create the meshes
mesh_Q4 = FEMOL.mesh.rectangle_Q4(1, 1, 10, 10)
mesh_T6 = FEMOL.mesh.rectangle_T6(1, 1, 10, 10)
meshes = [mesh_Q4, mesh_T6]
wis, vis = [], []

# Solve the clamped plate problem for each mesh
for mesh in meshes:
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
    problem.define_materials(FEMOL.materials.isotropic_bending_benchmark())
    problem.define_tensors(0.01)
    problem.add_fixed_domain(FEMOL.domains.outside_box(0.01, 0.99, 0.01, 0.99))
    problem.add_forces(force=[0, 0, -10, 0, 0, 0], domain=FEMOL.domains.inside_box([[0.05, 0.99]], [[0.05, 0.99]]))
    w, v = problem.solve(verbose=False, filtre=1)
    wis.append(w)
    vis.append(v)

# Plot the mode frequencies
for w, ele in zip(wis, ['Q4', 'T6']):
    plt.plot(w[:50], label=ele)
plt.title('Modal frequencies for T6 and Q4 elements')
plt.xlabel('Mode number')
plt.ylabel('Frequency')
plt.legend()
#save
#plt.gcf().savefig('conv_plate_modal', dpi=250)

# Plot mode shapes
fig, axs = plt.subplots(2, 2, figsize=(8,8))
ax1, ax2, ax3, ax4 = axs.flatten()

plt.sca(ax1)
meshes[0].add_mode('m', vis[0][0], 6)
meshes[0].plot.point_data('m_Uz')
ax1.set_title('Q4 mode (1,1)')

plt.sca(ax2)
meshes[1].add_mode('m', -vis[1][0], 6)
meshes[1].plot.point_data('m_Uz')
ax2.set_title('T6 mode (1,1)')

plt.sca(ax3)
meshes[0].add_mode('m', vis[0][3], 6)
meshes[0].plot.point_data('m_Uz')
ax3.set_title('Q4 mode (2,2)')

plt.sca(ax4)
meshes[1].add_mode('m', vis[1][3], 6)
meshes[1].plot.point_data('m_Uz')
ax4.set_title('T6 mode (2,2)')

plt.gcf().savefig('plate_modes', dpi=250)