import FEMOL
import matplotlib.pyplot as plt

# Create the meshes
mesh_Q4 = FEMOL.mesh.circle_Q4(1, 20)
mesh_T6 = FEMOL.mesh.circle_T6(1, 10)
mesh_T3 = FEMOL.mesh.circle_T3(1, 10)
meshes = [mesh_Q4, mesh_T6, mesh_T3]

# List for eigenvals and vectors
wi = []
vi = []

# Solve for each mesh
for mesh in meshes:
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
    problem.define_materials(FEMOL.materials.isotropic_bending_benchmark())
    problem.define_tensors(0.01)
    problem.add_fixed_domain(FEMOL.domains.outside_circle(0, 0, 0.99))
    w, v = problem.solve(verbose=False, filtre=0)
    wi.append(w)
    vi.append(v)

# Filter the node modes
good_wi = []
good_vi = []
for w, v in zip(wi, vi):
    good_wi.append(w[w > 1])
    good_vi.append(v[w > 1])

# Plot the eigenvalues
for w, ele in zip(good_wi, ['Q4', 'T6', 'T3']):
    plt.plot(w[:500], label=ele)
plt.legend()
ax = plt.gca()
ax.set_xlabel('Mode number')
ax.set_ylabel('Frequency')
ax.set_title('In plane eigenvalues for T3, T6 and Q4 elements')

# Plot the first mode
fig, axs = plt.subplots(1, 3, figsize=(8, 5))
for v, mesh, ele, ax in zip(good_vi, meshes, ['Q4', 'T6', 'T3'], axs):
    plt.sca(ax)
    mesh.add_mode('m', v[0], 2)
    mesh.wrap('m', 0.05)
    mesh.plot.wrapped_2D(color='w')
    ax.set_title(ele)
