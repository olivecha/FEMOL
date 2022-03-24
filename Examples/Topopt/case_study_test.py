import FEMOL
import matplotlib

# Solve the eigenvalue problem for a 1x1 flax + carbon plate
# Square mesh
mesh = FEMOL.mesh.rectangle_Q4(1, 1, 20, 20)
# laminates and materials
plies1 = [0, 0, 0, 0]
plies2 = [90, 90]
flax = FEMOL.materials.general_flax()
carbon = FEMOL.materials.general_carbon()
layup1 = FEMOL.laminate.Layup(material=flax, plies=plies1, symetric=True)
layup2 = FEMOL.laminate.Layup(material=carbon, plies=plies2, symetric=False, h_core=0, z_core = 0.05 + carbon.hi)
# FEM problem definition
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
problem.define_materials(flax, carbon)
problem.define_tensors(layup1, layup2)
# First modal solve
w, v = problem.solve(filtre=0)

#Find the first bending mode
mesh.add_mode('m', v[2], 6)
mesh.wrap('m')
#mesh.plot.point_data('m_Uz')
ref_vector = v[2]

# Solve the eigenvalue maximization of the first mode
SIMP = FEMOL.SIMP_VIBE(Problem=problem, objective='max eig')
mesh = SIMP.solve(ref_vector, save=False, plot=False, converge=0.05)
#mesh.save('TOM_result_1')

# Plot the result and core height
fig, axs = plt.subplots(1, 2)
plt.sca(axs[0])
mesh.plot.cell_data('zc', cmap='plasma')
axs[0].set_title('core height')
plt.sca(axs[1])
mesh.plot.cell_data('X_real', cmap='plasma')
axs[1].set_title('coating density')

# Plot the plate cross sections
Vx = (mesh.cell_data['X']['quad'].reshape(20, 20))**(1/3)
Vh = mesh.cell_data['zc']['quad'].reshape(20, 20)
Vx *= Vh.max()
fig, ax = plt.subplots()
ax.plot(Vx[range(20), range(20)], label='density [0, 1]', color='b')
ax.plot(Vh[range(20), range(20)], label='height (normalized)', color='orange')
ax.set_yscale('log')
ax.set_xlabel('plate diagonal cross section')
plt.legend()

# Plot the density to core height transformation
# Create core height, bending stiffness vectors
corez = np.linspace(0, 1)
pliest = [90, 90]
D11_list = []
for z in corez:
    layupt = FEMOL.laminate.Layup(material=carbon, plies=pliest, symetric=False, 
                                  h_core=0, z_core= z + carbon.hi)
    D11_list.append(layupt.D_mat[0,0])

X_list = (np.array(D11_list)/D11_list[-1])**(1/3)

# Plot the density vs core height relation
fig, ax = plt.subplots()
ax.plot(X_list, corez, color='k')
ax.plot([0, 1], [0, 1], '--', color='0.8')
ax.set_xlabel(r'Density $\rho_e$')
ax.set_ylabel(r'Coating height $h / h_{max}$')
ax.grid('on')

# Plot the eigenvalue distribution for each iteration
cmap = matplotlib.cm.get_cmap('viridis')

for i, wi in enumerate(SIMP.all_lmbds):
    color = cmap(i/len(SIMP.all_lmbds))
    plt.plot(wi[wi>1][:10], color=color)

norm = matplotlib.colors.Normalize(0, len(SIMP.all_lmbds))
plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax = plt.gca(), label='iteration')
plt.grid('on')
plt.xlabel('mode number')
plt.ylabel('Frequency (Hz)')