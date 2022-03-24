# Aspect ratio like à 180cm x 100mm ski
mesh = FEMOL.mesh.rectangle_Q4(18, 1, 90, 6)
# laminates and materials
# Flax base (softer)
plies1 = [0, 0, 0, 0]
flax = FEMOL.materials.general_flax()
layup1 = FEMOL.laminate.Layup(material=flax, plies=plies1, symetric=True)
# Carbon reinforcement
plies2 = [90, 90]
carbon = FEMOL.materials.general_carbon()
layup2 = FEMOL.laminate.Layup(material=carbon, plies=plies2, symetric=False, h_core=0, 
                              z_core= 0.1 + carbon.hi)
# FEM problem definition
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
problem.define_materials(flax, carbon)
problem.define_tensors(layup1, layup2) 
# First modal solve
w, v = problem.solve(filtre=0)
# Find the first good mode
mesh.add_mode('m', v[1], 6)
mesh.wrap('m')
mesh.plot.point_data('m_Uz')

# Solve for the 9 first modes
for i, vi in enumerate(v[1:10]): 
    # solve the SIMP problem
    SIMP = FEMOL.SIMP_VIBE(Problem=problem, objective='max eig')
    mesh = SIMP.solve(vi, save=False, plot=True, converge=0.03)
    mesh.save('ski_modal/tom_ski' + str(i))
    
# Plot the results
fig = plt.figure(figsize=(10, 6))
n = 9

def plot_ski():
    ax = plt.gca()
    ax.plot([0, 18, 18, 0, 0], [0, 0, 1, 1, 0], color='r')

for i in range(n):
    ax = fig.add_subplot(n, 1, i+1)
    mesh = FEMOL.mesh.load_vtk('tom_ski' + str(i) + '.vtk')
    mesh.plot.cell_data('X')
    ax.set_title('mode ' + str(i))
    plot_ski()
    
plt.tight_layout()