import FEMOL.misc

# Create a round L bracket mesh
mesh = FEMOL.misc.L_bracket_mesh2(0.005)

# Create a FEM Problem from the mesh (compute displacement with a plane stress model)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')

# Define the problem material and part thickness
PLA = FEMOL.materials.IsotropicMaterial(100e2, 0.3, 1)
problem.define_materials(PLA, PLA)
problem.define_tensors(1, 2)

# Add a force on the top circle nodes
top_circle = FEMOL.domains.inside_circle(*[0.135, 0.160], 0.0105)
problem.add_forces([-1000, -1000], top_circle)

# Fix the two bottom circles
bot_circle_1 = FEMOL.domains.inside_circle(*[0.012, 0.016], 0.0034/2 + 0.001)
bot_circle_2 = FEMOL.domains.inside_circle(*[0.022, 0.016], 0.0034/2 + 0.001)
problem.add_fixed_domain(bot_circle_1)
problem.add_fixed_domain(bot_circle_2)


# Define and solve the topology problem
topo_problem = FEMOL.SIMP_COMP(problem, volfrac=0.45, rmin=3.5)
mesh = topo_problem.solve(converge=0.02, max_iter=100, plot=True, save=False)
mesh.stress_from_displacement(PLA.plane_tensor(1), N_dof=2)

mesh.plot.cell_data('X')
mesh.plot.cell_data('Sv', cmap='plasma')
