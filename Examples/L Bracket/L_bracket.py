import FEMOL
import FEMOL.misc
from plot import contour_stress_plot

# Create a round L bracket mesh
mesh = FEMOL.misc.L_bracket_mesh2(0.005)

# Create a FEM Problem from the mesh (compute displacement with a plane stress model)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')

# Define the problem material and part thickness
PLA = FEMOL.materials.IsotropicMaterial(4.8e9, 0.3, 1)
problem.define_materials(PLA, PLA)
problem.define_tensors(0.005, 0.005)

# Add a force on the top circle nodes
top_circle = FEMOL.domains.inside_circle(*[0.135, 0.160], 0.0105)
problem.add_forces([-1e3, -1e3], top_circle)

# Fix the two bottom circles
bot_circle_1 = FEMOL.domains.inside_circle(*[0.012, 0.016], 0.0034/2 + 0.001)
bot_circle_2 = FEMOL.domains.inside_circle(*[0.022, 0.016], 0.0034/2 + 0.001)
problem.add_fixed_domain(bot_circle_1)
problem.add_fixed_domain(bot_circle_2)

# Define and solve the topology problem
topo_problem = FEMOL.SIMP_COMP(problem, volfrac=0.45, rmin=3.5)
mesh = topo_problem.solve(converge=0.02, max_iter=100, plot=False, save=False)

# point data plot of the density
mesh.cell_to_point_data('X')
mesh.plot.point_data('X', cmap='Greys', wrapped=False)

# Save the mesh to STL with millimeter units
mesh.point_data_to_STL('L_mesh', which='X', hb = 0.0025, hc = 0.0025, symmetric=True, scale=1000)

# Compute and plot the stresses
mesh.stress_from_displacement(PLA.plane_tensor(0.0025), PLA.plane_tensor(0.0025))
mesh.cell_to_point_data('Sv')
mesh.plot.point_data('Sv', cmap='plasma')

# contour stress plot
contour_stress_plot(mesh)