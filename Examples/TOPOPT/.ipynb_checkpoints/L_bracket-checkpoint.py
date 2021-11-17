import FEMOL
import pygmsh

# Creating a mesh looking like the problem
with pygmsh.geo.Geometry() as geom:
    # Top circle
    circle = geom.add_circle([0.135, 0.160], 0.010, mesh_size=0.005, make_surface=False)
    # Bottom circles
    circle2 = geom.add_circle([0.012, 0.016], 0.0034 / 2, mesh_size=0.003, make_surface=False)
    circle3 = geom.add_circle([0.022, 0.016], 0.0034 / 2, mesh_size=0.003, make_surface=False)

    # main polygon
    poly = geom.add_polygon(
        [[0.0, 0.0],
         [0.152, 0.0],
         [0.152, 0.180],
         [0.113, 0.180],
         [0.113, 0.044],
         [0.0, 0.044], ],
        mesh_size=0.003,
        holes=[circle.curve_loop, circle2.curve_loop, circle3.curve_loop]
    )
    # Make it into quads
    geom.set_recombined_surfaces([poly.surface])
    # Create the meshio mesh
    mesh = geom.generate_mesh(dim=2)

# Transform into FEMOL mesh
mesh = FEMOL.Mesh(mesh.points, mesh.cells_dict)


# Create a FEM Problem from the mesh (compute displacement with a plane stress model)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')

# Define the problem material and part thickness
problem.define_materials(FEMOL.materials.IsotropicMaterial(100e2, 0.3, 1))
problem.define_tensors(1)
problem.assemble('K')

# FEM problem
problem = FEMOL.FEM_Problem('displacement', 'plane', mesh=mesh)

# Define material and thickness
problem.define_materials(FEMOL.materials.IsotropicMaterial(100e2, 0.3, 1))
problem.define_tensors(1)

# Add a force on the top circle nodes
top_circle = FEMOL.domains.inside_circle(*[0.135, 0.160], 0.011)
problem.add_forces([1, 1], top_circle)

# Fix the two bottom circles
bot_circle_1 = FEMOL.domains.inside_circle(*[0.012, 0.016], 0.0034/2 + 0.001)
bot_circle_2 = FEMOL.domains.inside_circle(*[0.022, 0.016], 0.0034/2 + 0.001)
problem.add_fixed_domain(bot_circle_1)
problem.add_fixed_domain(bot_circle_2)

# Plot the problem to see everything is alright
problem.plot()

# Solve the FEM problem
mesh = problem.solve(verbose=False)

# Plot the X displacement
mesh.plot.point_data('Ux')

# Define and solve the topology problem
topo_problem = FEMOL.TOPOPT_Problem(problem, volfrac=0.8, rmin=2.5, penal=3, method='SIMP')
mesh = topo_problem.solve(converge=0.02, max_loops=100, plot=True, save=True)
mesh.plot.cell_data('X')
