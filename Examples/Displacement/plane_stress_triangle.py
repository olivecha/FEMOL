import FEMOL
# to create a triangular mesh
import meshzoo
import numpy as np

# Compute the mesh points and cells (taken from meshzoo : https://github.com/nschloe/meshzoo)
bary, cells = meshzoo.triangle(8)
corners = np.array([[0.0, -0.5 * np.sqrt(3.0), +0.5 * np.sqrt(3.0)], [1.0, -0.5, -0.5],])
points = np.dot(corners, bary).T
# Put the cell in a dict to create the FEMOL mesh
cells_dict = {'triangle': cells}

# Create a mesh with the T3 elements
mesh = FEMOL.Mesh(points, cells_dict, tri_element=FEMOL.elements.T3)

# Create a material with E = 100 MPa, mu = 0.3
E, mu, rho = 100e6, 0.3, 1
my_material = FEMOL.materials.IsotropicMaterial(E, mu, rho)

# define the part thickness
t = 1

# Create a FEM Problem from the mesh (compute displacement with a plane stress model)
problem = FEMOL.FEM_Problem(mesh, physics='displacement', model='plane')

# define the material and tensors (only the thickness is required for the tensor)
problem.define_materials(my_material)
problem.define_tensors(t)

# Define the problem forces and boundary conditions

# Create a rectangular domain from xmin to xmax at ymin
fixed_domain = FEMOL.domains.inside_box([[mesh.points.T[0].min(), mesh.points.T[0].max()]], [-0.5])

# Fix the two degrees of freedom
problem.add_fixed_domain(fixed_domain, ddls=[0, 1])

# Create a force domain at the top corner
force_domain = FEMOL.domains.inside_box([0], [1])
# Create a force vector
Fx = 1e6  # 1000 kN
Fy = 0.5e6  # 500 kN
F = [Fx, Fy]

# add the forces
problem.add_forces(F, force_domain)

# Validate the problem definition
problem.plot()

# Assemble and solve (the solved problems returns the mesh with the computed data)
problem.assemble('K')
mesh = problem.solve()

# visualize the results
mesh.plot.wrapped_2D()
