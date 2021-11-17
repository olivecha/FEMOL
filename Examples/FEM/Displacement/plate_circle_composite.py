import FEMOL
# to create a triangular mesh
import meshzoo
import numpy as np

"""
Circle of quads with Z load
"""

# Get a Q4 circle mesh (from meshzoo)
mesh = FEMOL.mesh.circle_Q4(1, 20)

# Display the mesh
mesh.display()

# Create a FEM Problem from the mesh (compute displacement with a plate bending model)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')

# Choose an orthortopic material
composite_material = FEMOL.materials.general_carbon()

# Create a composite material layup
plies = [0, 45, -45, -45, 45, 0]   # total plies
layup = FEMOL.laminate.Layup(composite_material, plies=plies, symetric=False)

# define the material and tensors the stiffness tensors are computed from the layup
problem.define_materials(composite_material)
problem.define_tensors(layup)

# Define the problem forces and boundary conditions
# Create a custom domain around the circle
def my_domain(x, y):
    # Circle radius minus a small value
    R = np.abs(mesh.points.max()) - 0.01
    # return true if the value is at the radius
    if np.sqrt(x**2 + y**2) >= R:
        return True
    else:
        return False

# Fix all the degrees of freedom
problem.add_fixed_domain(my_domain)

# Create a force domain at the center
force_domain = FEMOL.domains.inside_box([[-0.09, 0.09]], [[-0.09, 0.09]])
# Create a force vector
Fz = -1e3  # F = -10 kN
F = [0, 0, Fz, 0, 0, 0]   # F = [Fx, Fy, Fz, Mx, My, Mz] for the plate model
# add the forces
problem.add_forces(F, force_domain)

# Validate the problem definition
problem.plot()

# Assemble and solve (the solved problems returns the mesh with the computed data)
problem.assemble('K')
mesh = problem.solve()

# Display the Z displacement
mesh.plot.point_data('Z')
