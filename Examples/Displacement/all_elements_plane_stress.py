import FEMOL
import numpy as np
import matplotlib.pyplot as plt

"""
Example showing how to solve the same problem for three different meshes with 
different element types
"""
# Create the three meshes
mesh_Q4 = FEMOL.mesh.rectangle_Q4(2, 1, 10, 10)
mesh_T3 = FEMOL.mesh.rectangle_T3(2, 1, 10, 10)
mesh_T6 = FEMOL.mesh.rectangle_T6(2, 1, 10, 10)
meshes = [mesh_Q4, mesh_T3, mesh_T6]
# Empty for solutions
solved_meshes = []
# Define and solve for every mesh
for mesh in meshes:
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')
    problem.define_materials(FEMOL.materials.general_isotropic())
    problem.define_tensors(1)
    problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, 1]]))
    problem.add_forces(force=[0, -0.01], domain=FEMOL.domains.inside_box([2], [[0, 1]]))
    solved_meshes.append(problem.solve(verbose=False))
# Plot the displacement for every mesh
for mesh, ele in zip(solved_meshes, ['Q4', 'T3', 'T6']):
    plt.figure()
    mesh.plot.point_data('Uy')
    d = mesh.point_data['Uy'].min()
    plt.title(ele + ' Deflexion max:' + str(np.around(d, 2)))
    # plt.gcf().savefig(ele, dpi=250)