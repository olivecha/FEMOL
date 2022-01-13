import FEMOL
import numpy as np
import matplotlib.pyplot as plt

# Convergence analysis for a plate deflection problem

# Solve over the mesh number for the T6 element
d1 = []
for S in np.arange(6, 25, 2):
    mesh = FEMOL.mesh.rectangle_T6(1, 1, S, S)
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
    problem.define_materials(FEMOL.materials.isotropic_bending_benchmark())
    problem.define_tensors(0.01)
    problem.add_fixed_domain(FEMOL.domains.outside_box(0.01, 0.99, 0.01, 0.99))
    problem.add_forces(force=[0, 0, -10, 0, 0, 0], domain=FEMOL.domains.inside_box([[0.05, 0.99]], [[0.05, 0.99]]))
    mesh = problem.solve(verbose=False)
    d1.append(mesh.point_data['Uz'].min())

# Solve over the mesh number for the Q4 element
d2 = []
for S in np.arange(6, 25, 2):
    mesh = FEMOL.mesh.rectangle_Q4(1, 1, S, S)
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
    problem.define_materials(FEMOL.materials.isotropic_bending_benchmark())
    problem.define_tensors(0.01)
    problem.add_fixed_domain(FEMOL.domains.outside_box(0.01, 0.99, 0.01, 0.99))
    problem.add_forces(force=[0, 0, -10, 0, 0, 0], domain=FEMOL.domains.inside_box([[0.05, 0.99]], [[0.05, 0.99]]))
    mesh = problem.solve(verbose=False)
    d2.append(mesh.point_data['Uz'].min())

# Comparative plot
plt.plot(np.arange(6, 21, 2), d2[:-2], label='Q4')
plt.plot(np.arange(6, 21, 2), d1[:-2], label='T6')
plt.legend()
plt.xlabel('mesh size')
plt.ylabel('max deflexion')
plt.title('T6 and Q4 elements mesh convergence')
#plt.gcf().savefig('conv_Q4_T6', dpi=200)