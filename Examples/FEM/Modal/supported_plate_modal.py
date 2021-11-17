import FEMOL
import numpy as np

# Simply supported reference eigenvalues (Hz)
REF_W = np.array([2.377, 5.942, 5.942, 9.507, 11.884, 11.884, 15.449, 15.449])

# Square mesh
mesh = FEMOL.mesh.rectangle_Q4(10, 10, 15, 15)

# FEM problem
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')

# Material with E = 200e9 Pa, mu = 0.5, rho = 8000
material = FEMOL.materials.IsotropicMaterial(200e9, 0.3, 8000)
problem.define_materials(material)
# Thickness  = 0.05 m
problem.define_tensors(0.05)

# Fixed X, Y, Tz displacement everywhere
problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [[0, 10]]), ddls=[0, 1, 5])
# Simply supported Y
problem.add_fixed_domain(FEMOL.domains.inside_box([0, 10], [[0, 10]]), ddls=[2, 4])
# Simply supported X
problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [0, 10]), ddls=[2, 3])

# Solve for eigenvalues and vectors
w, v = problem.solve(filtre=1)

print('Reference \t FEMOL')
print('__________________')
for w1, w2 in zip(REF_W, w):
    print(w1, ':', np.around(w2, 3))
