import FEMOL
import numpy as np

"""
Example with reference solution

Reference values taken from :
Park, C. I. (2008). Frequency equation for the in-plane vibration of a clamped circular plate.
Journal of Sound and Vibration, 313(1‑2), 325‑333. https://doi.org/10.1016/j.jsv.2007.11.034
"""
# Reference eigenvalues (Hz) (Park, C. I. (2008)).
REF_W = np.array([3363.6, 3836.4, 5217.5, 5380.5,
                  6624, 6749.3, 6929, 7019.3, 8093,
                  8476.5, 8530.6, 9258, 9328.1, 9887.7])

# Circle mesh with R = 0.5 and ~25 ** 2 elements
R = 0.5  # m
N_ele = 25
mesh = FEMOL.mesh.circle_Q4(R, N_ele)

# Problem definition
thickness = 0.005
aluminium = FEMOL.materials.IsotropicMaterial(71e9, 0.33, 2700)

# Create a FEM Problem from the mesh (compute displacement with a plate bending model)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
problem.define_materials(aluminium)
problem.define_tensors(thickness)

circle_domain = FEMOL.domains.outside_circle(0, 0, R - 0.005)

# Fix all the degrees of freedom
problem.add_fixed_domain(circle_domain)
problem.assemble('K')
problem.assemble('M')

# Solve the eigenvalue problem and store the frequencies
w, _ = problem.solve(filtre=2)

# Print the results
print('ID  |   FEMOL   |  REF.    |  DIFF(%) ')
print('______________________________________')
for i, r in enumerate(REF_W):
    diff_pr100 = 100 * np.abs(np.abs(r - w[i]) / (r + w[i]) / 2)
    index = str(i) + ' ' if len(str(i)) == 1 else str(i)
    FEM_W = str(np.around(w[i], 1))
    value = '  ' + FEM_W if len(FEM_W) == 6 else ' ' + FEM_W
    print(index, ' |', value, ' | ', r, ' | ', np.around(diff_pr100, 3))
