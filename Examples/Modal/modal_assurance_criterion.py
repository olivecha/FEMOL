import FEMOL
import matplotlib.pyplot as plt

"""
Demonstration of the Modal Assurance Criterion between the modal
analysis of the same problem with different materials.
See : 
Pastor, M., Binda, M., & Harčarik, T. (2012). Modal Assurance Criterion. 
Procedia Engineering, 48, 543‑548. https://doi.org/10.1016/j.proeng.2012.09.551
"""

# 10x10 mesh
mesh = FEMOL.mesh.rectangle_Q4(1, 1, 10, 10)

# define the problem
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')

# Two different materials
material1 = FEMOL.materials.IsotropicMaterial(200e9, 0.3, 8000)
material2 = FEMOL.materials.IsotropicMaterial(170e9, 0.28, 4000)

# Simply supported
problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [[0, 10]]), ddls=[0, 1, 5])
problem.add_fixed_domain(FEMOL.domains.inside_box([0, 10], [[0, 10]]), ddls=[2, 4])
problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [0, 10]), ddls=[2, 3])

# Solve for first material
problem.define_materials(material1)
problem.define_tensors(0.05)
w, v1 = problem.solve(verbose=False, filtre=1)

# Solve for second material
problem.define_materials(material2)
problem.define_tensors(0.05)
w, v2 = problem.solve(verbose=False, filtre=1)

# Compute the Modal Assurance Criterion Matrix
mac_mat = FEMOL.utils.MAC_mat(v1, v2)
# Plot the first 40 modes
plt.imshow(mac_mat[:40, :40], cmap='rainbow')
plt.show()
