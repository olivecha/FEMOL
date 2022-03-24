import FEMOL
"""
Modal eigenvalue maximization for a simply supported beam
Example taken from :
Du, J., & Olhoff, N. (2007). Topological design of freely vibrating continuum structures 
for maximum values of simple and multiple eigenfrequencies and frequency gaps. Structural 
and Multidisciplinary Optimization, 34(2), 91‑110. https://doi.org/10.1007/s00158-007-0101-y
"""
# h = 1, w = 8 mesh
mesh = FEMOL.mesh.rectangle_Q4(80, 10, 64, 8)
# define the FEM parameters
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
material1 = FEMOL.materials.IsotropicMaterial(1000000, 0.3, 1000)
problem.define_materials(material1)
problem.define_tensors(0.001)  # thick=1
problem.add_fixed_domain(FEMOL.domains.inside_box([0], [5]), ddls=[0, 1])
problem.add_fixed_domain(FEMOL.domains.inside_box([80], [5]), ddls=[0, 1])
# Solve the eigenvalue problem
w, v = problem.solve(filtre=0)
# Find the eigenvector for the first mode and save it
mesh.add_mode('m', v[0], 2)
mesh.wrap('m', factor=1000)
mesh.plot.wrapped_2D()
reference_vector = v[0]
# solve the SIMP eigenvalue maximization problem
SIMP = FEMOL.SIMP_VIBE(problem, objective='max eig')
mesh = SIMP.solve(reference_vector, plot=False)
# Plot the result
mesh.plot.cell_data('X')
