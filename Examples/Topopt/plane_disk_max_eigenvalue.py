import FEMOL

n = 10
mesh = FEMOL.mesh.circle_Q4(1, 20)

problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
material1 = FEMOL.materials.IsotropicMaterial(2e8, 0.3, 2000)
problem.define_materials(material1, material1)
problem.define_tensors(1, 1)  # thick=1

problem.add_fixed_domain(FEMOL.domains.outside_circle(0, 0, 0.99), ddls=[0, 1])

w, v = problem.solve(filtre=0)
reference_vector = v[160]  # Save for MAC Analysis

SIMP = FEMOL.SIMP_VIBE(Problem=problem, objective='max eig')
mesh = SIMP.solve(v[160])