import FEMOL

mesh = FEMOL.mesh.rectangle_Q4(75, 25, 75, 25)

problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
material1 = FEMOL.materials.isotropic_bending_benchmark()
problem.define_materials(material1)
problem.define_tensors(1)  # thick=1

problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, 15]]), ddls=[0, 1])

w, v = problem.solve(filtre=0)

reference_vector = v[0]  # Save for MAC Analysis

SIMP = FEMOL.SIMP_VIBE(problem, objective='max eig')
mesh = SIMP.solve(reference_vector, plot=False)
