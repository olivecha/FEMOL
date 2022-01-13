import FEMOL

# Square mesh
mesh = FEMOL.mesh.rectangle_Q4(1, 1, 15, 15)
# laminates and materials
plies1 = [0, 90, 45, -45]
plies2 = [0, 90]
flax = FEMOL.materials.general_flax()
carbon = FEMOL.materials.general_carbon()
layup1 = FEMOL.laminate.Layup(material=flax, plies=plies1, symetric=True)
layup2 = FEMOL.laminate.Layup(material=carbon, plies=plies2, symetric=True, h_core=layup1.h/2)
# FEM problem definition
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
problem.define_materials(flax, carbon)
problem.define_tensors(layup1, layup2)  # thick=1
problem.add_fixed_domain(FEMOL.domains.outside_box(0.01, 0.99, 0.01, 0.99))
# First modal solve
w, v = problem.solve(filtre=0)
# Find the vector of interest
mesh.add_mode('m', v[360], 6)
mesh.wrap('m', factor=1)
mesh.plot.point_data('m_Uz')
# solve the SIMP problem
SIMP = FEMOL.SIMP_VIBE(Problem=problem, objective='max eig')
mesh = SIMP.solve(v[360], save=False, plot=False)
mesh.plot.cell_data('X')
