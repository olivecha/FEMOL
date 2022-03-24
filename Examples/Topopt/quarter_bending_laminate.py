import FEMOL

"""
quarter simply supported plate laminate core height optimization
problem taken from :
Harvey, D., & Hubert, P. (2022). 3D topology optimization of sandwich structures with anisotropic shells.
Composite Structures, 285, 115237. https://doi.org/10.1016/j.compstruct.2022.115237
"""
# Create a quarter plate mesh
L = 0.256/2 # m
edge = 0.025
N_ele = 50
mesh = FEMOL.mesh.rectangle_Q4(L, L, N_ele, N_ele)
# Define the FEM problem
problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
carbon = FEMOL.materials.general_carbon()
carbon.hi = 0.00025
h = 0.02 + 12*carbon.hi 
plies_base = [0, 45, -45, 90]#s
plies_coat = [0, 90]#s
layup_base = FEMOL.laminate.Layup(material=carbon, plies=plies_base, z_core= -h/2 + 4*carbon.hi)
layup_coat = FEMOL.laminate.Layup(material=carbon, plies=plies_base, z_core= h/2 - 2*carbon.hi)
problem.define_materials(carbon, carbon)
problem.define_tensors(layup_base, layup_coat)
problem.add_fixed_domain(ddls=[0], domain = FEMOL.domains.inside_box([ L], [[0, L]]))
problem.add_fixed_domain(ddls=[1], domain = FEMOL.domains.inside_box([[0, L]], [ L]))
problem.add_fixed_domain(ddls=[2, 4], domain=FEMOL.domains.inside_box([[0, L]], [0]))
problem.add_fixed_domain(ddls=[2, 3], domain=FEMOL.domains.inside_box([0], [[0, L]]))
#problem.add_fixed_domain(ddls=[2], domain = FEMOL.domains.inside_box([0],[0]))
problem.add_forces(force=[0, 0, -10, 0, 0, 0], domain=FEMOL.domains.inside_circle(R=L/20, x_pos=L, y_pos=L))

topo_problem = FEMOL.SIMP_COMP(problem, volfrac=0.8, penal=1.5)
domain1 = FEMOL.domains.inside_box([[0, edge]], [[0, L]])
domain2 = FEMOL.domains.inside_box([[0, L]], [[0, edge]])
def domain(*points):
    return domain1(points[0],points[1]) | domain2(points[0],points[1])
topo_problem.void_domain = domain
mesh = topo_problem.solve(converge=0.01, max_iter=50, plot=True, save=False)

mesh.cell_to_point_data('X')
mesh.plot.point_data('X')