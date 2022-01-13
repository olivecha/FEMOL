import FEMOL

mesh = FEMOL.mesh.guitar()
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
material1 = FEMOL.materials.general_isotropic()
problem.define_materials(material1, material1)
problem.define_tensors(1, 1)  # thick=1
w, v = problem.solve(filtre=0)

mesh.add_mode('m',V=v[1314], N_dof=6)
mesh.plot.point_data('m_Uz')
