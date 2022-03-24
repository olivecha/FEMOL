import FEMOL

# guitar mesh
problem_mesh = FEMOL.mesh.guitar(lcar=0.05)

# flax material definition
flax_base = 0.003
n_plies_flax = 6
flax = FEMOL.materials.general_flax()  # material from library
flax_call = {key:flax.__dict__[key] for key in ['name', 'Ex', 'Ey', 'Es', 'vx', 'rho']}
Es  = 1.5e9  # Shear modulus (Pa)
flax_call['Es'] = Es
flax_call['ho'] = flax_base / n_plies_flax
flax.__init__(**flax_call)
flax.rho = 1100  # density kg/m^3

# Flax layup definition
plies_flax = [0, 0, 0, 0, 0, 0]
flax_base = FEMOL.laminate.Layup(material=flax, plies=plies_flax, symetric=False)

# Define the FEM problem
problem = FEMOL.FEM_Problem(mesh=problem_mesh, physics='modal', model='plate')
problem.define_materials(flax)
problem.define_tensors(flax_base)
# Clamped
problem.add_fixed_domain(FEMOL.domains.outside_guitar(L=1))
# Simply supported ?

w_ref, v_ref = problem.solve(filtre=0)