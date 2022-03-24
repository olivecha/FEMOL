import FEMOL
import numpy as np

# Experimental validation for laminate plate in free vibration

# Experimental data
EXPERIMENTAL_EIGENFREQUENCIES = [203.152, 723.772]  # HZ

# FEM problem

# plate dimensions
a = 0.1235 # Plate side dimension (m)
mesh = FEMOL.mesh.rectangle_Q4(a, a, 16, 16)
n = 8  # number of plies
t = 0.00245  # thickness (m)
hi = t/n  # ply thickness

# material definition
flax = FEMOL.materials.general_flax()  # material from library
flax_call = {key:flax.__dict__[key] for key in ['name', 'Ex', 'Ey', 'Es', 'vx', 'rho']}
flax_call['ho'] = hi
Es  = 1.5e9  # Shear modulus (Pa)
flax_call['Es'] = Es
flax.__init__(**flax_call)
flax.rho = 1000  # density kg/m^3

# layup definition
plies = [0]*n # layup
layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)

# FEM problem definition
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
problem.define_materials(flax)
problem.define_tensors(layup)
# define a fixed translation where the plate is suspended experimentally
problem.add_fixed_domain(FEMOL.domains.inside_box([[a/2 - 1e-3, a/2 + 1e-3]], [[a - 0.01, a]]), ddls=[0, 1])
# Solve
w, v = problem.solve(filtre=0, verbose=False)

ind1 = np.argmax([FEMOL.utils.MAC(vi, np.load('free_plate_mode0_16x16.npy')) for vi in v])
ind2 = np.argmax([FEMOL.utils.MAC(vi, np.load('free_plate_mode1_16x16.npy')) for vi in v])

print(f'FEM mode 1 : {np.around(w[ind1], 3)}, EXPE mode 1 : {EXPERIMENTAL_EIGENFREQUENCIES[0]}')
print(f'FEM mode 1 : {np.around(w[ind2], 3)}, EXPE mode 1 : {EXPERIMENTAL_EIGENFREQUENCIES[1]}')
