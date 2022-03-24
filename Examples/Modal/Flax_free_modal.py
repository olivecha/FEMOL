import FEMOL
import numpy as np

#alphas = [1.875, 4.694, 7.885]
L = 0.245
b = 0.026
t = 0.00255
n = 8
hi = t/n
flax = FEMOL.materials.general_flax()
flax.hi = hi
flax.rho = 1000
plies = [0]*n
layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)

mesh = FEMOL.mesh.rectangle_Q4(L, b, 30, 3)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
problem.define_materials(flax)
problem.define_tensors(layup)
#problem.add_fixed_domain(domain=FEMOL.domains.inside_box([0], [[0, b]]))
w, v = problem.solve(filtre=0)

index1 = np.argmax([FEMOL.utils.MAC(vi, np.load('free_bar_mode0.npy')) for vi in v])
index2 = np.argmax([FEMOL.utils.MAC(vi, np.load('free_bar_mode1.npy')) for vi in v])

print('FEM : ', np.around(w[index1], 1))
print('FEM : ', np.around(w[index2], 1))
 
# Analytical values
alphas = [1.875, 4.694, 7.885]
I = (b*t**3)/12
A = b*t
E = flax.Ex
rho = flax.rho

for i, ai in enumerate(alphas[1:]):
    w = ai**2 * np.sqrt((E * I) / (rho * A * L**4))
    w *= (1/(2 * np.pi))
    print(f'EXP : {np.around(w, 1)}' )
    