import FEMOL
import matplotlib.pyplot as plt
import numpy as np

L = 0.205
b = 0.0245
t = 0.002
rho = 7840
E = 190e9
mu = 0.28

steel = FEMOL.materials.IsotropicMaterial(E=E, mu=0.325, rho=rho)
mesh = FEMOL.mesh.rectangle_Q4(L, b, 30, 3)
problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
problem.define_materials(steel)
problem.define_tensors(t)
problem.add_fixed_domain(domain=FEMOL.domains.inside_box([0], [[0, b]]))
w, v = problem.solve(filtre=0)

# Load the reference eigenvectors
files = ['Results/bar_eigenvectors/Al_bar_mode{m}.npy'.format(m=m) for m in [0, 1, 2]]
mode1, mode2, mode3 = [np.load(file) for file in files]
indexes = [np.argmax([FEMOL.utils.MAC(vi, v_ref) for vi in v]) for v_ref in [mode1, mode2, mode3]]
vectors = v[indexes]
frequencies = w[indexes]
fig, axs = plt.subplots(3, 1)

titles = ['mode 1', 'mode 2', 'mode 3']
for ax, vi, wi, ti  in zip(axs, vectors, frequencies, titles):
    plt.sca(ax)
    mesh.add_mode('m', vi, 6)
    mesh.plot.point_data('m_Uz', wrapped=False)
    ax.set_title(ti + ' frequency = {f}'.format(f=np.around(wi,1)))

# Theoretical frequencies
alphas = [1.875, 4.694, 7.885]
I = (b*t**3)/12
A = b*t

for i, ai in enumerate(alphas):
    w = ai**2 * np.sqrt((E * I) / (rho * A * L**4))
    w *= (1/(2 * np.pi))
    print(f'mode {i} : {np.around(w, 1)}' )
    