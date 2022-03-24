import FEMOL

# define the layup
components = ['flax', 'resin', 'core', 'resin', 'carbon']
angles = [0, 0, 0, 0, 90]
thickness = [0.002, 0.0001, 0.005, 0.0001, 0.000125]
h = np.sum(thickness)

# define the materials
flax = FEMOL.materials.general_flax()
resin = FEMOL.materials.laminate_resin()
core = FEMOL.materials.laminate_core(infill_density=0.01)
carbon = FEMOL.materials.general_carbon()
materials_dict = {'flax':flax, 'resin':resin, 'core':core, 'carbon':carbon}

# Compute the A, B, D matrices
integration_points = [0]
_ = [integration_points.append(integration_points[i] + ti) for i, ti in enumerate(thickness)]
integration_points = np.array(integration_points) - integration_points[-1]/2

A, B, D, G = 0, 0, 0, 0

for theta, mtr, i in zip(angles, components, range(len(components))):
    A += materials_dict[mtr].Q_mat(theta)*(integration_points[i+1] - integration_points[i])
    G += materials_dict[mtr].G_mat(theta)*(integration_points[i+1] - integration_points[i])
    B += materials_dict[mtr].Q_mat(theta)*(integration_points[i+1]**2 - integration_points[i]**2)*(1/2)
    D += materials_dict[mtr].Q_mat(theta)*(integration_points[i+1]**3 - integration_points[i]**3)*(1/3)

G *= 5/6


# Compute the intertia tensor V
V1 = 0
for mtr, ti in zip(components, thickness):
    V1 += np.identity(3)*materials_dict[mtr].rho*ti

V2 = np.zeros((3, 3))

V3 = 0
centroidal_distances = np.abs([i_pt + ti/2 for i_pt, ti in zip(integration_points, thickness)])
for di, hi, mtr in zip(centroidal_distances, thickness, components):
    V3 += np.identity(3) * materials_dict[mtr].rho * ((hi**3)/12)

V = np.vstack([np.hstack([V1, V2]), np.hstack([V2, V3])])

problem_mesh = FEMOL.mesh.rectangle_Q4(0.2, 0.2, 16, 16)
problem2 = FEMOL.FEM_Problem(mesh=problem_mesh, physics='modal', model='plate')
problem2.tensors = [A, D, G, B]
problem2.V = V
w2, v2 = problem2.solve(filtre=0)
ind = np.argmax([FEMOL.utils.MAC(vi, np.load('free_plate_mode0_16x16.npy')) for vi in v2])
print(f'w = {w2[ind]} Hz')