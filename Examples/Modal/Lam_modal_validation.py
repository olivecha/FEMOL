import FEMOL
import matplotlib.pyplot as plt
import numpy as np

# Validation laminates in vibration
# Experimental data
EXP_1ST_EIGENFREQ = 181.8  # HZ
# Geometric data for the plate
b = 0.205
h = 0.244
t = 0.0035
n_plies = 8
rho = 1150
hi = t/n_plies  # ply thickness
# define the mesh
mesh = FEMOL.mesh.rectangle_Q4(b, h, 15, 15)  # mesh matching the dimensions


# Analytical frequency solution
def plate_frequency_THE(rho=1130, hi=0.0035/8):
    """ Compute the first mode eigenfrequency of an orthotropic plate"""
    a = h
    R = b/a
    flax = FEMOL.materials.general_flax()
    flax.rho = rho
    flax.hi = hi
    layup = FEMOL.laminate.Layup(material=flax, plies=[0]*8, symetric=False)
    D11 = layup.D_mat[0, 0]
    D12 = layup.D_mat[0, 1]
    D66 = layup.D_mat[2, 2]
    D22 = layup.D_mat[1, 1]
    w_11 = (np.pi**2 / (R**2 * b ** 2)) * (1 / np.sqrt(layup.mtr.rho)) * np.sqrt(D11 + 2*(D12 + 2*D66)*R**2 + D22*R**4)
    return w_11


# FEM Solution
def plate_frequency_FEM(rho=1130, hi=0.0035/8, ref_v=None):
    # material definition
    flax = FEMOL.materials.general_flax()  # material from library
    flax.hi = hi  # set the experimental ply thickness
    flax.rho = rho # set the experimental density

    # layup definition
    plies = [0]*n_plies # layup
    layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)

    # FEM problem definition
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
    problem.define_materials(flax)
    problem.define_tensors(layup)

    # First modal solve
    w, v = problem.solve(filtre=0)
    
    if ref_v:
        ind = np.argmax([FEMOL.utils.MAC(ref_v, vi) for vi in v])
        wj = w[ind]
        return wj
    else:
        return w, v
    
    
def density_study(ref_v):
    # eigenfrequency sensibility to density
    # density values to test
    rho_values = [1000, 1050, 1100, 1150, 1200, 1250]
    rho_gcm3 = [r/1000 for r in rho_values]
    freqs = []
    th_freqs = []
    
    for rho in rho_values:
        freqs.append(plate_frequency_FEM(rho=rho, ref_v=ref_v))
        th_freqs.append(plate_frequency_THE(rho))

    # plot the result
    fig, ax = plt.subplots(figsize=(6,6))
    plt.sca(ax)
    plt.plot(rho_gcm3, freqs, label='FEM')
    plt.plot([rho_gcm3[0], rho_gcm3[-1]], [EXP_1ST_EIGENFREQ]*2, label='EXP')
    plt.plot([1.14, 1.14], [freqs[-1], EXP_1ST_EIGENFREQ],'--', color='0.4', label='measured density')
    plt.plot(rho_values, th_freqs, label='THE')
    ax.set_xlabel('plate density (g/cm3)')
    ax.set_ylabel('First eigenfrequency (Hz)')
    ax.legend()
    plt.grid('on')


def ply_thickness_study(ref_v):

    # sensibility analysis to the plate thickness
    t_values = np.linspace(0.002, 0.005, 10)
    freqs = []
    for ti in t_values:
        # FEM problem
        flax = FEMOL.materials.general_flax()
        flax.hi = ti/n_plies
        flax.rho = 1140

        layup = FEMOL.laminate.Layup(material=flax, plies=[0]*8, symetric=False)
        # FEM problem definition

        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        problem.define_materials(flax)
        problem.define_tensors(layup)

        # First modal solve
        w, v = problem.solve(filtre=0)
        ind = np.argmax([FEMOL.utils.MAC(ref_v, vi) for vi in v])
        wj = w[ind]
        freqs.append(wj)

    # plot the results
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.sca(ax)
    eq_ply_thickness = 1000*np.array(t_values)/n_plies
    plt.plot(eq_ply_thickness, freqs, label = 'FEM')
    plt.plot([eq_ply_thickness[0], eq_ply_thickness[-1]], [EXP_1ST_EIGENFREQ]*2, label='EXP')
    plt.plot([t/n_plies, t/n_plies], [freqs[0], freqs[-1]], '--', color='0.4', label='measured ply thickness')
    ax.set_xlabel('Ply tickness used in the simulation (mm)')
    ax.set_ylabel('First eigenfrequency (Hz)')
    ax.legend()
    plt.grid('on')

    
def modulus_study():
    # Validation laminates in vibration
    EXP_1ST_EIGENFREQ = 181.8  # HZ
    # Plate dimensions
    b = 0.205
    h = 0.244
    n = 8
    t = 0.0035
    hi = t / n

    # FEM problem
    flax = FEMOL.materials.general_flax()
    rho = 1150
    flax.hi = t / 8
    flax.rho = rho
    mesh = FEMOL.mesh.rectangle_Q4(b, h, 15, 15)
    plies = [0] * 8
    layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)
    # FEM problem definition
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
    problem.define_materials(flax)
    problem.define_tensors(layup)  # thick=1
    # First modal solve
    w, v = problem.solve(filtre=0)
    ref_vector = v[6]

    Ex_values = 20 * np.logspace(7, 11, 15, base=10)
    Ey_values = 3 * np.logspace(8, 12, 15, base=10)
    Es_values = np.linspace(3e9, 10e9, 15)
    Gyz_values = Ey_values
    Gxz_values = Es_values

    mtr_call = {'name': None, 'Ex': None, 'Ey': None, 'Es': None, 'vx': None, 'rho': None, 'Gyz': None, 'Gxz': None}
    for k in mtr_call:
        mtr_call[k] = flax.__dict__[k]
    mtr_call['ho'] = flax.__dict__['hi']

    freqs_Ex = []
    for Exi in Ex_values:
        mtr_call['Ex'] = Exi
        flax.__init__(**mtr_call)
        layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        problem.define_materials(flax)
        problem.define_tensors(layup)
        w, v = problem.solve(filtre=0, verbose=False)
        ind = np.argmax([FEMOL.utils.MAC(ref_vector, vi) for vi in v])
        freqs_Ex.append(w[ind])

    mtr_call['Ex'] = FEMOL.materials.general_flax().Ex
    freqs_Ey = []
    for Eyi in Ey_values:
        mtr_call['Ey'] = Eyi
        flax.__init__(**mtr_call)
        layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        problem.define_materials(flax)
        problem.define_tensors(layup)
        w, v = problem.solve(filtre=0, verbose=False)
        ind = np.argmax([FEMOL.utils.MAC(ref_vector, vi) for vi in v])
        freqs_Ey.append(w[ind])

    mtr_call['Ey'] = FEMOL.materials.general_flax().Ey
    freqs_Es = []
    for Esi in Es_values:
        mtr_call['Es'] = Esi
        flax.__init__(**mtr_call)
        layup = FEMOL.laminate.Layup(material=flax, plies=plies, symetric=False)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        problem.define_materials(flax)
        problem.define_tensors(layup)
        w, v = problem.solve(filtre=0, verbose=False)
        ind = np.argmax([FEMOL.utils.MAC(ref_vector, vi) for vi in v])
        freqs_Es.append(w[ind])

    fig, ax = plt.subplots(figsize=(8.5, 6))
    plt.plot(Ex_values, freqs_Ex, label='Ex')
    plt.scatter(Ex_values[7], freqs_Ex[7])
    plt.plot(Ey_values, freqs_Ey, label='Ey')
    plt.scatter(Ey_values[3], freqs_Ey[3])
    plt.plot(Es_values, freqs_Es, label='Es')
    plt.scatter((Es_values[10] + Es_values[11]) / 2.005, (freqs_Es[10] + freqs_Es[11]) / 2.005)
    plt.legend()
    plt.xlabel('modulus (Pa)')
    plt.ylabel('Frequency (Hz)')
    plt.grid('on')
    plt.xscale('log')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(Es_values, freqs_Es, label='Es')
    plt.plot([Es_values[0], Es_values[-1]], 2 * [EXP_1ST_EIGENFREQ], label='THE')
    plt.xlabel('modulus (Pa)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid('on')