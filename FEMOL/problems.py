import FEMOL
import numpy as np


class GuitarDeflexion(object):

    # materials
    n_plies_carbon = 2
    n_plies_flax = 9
    plies_flax = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    flax = FEMOL.materials.general_flax()  # material from library
    carbon = FEMOL.materials.general_carbon()

    def __init__(self, lcar=0.03,
                 h_flax=0.0025, h_carbon=0.0002, hc_opt=0.010,
                 plies_carbon=None, factor=0.480, core_data_file=None):

        # mesh
        self.mesh = FEMOL.mesh.guitar_sym(L=1, lcar=lcar)
        self.factor = factor
        self.core_file = core_data_file

        # Ply thickness
        self.flax.hi = h_flax / self.n_plies_flax
        self.carbon.hi = h_carbon / self.n_plies_carbon
        # layups
        h = hc_opt + h_flax + h_carbon
        if plies_carbon is None:
            plies_carbon = [0, 90]
        z_flax = -h / 2 + (self.n_plies_flax / 2) * self.flax.hi
        self.flax_layup = FEMOL.laminate.Layup(material=self.flax, plies=self.plies_flax, symetric=False, z_core=z_flax)
        z_carbon = h / 2 - (self.n_plies_carbon / 2) * self.carbon.hi
        self.carbon_layup = FEMOL.laminate.Layup(material=self.carbon, plies=plies_carbon, symetric=False, z_core=z_carbon)

    def construct_FEM_problem(self):

        # Create a FEM Problem from the mesh (compute displacement with a plate bending model)
        problem = FEMOL.FEM_Problem(mesh=self.mesh, physics='displacement', model='plate')

        problem.add_fixed_domain(FEMOL.domains.outside_guitar(L=1))
        problem.define_materials(self.flax, self.carbon)
        problem.define_tensors(self.flax_layup, self.carbon_layup)
        x_values = [0.135 / self.factor, 0.165 / self.factor]
        y_values = [0.095 / self.factor, 0.265 / self.factor]
        bridge = FEMOL.domains.inside_box([x_values], [y_values])
        # string tension
        problem.add_forces([0, 0, 0, -21, 0, 0], bridge)
        zc_mesh = FEMOL.mesh.load_vtk(self.core_file)
        zc_mesh.cell_data['zc'] = FEMOL.utils.add_top_brace_to_zc(zc_mesh, zc_mesh.cell_data['zc'], 0.010)
        problem.mesh.cell_data['zc'] = zc_mesh.cell_data['zc']
        return problem

    def solve(self, h_min=0):
        problem = self.construct_FEM_problem()
        return problem.solve(solve_from_zc=True, h_min=h_min)


class GuitarSimpVibe(object):
    """ Class containing a guitar Topology optimization problem"""
    # problem data
    hc_opt = 0.010  # optimized core thickness
    h_flax = 0.003  # flax baseplate thickness
    h_carbon = 0.000250  # carbon coating plies thickness
    n_plies_carbon = 2  # N. of carbon plies
    n_plies_flax = 9  # N. of flax plies

    # flax material definition
    flax = FEMOL.materials.general_flax()  # material from library
    flax.hi = h_flax / n_plies_flax

    # carbon material definition
    carbon = FEMOL.materials.general_carbon()
    carbon.hi = h_carbon

    # Laminates definitions
    h = hc_opt + h_flax + h_carbon

    # Reference layups
    plies_flax = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    flax_layup = FEMOL.laminate.Layup(material=flax, plies=plies_flax, symetric=False,
                                      z_core=-h / 2 + (n_plies_flax / 2) * flax.hi)

    def __init__(self, mode, mesh_lcar, volfrac=0, p=3, q=1, plies_carbon=None):
        # define the mesh
        if plies_carbon is None:
            plies_carbon = [0, 90]
        self.p = p
        self.q = q
        self.mode = mode
        self.lcar = mesh_lcar
        self.volfrac = volfrac
        self.mesh = FEMOL.mesh.guitar_sym(lcar=mesh_lcar)
        carbon_layup = FEMOL.laminate.Layup(material=self.carbon, plies=plies_carbon, symetric=False,
                                            z_core=self.h / 2 - (self.n_plies_carbon / 2) * self.carbon.hi)
        # define the modal FEM problem
        # FEM problems definition
        self.problem = FEMOL.FEM_Problem(mesh=self.mesh, physics='modal', model='plate')
        self.problem.define_materials(self.flax, self.carbon)
        self.problem.define_tensors(self.flax_layup, carbon_layup)
        # clamped outer boundary
        self.problem.add_fixed_domain(FEMOL.domains.outside_guitar(L=1))
        self.SIMP = None

    def solve(self, save=True, plot=False, verbose=True, **kwargs):
        vmac = np.load(f'Results/guitar_modes/guitar_sym_mode_{self.mode}_lcar{str(self.lcar)[-2:]}.npy')
        print(f'solving SIMP problem for mode {self.mode} ' + FEMOL.utils.unique_time_string())
        self.SIMP = FEMOL.SIMP_VIBE(Problem=self.problem,
                                    volfrac=self.volfrac,
                                    FEM_solver_type='guitar',
                                    p=self.p,
                                    q=self.q)
        self.SIMP.void_domain = FEMOL.domains.inside_box([[0.57, 1]], [[0, 1]])
        self.mesh = self.SIMP.solve(vmac, save=save, plot=plot, verbose=verbose, **kwargs)
        print(f'Successful solve for mode {self.mode}')
        now = FEMOL.utils.unique_time_string()
        if save:
            np.save(f'eigen_values_lcar{str(self.lcar)[-2:]}_{self.mode}_{now}', self.SIMP.lmbds)
        return self.SIMP


class GuitarModal(object):
    """ Class representing a guitar modal analysis problem"""
    n_plies_carbon = 2
    n_plies_flax = 9
    # carbon material definition

    def __init__(self, mesh_lcar=0.03,
                 hc_opt=0.010, h_flax=0.003, h_carbon=0.000250,
                 plies_flax=None, plies_carbon=None):
        # Mesh definition
        self.mesh = FEMOL.mesh.guitar_sym(lcar=mesh_lcar)
        self.lcar = mesh_lcar
        # Materials and layups
        flax = FEMOL.materials.general_flax()  # material from library
        flax.hi = h_flax / self.n_plies_flax
        carbon = FEMOL.materials.general_carbon()
        carbon.hi = h_carbon / self.n_plies_carbon
        h = hc_opt + h_flax + h_carbon
        if plies_flax is None:
            plies_flax = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if plies_carbon is None:
            plies_carbon = [90, 90]
        flax_layup = FEMOL.laminate.Layup(material=flax, plies=plies_flax, symetric=False,
                                          z_core=-h / 2 + (self.n_plies_flax / 2) * flax.hi)
        carbon_layup = FEMOL.laminate.Layup(material=carbon, plies=plies_carbon, symetric=False,
                                            z_core=h / 2 - (self.n_plies_carbon / 2) * carbon.hi)
        # FEM problems definition
        self.problem = FEMOL.FEM_Problem(mesh=self.mesh, physics='modal', model='plate')
        self.problem.define_materials(flax, carbon)
        self.problem.define_tensors(flax_layup, carbon_layup)
        self.problem.add_fixed_domain(FEMOL.domains.outside_guitar(L=1))

    def solve(self, mac_find=True, **kwargs):
        # Solve the reference problem
        w_opt, v_opt = self.problem.solve(**kwargs)
        if mac_find:
            # Find guitar reference eigenfrequencies and vectors
            modes = ['T11', 'T21', 'T12', 'T31']
            try:  # loading the reference eigenvectors
                mac_vecs = [np.load(f'Results/guitar_modes/guitar_mode_{m}_lcar{str(self.lcar)[-2:]}.npy') for m in modes]
                idxs = [np.argmax([FEMOL.utils.MAC(vi, vref) for vi in v_opt]) for vref in mac_vecs]

            except FileNotFoundError:  # interpolate from existing ones
                print('No existing reference eigenvectors, interpolating new ones...')
                old_vecs = [np.load(f'Results/guitar_modes/guitar_mode_{m}_lcar04.npy') for m in modes]
                old_mesh = FEMOL.mesh.guitar(lcar=0.04)
                mac_vecs = [FEMOL.utils.interpolate_vector(v, old_mesh, self.mesh) for v in old_vecs]
                for v, m in zip(mac_vecs, modes):
                    np.save(f'Results/guitar_modes/guitar_mode_{m}_lcar{str(self.lcar)[-2:]}.npy', v)
                idxs = [np.argmax([FEMOL.utils.MAC(vi, vref) for vi in v_opt]) for vref in mac_vecs]

            except ValueError:
                print('Tried modal assurance criterion with non symmetric mesh and failed...')
                try:
                    mac_vecs = [np.load(f'Results/guitar_modes/guitar_sym_mode_{m}_lcar{str(self.lcar)[-2:]}.npy') for m in
                                modes]
                    idxs = [np.argmax([FEMOL.utils.MAC(vi, vref) for vi in v_opt]) for vref in mac_vecs]
                except FileNotFoundError:
                    print('No existing symmetric reference eigenvectors, interpolating new ones...')
                    old_vecs = [np.load(f'Results/guitar_modes/guitar_mode_{m}_lcar04.npy') for m in modes]
                    old_mesh = FEMOL.mesh.guitar(lcar=0.04)
                    mac_vecs = [FEMOL.utils.interpolate_vector(v, old_mesh, self.mesh) for v in old_vecs]
                    for v, m in zip(mac_vecs, modes):
                        np.save(f'Results/guitar_modes/guitar_sym_mode_{m}_lcar{str(self.lcar)[-2:]}.npy', v)
                    idxs = [np.argmax([FEMOL.utils.MAC(vi, vref) for vi in v_opt]) for vref in mac_vecs]

            return w_opt[idxs], v_opt[idxs]
        else:
            return w_opt, v_opt
