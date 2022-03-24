import FEMOL
import numpy as np


class GuitarSimpVibe(object):

    # problem data
    hc_opt = 0.010  # optimized core thickness
    h_flax = 0.003  # flax baseplate thickness
    h_carbon = 0.000250 # carbon coating plies thickness
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
    plies_carbon = [90, 90]
    carbon_layup = FEMOL.laminate.Layup(material=carbon, plies=plies_carbon, symetric=False,
                                        z_core=h / 2 - (n_plies_carbon / 2) * carbon.hi)

    def __init__(self, mode, mesh_lcar, volfrac=0):
        # define the mesh
        self.mode = mode
        self.lcar = mesh_lcar
        self.volfrac = volfrac
        self.mesh = FEMOL.mesh.guitar(lcar=mesh_lcar)
        # define the modal FEM problem
        # FEM problems definition
        self.problem = FEMOL.FEM_Problem(mesh=self.mesh, physics='modal', model='plate')
        self.problem.define_materials(self.flax, self.carbon)
        self.problem.define_tensors(self.flax_layup, self.carbon_layup)
        # clamped outer boundary
        self.problem.add_fixed_domain(FEMOL.domains.outside_guitar(L=1))

    def solve(self, **kwargs):
        try:
            vmac = np.load(f'Results/guitar_modes/guitar_mode_{self.mode}_lcar{str(self.lcar)[-2:]}.npy')
            print(f'solving SIMP problem for mode {self.mode} ' + FEMOL.utils.unique_time_string())
            SIMP = FEMOL.SIMP_VIBE(Problem=self.problem, volfrac=self.volfrac, objective='max eig')
            SIMP.void_domain = FEMOL.domains.inside_box([[0.57, 1]], [[0, 1]])
            self.mesh = SIMP.solve(vmac, save=True, plot=False, verbose=True, **kwargs)
            print(f'Successful solve for mode {self.mode}')
            now = FEMOL.utils.unique_time_string()
            np.save(f'batch_results/eigen_values_lcar{str(self.lcar)[-2:]}_{self.mode}_{now}', SIMP.lmbds)
        except Exception as exc:
            print(exc)
