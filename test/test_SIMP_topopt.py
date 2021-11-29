import unittest
import FEMOL
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_2dof_ordered_mesh_SIMP_topopt(self):
        # Mesh
        nelx = 30
        nely = 15
        Lx = nelx
        Ly = nely
        mesh = FEMOL.mesh.rectangle_Q4(nelx, nely, Lx, Ly)

        # FEM Problem
        plate_FEM = FEMOL.FEM_Problem('displacement', 'plane', mesh)

        # Define the composite material layups
        material = FEMOL.materials.general_isotropic()

        plate_FEM.define_materials(material)
        plate_FEM.define_tensors(1)

        # Define the boundary conditions
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly / 4]])
        plate_FEM.add_fixed_domain(fixed_domain, ddls=[0, 1])
        fixed_domain = FEMOL.domains.inside_box([0], [[3 * Ly / 4, Ly]])
        plate_FEM.add_fixed_domain(fixed_domain, ddls=[0, 1])

        # Define the applied force
        force = [0, -0.1]
        force_domain = FEMOL.domains.inside_box([Lx], [[Ly / 2 - 1, Ly / 2 + 1]])
        plate_FEM.add_forces(force, force_domain)

        topo_problem = FEMOL.SIMP_COMP(plate_FEM, volfrac=0.4, penal=3)
        mesh = topo_problem.solve(converge=0.03, max_iter=1, plot=False, save=False)
        X = np.array(list(mesh.cell_data['X'].values())).flatten()
        self.assertTrue(np.isclose(np.sum(X), mesh.N_ele*0.4, 1e-2))


if __name__ == '__main__':
    unittest.main()
