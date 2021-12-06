import unittest
import numpy as np
import FEMOL
import os

class MyTestCase(unittest.TestCase):

    def test_write_load(self):
        """
        Test the Mesh.save, mesh.load features
        """
        L = 15
        mesh = FEMOL.mesh.rectangle_Q4(L, L, L, L)

        # FEM Problem
        plate_FEM = FEMOL.FEM_Problem('displacement', 'plane', mesh)

        # Define the composite material layups
        material = FEMOL.materials.general_isotropic()

        plate_FEM.define_materials(material)
        plate_FEM.define_tensors(1)

        # Define the boundary conditions
        fixed_domain = FEMOL.domains.inside_box([0], [[0, L / 4]])
        plate_FEM.add_fixed_domain(fixed_domain, ddls=[0, 1])
        fixed_domain = FEMOL.domains.inside_box([0], [[3 * L / 4, L]])
        plate_FEM.add_fixed_domain(fixed_domain, ddls=[0, 1])

        # Define the applied force
        force = [0, -0.1]
        force_domain = FEMOL.domains.inside_box([L], [[L / 2 - 1, L / 2 + 1]])
        plate_FEM.add_forces(force, force_domain)

        topo_problem = FEMOL.SIMP_COMP(plate_FEM, volfrac=0.4, penal=3)
        mesh = topo_problem.solve(converge=0.03, max_iter=1, plot=False, save=False)
        mesh.save('temp')
        mesh2 = FEMOL.mesh.load_vtk('temp.vtk')
        os.remove('temp.vtk')
        self.assertTrue(np.allclose(mesh.point_data['Ux'], mesh2.point_data['Ux']))
        self.assertTrue(np.allclose(mesh.cell_data['X']['quad'], mesh2.cell_data['X']['quad']))

if __name__ == '__main__':
    unittest.main()
