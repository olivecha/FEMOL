import unittest
import numpy as np
import FEMOL
import FEMOL.misc
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

    def test_stress_from_displacement_plane(self):
        """
        Test the stress computation on the mesh for a plane stress problem
        """
        thickness = 0.1
        Lx, Ly = 20, 10
        F = 0.5
        mesh = FEMOL.mesh.rectangle_Q4(20, 10, 10, 30)
        problem = FEMOL.FEM_Problem('displacement', 'plane', mesh)
        material = FEMOL.materials.general_isotropic()
        problem.define_materials(material)
        problem.define_tensors(thickness)
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0])
        fixed_domain = FEMOL.domains.inside_box([0], [Ly / 2])
        problem.add_fixed_domain(fixed_domain, ddls=[1])
        force_domain_1 = FEMOL.domains.inside_box([Lx], [[0.01, Ly - 0.01]])
        problem.add_forces([F, 0], force_domain_1)
        force_domain_2 = FEMOL.domains.inside_box([Lx], [0, Ly])
        problem.add_forces([F / (30 - 1), 0], force_domain_2)
        mesh = problem.solve(verbose=False)
        # compute problem strain and stress
        mesh.stress_from_displacement(problem.tensors[0])
        # True value for stress
        A = thickness * Ly
        Sx = np.sum(problem.F) / A
        self.assertTrue(np.isclose(mesh.cell_data['Sx']['quad'].max() / thickness, Sx))

    def test_mesh_all_elements_plane_stress(self):
        mesh_Q4 = FEMOL.mesh.rectangle_Q4(2, 1, 10, 10)
        mesh_T3 = FEMOL.mesh.rectangle_T3(2, 1, 10, 10)
        mesh_T6 = FEMOL.mesh.rectangle_T6(2, 1, 10, 10)
        meshes = [mesh_Q4, mesh_T3, mesh_T6]
        solved_meshes = []

        for mesh in meshes:
            problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')
            problem.define_materials(FEMOL.materials.general_isotropic())
            problem.define_tensors(1)
            problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, 1]]))
            problem.add_forces(force=[0, -1], domain=FEMOL.domains.inside_box([2], [0]))
            problem.plot()

    def test_pygmsh_mesh(self):
        mesh = FEMOL.misc.L_bracket_mesh2()
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
        material1 = FEMOL.materials.IsotropicMaterial(2e8, 0.3, 2000)
        problem.define_materials(material1, material1)
        problem.define_tensors(1, 1)  # thick=1
        w, v = problem.solve(filtre=0)
        self.assertIsNotNone(w)

    def test_height_with_area(self):
        mesh = FEMOL.mesh.load_vtk('data/height_with_area_mesh.vtk')
        h = mesh.height_with_area('zc', 0.5)
        self.assertTrue(np.isclose(h, 0.0129))


if __name__ == '__main__':
    unittest.main()
