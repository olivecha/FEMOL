import unittest
import sys
sys.path.append('../')
import FEMOL
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_A_tensor_1(self):
        """
        Validation of the A matrix tensor from a laminate with laminate theory formulas from
        MECH530 course at McGill (Larry Lessard, 2020)
        """
        # reference A matrix
        A_ref = 1e6 * np.array([[136.231, 40.287, 0.],
                                [40.287, 179.097, 0.],
                                [0., 0., 48.833], ])

        material = FEMOL.materials.T300_N5208()
        plies_s = [0, 45, -45, 90, 90, 30, -30, 90]
        plies_t = np.hstack([plies_s, np.flip(plies_s)])
        layup1 = FEMOL.laminate.Layup(material=material, plies=plies_s, symetric=True)
        layup2 = FEMOL.laminate.Layup(material=material, plies=plies_t, symetric=False)

        self.assertTrue(np.allclose(layup1.A_mat, A_ref, atol=10e-4))
        self.assertTrue(np.allclose(layup2.A_mat, A_ref, atol=10e-4))

    def test_A_tensor_2(self):
        """
        Validation of the A matrix tensor computation with an example from MECH530 course.
        (Larry Lessard, McGill 2020)
        """
        A_ref = 1e6 * np.array([[211.678, 12.489, 0.],
                                [12.489, 60.27, 0.],
                                [0., 0., 18.899], ])

        material = FEMOL.materials.T300_N5208()
        plies_s = [0, 0, 0, 20, -20, 90]  # s
        plies_t = np.hstack([plies_s, np.flip(plies_s)])
        layup1 = FEMOL.laminate.Layup(material=material, plies=plies_s, symetric=True, z_core=0, h_core=0.010)
        layup2 = FEMOL.laminate.Layup(material=material, plies=plies_t, symetric=False, z_core=0, h_core=0.010)

        self.assertTrue(np.allclose(layup1.A_mat, A_ref, 10e-3))
        self.assertTrue(np.allclose(layup2.A_mat, A_ref, 10e-3))

    def test_D_tensor(self):
        """
        Validation of the D matrix computation test from MECH530 course example
        (Larry Lessard, McGill, 2020)
        """
        D_ref = np.array([[6292.09, 350.25, 15.41],
                          [350.25, 1599.72, 2.67],
                          [15.41, 2.67, 535.73], ])

        material = FEMOL.materials.T300_N5208()
        plies_s = [0, 0, 0, 20, -20, 90]  # s
        plies_t = np.hstack([plies_s, np.flip(plies_s)])
        layup1 = FEMOL.laminate.Layup(material=material, plies=plies_t, symetric=False, z_core=0, h_core=0.010)
        layup2 = FEMOL.laminate.Layup(material=material, plies=plies_s, symetric=True, z_core=0, h_core=0.010)

        self.assertTrue(np.allclose(D_ref, layup1.D_mat, 10e-3))
        self.assertTrue(np.allclose(D_ref, layup2.D_mat, 10e-3))

    def test_A_D_tensors(self):
        """
        A and D matrix tensor computation test
        Example from MECH530 course (Larry Lessard, McGill, 2020)
        """
        A_ref = 1e6 * np.array([[58.902, 28.55, 0.],
                                [28.55, 69.82, 0.],
                                [0., 0., 31.794], ])

        D_ref = np.array([[4.91, 4.82, 1.03],
                          [4.82, 9.66, 1.75],
                          [1.03, 1.75, 5.24], ])

        material = FEMOL.materials.AS4_PEEK()
        plies_s = [50, -50, 60, -60, 0]  # s
        plies_t = np.hstack([plies_s, np.flip(plies_s)])

        layup1 = FEMOL.laminate.Layup(material=material, plies=plies_t, symetric=False)
        layup2 = FEMOL.laminate.Layup(material=material, plies=plies_s, symetric=True)

        self.assertTrue(np.allclose(D_ref, layup1.D_mat, 10e-3))
        self.assertTrue(np.allclose(D_ref, layup2.D_mat, 10e-3))

        self.assertTrue(np.allclose(layup1.A_mat, A_ref, 10e-3))
        self.assertTrue(np.allclose(layup2.A_mat, A_ref, 10e-3))

    def test_D_tensor_summation(self):
        """
        Testing of D1 + D2 = D3 if plies1 + plies2 = plies3
        for D1 and D2 the tensors of two symmetric laminates
        """
        material = FEMOL.materials.general_carbon()
        plies1 = [0, 90, -45, 45]  # s
        layup1 = FEMOL.laminate.Layup(material=material, plies=plies1, symetric=True)  # no core
        plies2 = [45, -45]  # s
        layup2 = FEMOL.laminate.Layup(material=material, plies=plies2, symetric=True, h_core=layup1.h)
        plies3 = [45, -45, 0, 90, -45, 45]  # s
        layup3 = FEMOL.laminate.Layup(material=material, plies=plies3, symetric=True)  # no core
        combined_D = layup1.D_mat + layup2.D_mat

        self.assertTrue(np.allclose(combined_D, layup3.D_mat))

    def test_offset_D_summation(self):
        """
        Test if creating offset layups works as the equivalent layup with core.
        The D matrices of two layups off set from the laminate center should be equal to
        the total laminate
        """
        material = FEMOL.materials.AS4_PEEK()
        # total layup
        plies_all = [45, -45, -45, 45]
        layup_all = FEMOL.laminate.Layup(material=material, plies=plies_all, symetric=False, h_core=0.010)  # 10mm core
        # two halves
        plies_bottom = [45, -45]
        layup_b = FEMOL.laminate.Layup(material=material, plies=plies_bottom, symetric=False, h_core=0,
                                       z_core=-0.005 - material.hi)
        plies_top = [-45, 45]
        layup_t = FEMOL.laminate.Layup(material=material, plies=plies_top, symetric=False, h_core=0,
                                       z_core=0.005 + material.hi)

        self.assertTrue(np.allclose(layup_b.D_mat, layup_t.D_mat))
        self.assertTrue(np.allclose(layup_b.D_mat + layup_t.D_mat, layup_all.D_mat))

    def test_orthotropic_isotropic_tensors(self):
        """
        Tests that for an Orthotropic material instance with isotropic proprieties the
        tensors are equal to the Isotropic material tensors
        """
        Lx, Ly, n, m = 1, 1, 20, 20
        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, n, m)

        material1 = FEMOL.materials.isotropic_bending_benchmark()
        t = 0.1
        problem1 = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        problem1.define_materials(material1)
        problem1.define_tensors(t)

        material2 = FEMOL.materials.isotropic_laminate()
        layup = FEMOL.laminate.Layup(plies=[0] * 10, material=material2)
        problem2 = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        problem2.define_materials(material2)
        problem2.define_tensors(layup)

        self.assertTrue(np.allclose(problem1.C_A, problem2.C_A))
        self.assertTrue(np.allclose(problem1.C_D, problem2.C_D))
        self.assertTrue(np.allclose(problem1.C_G, problem2.C_G))


if __name__ == '__main__':
    unittest.main()
