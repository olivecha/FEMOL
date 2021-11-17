import numpy as np
import sys
sys.path.append('../')
import FEMOL

def reference_2dof_stiffness_matrix():
    """
    Element stiffness matrix k1 from :
    Kattan, P. I. (2007). MATLAB guide to finite elements: an interactive approach (2nd ed). Berlin ; New York : Springer.
    Example 13.1, page 285.
    With E = 210e6, nu=0.3, t = 0.0250, a = b = 0.25
    Returns
    -------
    """

    Ke = 1e6 * np.array([[ 2.5962,  0.9375, -1.5865, -0.0721, -1.2981, -0.9375,  0.2885,  0.0721],
                         [ 0.9375,  2.5962,  0.0721,  0.2885, -0.9375, -1.2981, -0.0721, -1.5865],
                         [-1.5865,  0.0721,  2.5962, -0.9375,  0.2885, -0.0721, -1.2981,  0.9375],
                         [-0.0721,  0.2885, -0.9375,  2.5962,  0.0721, -1.5865,  0.9375, -1.2981],
                         [-1.2981, -0.9375,  0.2885,  0.0721,  2.5962,  0.9375, -1.5865, -0.0721],
                         [-0.9375, -1.2981, -0.0721, -1.5865,  0.9375,  2.5962,  0.0721,  0.2885],
                         [ 0.2885, -0.0721, -1.2981,  0.9375, -1.5865,  0.0721,  2.5962, -0.9375],
                         [ 0.0721, -1.5865,  0.9375, -1.2981, -0.0721,  0.2885, -0.9375,  2.5962], ])
    return Ke

def reference_T3_2dof_stiffnes_matrix():
    """
    Reference element stiffness matrix from :
    Kattan, P. I. (2007). MATLAB guide to finite elements: an interactive approach (2nd ed).
    Berlin ; New York : Springer. Example 11.1
    """
    Ke_ref = np.array([[2.0192, 0, 0, -1.0096, -2.0192, 1.0096],
                       [0, 5.7692, -0.8654, 0, 0.8654, -5.7692],
                       [0, -0.8654, 1.4423, 0, -1.4423, 0.8654],
                       [-1.0096, 0, 0, 0.5048, 1.0096, -0.5048],
                       [-2.0192, 0.8654, -1.4423, 1.0096, 3.4615, -1.8750],
                       [1.0096, -5.7692, 0.8654, -0.5048, -1.8750, 6.2740], ])

    return Ke_ref

def n_element_plane_isotropic_problem(n):
    Lx, Ly = n, n
    mesh_args = [n]*4
    mesh = FEMOL.core.RectangleQuadMesh(*mesh_args)
    problem = FEMOL.core.FEM_Problem('displacement', 'plane', mesh)
    problem.define_materials(FEMOL.materials.general_isotropic())
    problem.define_tensors(1)
    force_domains = create_domain([Lx], [[0, Ly]])  # domain where the force is applied
    forces = [2, 0]  # Force vector [Fx, Fy]
    problem.add_forces(forces, force_domains)  # Add the force on the domains to the problem
    domain = create_domain([0], [[0, Ly]])  # create a domain object
    problem.add_fixed_domain(domain)  # Fix the boundary
    problem.assemble('K')

    return problem

def reshape_Ke_into_plane_stress(Ke):

    K = np.vstack([np.hstack([Ke[:2, :2], Ke[:2, 6:8], Ke[:2, 12:14], Ke[:2, 18:20]]),
                   np.hstack([Ke[6:8, :2], Ke[6:8, 6:8], Ke[6:8, 12:14], Ke[6:8, 18:20]]),
                   np.hstack([Ke[12:14, :2], Ke[12:14, 6:8], Ke[12:14, 12:14], Ke[12:14, 18:20]]),
                   np.hstack([Ke[18:20, :2], Ke[18:20, 6:8], Ke[18:20, 12:14], Ke[18:20, 18:20]])])

    return K

def reshape_Ke_into_bending(Ke):
    # reshape into the element stiffness in bending
    Kb = np.vstack([np.hstack([Ke[2:5, 2:5], Ke[2:5, 8:11], Ke[2:5, 14:17], Ke[2:5, 20:23]]),
                    np.hstack([Ke[8:11, 2:5], Ke[8:11, 8:11], Ke[8:11, 14:17], Ke[8:11, 20:23]]),
                    np.hstack([Ke[14:17, 2:5], Ke[14:17, 8:11], Ke[14:17, 14:17], Ke[14:17, 20:23]]),
                    np.hstack([Ke[20:23, 2:5], Ke[20:23, 8:11], Ke[20:23, 14:17], Ke[20:23, 20:23]]), ])

    return Kb
