import FEMOL
import numpy as np
import pygmsh
import matplotlib.pyplot as plt

def reduce_mesh(mesh, tr):
    """
    Remove the mesh cell having a density lower than the threshold
    """
    X = mesh.cell_data['X']['quad']
    cells = mesh.cells['quad']
    new_cells = cells[X>tr]
    new_mesh = FEMOL.Mesh(mesh.points, {'quad':new_cells})
    for key in mesh.cell_data:
        new_mesh.cell_data[key] = {}
        new_mesh.cell_data[key]['quad'] = mesh.cell_data[key]['quad'][X>tr]
    new_mesh.point_data = mesh.point_data
    return new_mesh

def plot_L_outline1():
    """
    Plot the square 'L' bracket outline
    :return:
    """
    FEMOL.utils.plot_circle(pos_x=0.135, pos_y=0.160, r=0.01)
    FEMOL.utils.plot_circle(0.012, 0.016, 0.0034/2)
    FEMOL.utils.plot_circle(0.022, 0.016, 0.0034/2)
    points = np.array([[0.0, 0.0],
                       [0.152, 0.0],
                       [0.152, 0.180],
                       [0.113, 0.180],
                       [0.113, 0.044],
                       [0.0, 0.044],
                       [0, 0]])
    for i in range(points.shape[0]-1):
        plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], color='k')

def L_bracket_mesh1():
    """
    FEMOL mesh for the square L bracket
    """
    # Creating a mesh looking like the problem
    with pygmsh.geo.Geometry() as geom:
        # Top circle
        circle = geom.add_circle([0.135, 0.160], 0.010, mesh_size=0.005, make_surface=False)
        # Bottom circles
        circle2 = geom.add_circle([0.012, 0.016], 0.0034 / 2, mesh_size=0.005, make_surface=False)
        circle3 = geom.add_circle([0.022, 0.016], 0.0034 / 2, mesh_size=0.005, make_surface=False)

        # main polygon
        poly = geom.add_polygon(
            [[0.0, 0.0],
             [0.152, 0.0],
             [0.152, 0.180],
             [0.113, 0.180],
             [0.113, 0.044],
             [0.0, 0.044], ],
            mesh_size=0.005,
            holes=[circle.curve_loop, circle2.curve_loop, circle3.curve_loop]
        )
        # Make it into quads
        geom.set_recombined_surfaces([poly.surface])
        # Create the meshio mesh
        mesh = geom.generate_mesh(dim=2)

    # Transform into FEMOL mesh
    mesh = FEMOL.Mesh(mesh.points, mesh.cells_dict)
    return mesh

def L_bracket_mesh2(lcar=0.05):
    """
    Function returning the round L bracket mesh
    """
    with pygmsh.geo.Geometry() as geom:
        # Fixation holes
        h0 = geom.add_circle([0.135, 0.160], 0.010, lcar, make_surface=False)
        h1 = geom.add_circle([0.012, 0.016], 0.0034 / 2, lcar, make_surface=False)
        h2 = geom.add_circle([0.022, 0.016], 0.0034 / 2, lcar, make_surface=False)
        # Top arc radius
        R = (0.152 - 0.135)
        # Corner arc
        Rc =  0.025
        x = 0.118
        y = 0.032
        c = [x - Rc, y + Rc]
        # Bottom left arc
        asa11 = geom.add_point([(0.012 + 0.022)/2, 0.], lcar)
        ac11 = geom.add_point([(0.012 + 0.022)/2, 0.016], lcar)
        aso11 = geom.add_point([0, 0.016], lcar)
        arc1_1 = geom.add_circle_arc(asa11, ac11, aso11)
        aso12 = geom.add_point([(0.012 + 0.022)/2, 0.032], lcar)
        arc1_2 = geom.add_circle_arc(aso11, ac11, aso12)
        # Horizontal line 1
        lso1 = geom.add_point([c[0], 0.032], lcar)
        l1 = geom.add_line(aso12, lso1)
        # Corner arc
        ac2 = geom.add_point(c, lcar)
        aso2 = geom.add_point([0.135 - R, c[1]], lcar)
        arc2 = geom.add_circle_arc(lso1, ac2, aso2)
        # Vertical line
        lso2 = geom.add_point([0.135 - R, 0.160], lcar)
        l2 = geom.add_line(aso2, lso2)
        # Top arc
        ac3 = geom.add_point([0.135, 0.160], lcar)
        aso31 = geom.add_point([0.135,  0.160 + R], lcar)
        arc3_1 = geom.add_circle_arc(lso2, ac3, aso31)
        aso32 = geom.add_point([0.135 + R,  0.160], lcar)
        arc3_2 = geom.add_circle_arc(aso31, ac3, aso32)
        # Vertical line 2
        lso3 = geom.add_point([0.135 + R, Rc], lcar)
        l3 = geom.add_line(aso32, lso3)
        # Corner arc 2
        ac4 = geom.add_point([0.152-Rc, Rc], lcar)
        aso4 = geom.add_point([0.135 + R - Rc, 0.], lcar)
        arc4 = geom.add_circle_arc(lso3, ac4, aso4)
        # Horizontal line 2
        l4 = geom.add_line(aso4, asa11)
        loop1 = geom.add_curve_loop([arc1_1, arc1_2, l1, arc2, l2, arc3_1, arc3_2, l3, arc4, l4])
        loop2 = geom.add_curve_loop(h0.curve_loop.curves)
        loop3 = geom.add_curve_loop(h1.curve_loop.curves)
        loop4 = geom.add_curve_loop(h2.curve_loop.curves)
        s1 = geom.add_plane_surface(loop1, [loop2, loop3, loop4])
        geom.set_recombined_surfaces([s1])
        mesh = geom.generate_mesh(dim=2, order=1)

    mesh = FEMOL.Mesh(mesh.points, mesh.cells_dict)
    return mesh

def plot_L_outline2():
    """
    Plot the outline of the round L bracket mesh
    """
    ax = plt.gca()
    ax.set_aspect('equal')
    # Fixation holes
    FEMOL.utils.plot_circle(*[0.135, 0.160], 0.010)
    FEMOL.utils.plot_circle(*[0.012, 0.016], 0.0034 / 2)
    FEMOL.utils.plot_circle(*[0.022, 0.016], 0.0034 / 2)
    # Top arc radius
    R = (0.152 - 0.135)
    # Corner arc
    Rc = 0.025
    x = 0.118
    y = 0.032
    c = [x - Rc, y + Rc]
    # Left arc
    FEMOL.utils.plot_arc2([(0.012 + 0.022) / 2, 0.], [(0.012 + 0.022) / 2, 0.016], [(0.012 + 0.022) / 2, 0.032])
    # Horizontal line 1
    FEMOL.utils.plot_line([(0.012 + 0.022) / 2, 0.032], [c[0], 0.032])
    # Corner arc
    FEMOL.utils.plot_arc2([c[0], 0.032], c, [0.135 - R, c[1]], flip1=1, flip2=-1)
    # Vertical line
    FEMOL.utils.plot_line([0.135 - R, c[1]], [0.135 - R, 0.160])
    # Top arc
    FEMOL.utils.plot_arc2([0.135 - R, 0.160], [0.135, 0.160], [0.135 + R, 0.160], flip2=-1)
    # Vertical line 2
    FEMOL.utils.plot_line([0.135 + R, 0.160], [0.135 + R, Rc])
    # Corner arc 2
    FEMOL.utils.plot_arc2([0.135 + R, Rc], [0.152 - Rc, Rc], [0.135 + R - Rc, 0.], flip2=-1, flip1=1)
    # Horizontal line 2
    FEMOL.utils.plot_line([0.135 + R - Rc, 0.], [(0.012 + 0.022) / 2, 0.])