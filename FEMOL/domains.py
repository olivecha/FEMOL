import numpy as np
from scipy.optimize import fsolve

"""
Domain functions that create domains that can be applied to meshes
__________________________________________________________________
domain = FEMOL.domain.domain_creator_function(args)
nodes_in_domain = Mesh.domain_nodes(domain)
"""

def inside_box(xs, ys):
    def domain(x, y):
        cond_x = False
        cond_y = False
        for value in xs:
            if type(value) == list:
                if value[0] <= x <= value[1]:
                    cond_x = True
            else:
                if np.isclose(x, value, 1e-5):
                    cond_x = True
        for value in ys:
            if type(value) == list:
                if value[0] <= y <= value[1]:
                    cond_y = True
            else:
                if np.isclose(y, value, 1e-5):
                    cond_y = True

        return cond_x * cond_y

    return domain

def outside_box(x1, x2, y1, y2):
    """
    Returns  a domain outside a defined box
    x1 < x2 & y1 < y2
    """

    def box(x, y):
        return ~((x > x1) & (x < x2) & (y > y1) & (y < y2))

    return box

def inside_circle(x_pos, y_pos, R):
    """
    Creates a domain inside a circle
    Parameters
    ----------
    x_pos, y_pos : position of the center
    R : circle radius

    Returns : Inside circle domain
    -------

    """
    def circle(x, y):

        if (x - x_pos) ** 2 + (y - y_pos) ** 2 < R ** 2:
            return True
        else:
            return False

    return circle

def outside_circle(x_pos, y_pos, R):
    """
    Creates a domain outside a circle
    Parameters
    ----------
    (x_pos, y_pos) = position of the center
    R : circle radius

    Return
    -------
    Outside_circle domain
    """
    def circle(x, y):

        if (x - x_pos) ** 2 + (y - y_pos) ** 2 >= R ** 2:
            return True
        else:
            return False
    return circle

def create_polynomial(x1, y1, x2, y2, p1=0, p2=None):
    """
    Creates a 3rd order polynomial between points 1 and 2 with slope p and -p
    :param x1: first point x
    :param y1: first point y
    :param x2: second point x
    :param y2: second point y
    :param p: slope at point 1
    :return: a, b, c, d parameters such as y = ax^3 + bx^2 +cx +d
    """
    if p2 is None:
        p2 = -p1

    def equations(v):
        a, b, c, d = v
        eq1 = a * x1 ** 3 + b * x1 ** 2 + c * x1 + d - y1
        eq2 = a * x2 ** 3 + b * x2 ** 2 + c * x2 + d - y2
        eq3 = 3 * a * x1 ** 2 + 2 * b * x1 + c - p1
        eq4 = 3 * a * x2 ** 2 + 2 * b * x2 + c - p2
        return eq1, eq2, eq3, eq4

    a, b, c, d = fsolve(equations, (1, 1, 1, 1))

    return a, b, c, d

def guitar_domain(Lx, Ly):
    """
    Creates a fixed guitar domain from rectangle mesh dimensions
    Parameters
    ----------
    Lx : Mesh dim x
    Ly : Mesh dim y

    Returns : Guitar domain true outside the guitar
    -------
    """
    def guitar_sides(Lx, Ly):
        angle = np.pi / 6
        p = angle / (np.pi / 2)
        x1 = 2 * Ly / 6 + 2 * Ly / 6 * np.sin(angle)
        y1 = 2 * Ly / 6 - 2 * Ly / 6 * np.cos(angle)
        x2 = Lx - Ly / 4 - Ly / 4 * np.sin(angle)
        y2 = 2 * Ly / 6 - Ly / 4 * np.cos(angle)
        a, b, c, d = create_polynomial(x1, y1, x2, y2, p)

        def sides(x, y):
            Y_val = a * x ** 3 + b * x ** 2 + c * x + d
            return ~((x > x1) & (x < x2) & (y > Y_val) & (y < -Y_val + Ly))

        return sides

    # Outside domain
    circle1 = outside_circle((2*Ly/6), (2*Ly/6), (2*Ly/6))
    circle2 = outside_circle((2*Ly/6), (4*Ly/6), (2*Ly/6))
    circle3 = outside_circle((Lx-Ly/4) , 2*Ly/6, Ly/4)
    circle4 = outside_circle((Lx-Ly/4) , 4*Ly/6, Ly/4)
    box1 = outside_box(0, Lx, 2*Ly/6, 4*Ly/6)
    sides = guitar_sides(Lx, Ly)

    def fixed_guitar(x, y):
        """
        Fixed boundary conditions surrounding the guitar
        """
        if np.array([circle1(x, y), circle2(x, y), circle3(x, y), circle4(x, y), box1(x, y), sides(x, y)]).all():
            return True
        else:
            return False

    return fixed_guitar
