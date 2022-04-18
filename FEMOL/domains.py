import numpy as np
from scipy.optimize import fsolve, broyden1

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

    :return Inside circle domain
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
    :rtype: list
    :param p2: Slope at point 2
    :param p1: Slope at point 1
    :param x1: first point x
    :param y1: first point y
    :param x2: second point x
    :param y2: second point y
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

    return fsolve(equations, np.array([1, 1, 1, 1]))


def outside_ellipse(center, start, stop, eps=1e-3):
    """function creating a domain function representing the domain outside an ellipse"""
    # Get the ellipse center and points
    h, k = center
    x1, y1 = start
    x2, y2 = stop

    # Ellipse function to find the axis dimensions
    def ellipse(x):
        s1 = (x1 - h) ** 2 / x[0] ** 2 + (y1 - k) ** 2 / x[1] ** 2 - 1
        s2 = (x2 - h) ** 2 / x[0] ** 2 + (y2 - k) ** 2 / x[1] ** 2 - 1
        return [s1, s2]

    # Axis dimension
    sol = broyden1(ellipse, center)
    a, b = sol

    # Ellipse domain function
    def domain(x, y):
        Ri = (x - h) ** 2 / (a - eps) ** 2 + (y - k) ** 2 / (b - eps) ** 2
        if Ri < 1:
            return False
        else:
            return True

    return domain


def outside_guitar(L):
    """ Function creating a domain function for the guitar boundary"""

    # Left ellipse
    elsa1 = (0, 0.38 * L)
    elso1 = (0.25 * L, 0.76 * L)
    elc1 = (0.25 * L, 0.38 * L)
    ellipse1 = outside_ellipse(elc1, elsa1, elso1)

    # Right ellipse
    elc2 = (0.8175 * L, 0.38 * L)
    elsa2 = (0.8175 * L, 0.09*L)
    elso2 = (1*L, 0.38 * L)
    ellipse2 = outside_ellipse(elc2, elsa2, elso2)

    # Top and bottom curves
    # Top side 1
    p1 = create_polynomial(0.25 * L, 0.76 * L, 0.625 * L, (0.71225 - 0.1645 / 2) * L, 0)
    # Top side 2
    p2 = create_polynomial(0.625 * L, 0.71225 - 0.1645 / 2, 0.8175 * L, 0.67, 0)
    # Bottom side 1
    p3 = create_polynomial(0.25 * L, 0, 0.625 * L, (0.04775 + 0.1645 / 2) * L, 0)
    # Bottom side 2
    p4 = create_polynomial(0.625 * L, 0.04775 + 0.1645 / 2, 0.8175 * L, 0.09, 0)

    def top(x):
        if (x > 0.25 * L) & (x <= 0.625 * L):
            return p1[0] * x ** 3 + p1[1] * x ** 2 + p1[2] * x + p1[3]
        elif (x > 0.625 * L) & (x < 0.8175 * L):
            return p2[0] * x ** 3 + p2[1] * x ** 2 + p2[2] * x + p2[3]
        elif x <= 0.25 * L:
            return 0.76 * L
        elif x >= 0.8175 * L:
            return 0.67 * L

    def bottom(x):
        if (x > 0.25 * L) & (x <= 0.625 * L):
            return p3[0] * x ** 3 + p3[1] * x ** 2 + p3[2] * x + p3[3]
        elif (x > 0.625 * L) & (x < 0.8175 * L):
            return p4[0] * x ** 3 + p4[1] * x ** 2 + p4[2] * x + p4[3]
        elif x <= 0.25 * L:
            return 0
        elif x >= 0.8175 * L:
            return 0.09 * L

    def sides(x, y):
        if (x >= 0.25 * L) & (x <= 0.8175 * L):
            if (y > top(x) - (1e-3 / L)) | (y < bottom(x) + (1e-3 / L)):
                return True
            else:
                return False
        else:
            return True

    def domain(x, y):
        if ellipse1(x, y) & ellipse2(x, y) & sides(x, y):
            return True
        else:
            return False

    return domain


def guitar_domain(Lx, Ly):
    """
    Creates a fixed guitar domain from rectangle mesh dimensions
    Parameters
    ----------
    Lx : Mesh dim x
    Ly : Mesh dim y

    :return : Guitar domain true outside the guitar
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


def top_brace(L=1):
    domain = inside_box([[0.8175 * L - (L/30), 0.8175 * L + (L/20)]], [[0, L]])
    return domain
