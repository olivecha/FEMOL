import FEMOL
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
from matplotlib import patches

"""
Plot/Validation functions
"""


def validate_N_fun():
    """
    Plot for the 4 interpolation functions of the quadrilateral reference element
    """

    def N1(xi, eta):
        return 0.25 * (1 - xi) * (1 - eta)

    def N2(xi, eta):
        return 0.25 * (1 + xi) * (1 - eta)

    def N3(xi, eta):
        return 0.25 * (1 + xi) * (1 + eta)

    def N4(xi, eta):
        return 0.25 * (1 - xi) * (1 + eta)

    XI = np.linspace(-1, 1, 30)
    ETA = np.linspace(-1, 1, 30)
    XI, ETA = np.meshgrid(XI, ETA)
    Z1 = N1(XI, ETA)
    Z2 = N2(XI, ETA)
    Z3 = N3(XI, ETA)
    Z4 = N4(XI, ETA)
    Zi = [Z1, Z2, Z3, Z4]

    plt.figure(figsize=(14, 6))

    for i in range(0, 4):
        ax = plt.subplot(1, 4, i + 1, projection='3d')
        ax.scatter([-1, 1, 1, -1], [-1, -1, 1, 1], 0, c='k')
        for x, y, node in zip([-1, 1, 1, -1], [-1, -1, 1, 1], range(1, 5)):
            ax.text(x, y, 0, str(node), size=15, zorder=1, color='k')
        ax.plot_surface(XI, ETA, Zi[i], color='w')
        plt.title('N' + str(i + 1))


def plot_circle(pos_x, pos_y, r):
    ax = plt.gca()
    theta = np.linspace(0, 2 * np.pi, 200)
    x = pos_x + r * np.cos(theta)
    y = pos_y + r * np.sin(theta)
    ax.plot(x, y, color='k')


def plot_arc(x0, x1, pos_x, pos_y, r, side):
    ax = plt.gca()
    x = np.linspace(x0, x1, 100)
    y1_1 = np.sqrt(r ** 2 - (x - pos_x) ** 2) + pos_y
    y1_2 = -np.sqrt(r ** 2 - (x - pos_x) ** 2) + pos_y
    if side == 'lower':
        ax.plot(x, y1_1, color='k')
    if side == 'upper':
        ax.plot(x, y1_2, color='k')


def plot_arc2(sta, c, sto, flip1=-1, flip2=1):
    r = np.sqrt((sta[0] - c[0]) ** 2 + (sta[1] - c[1]) ** 2)
    A1 = np.arctan2(sta[1] - c[1], sta[0] - c[0])
    A2 = np.arctan2(sto[1] - c[1], sto[0] - c[0])
    T = -np.linspace(A1, A2)
    x = flip1 * r * np.cos(T) + c[0]
    y = flip2 * r * np.sin(T) + c[1]
    ax = plt.gca()
    ax.plot(x, y, color='k')


def plot_ellipse_arc():
    # TODO
    pass


def plot_line(p1, p2):
    ax = plt.gca()
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k')


def guitar_domain_figure():
    """
    Plots the figure corresponding to a definition of the guitar domain
    :return: None
    """
    fig, ax = plt.subplots()
    Lx = 60
    Ly = 45

    # Circles
    plot_circle(2 * Ly / 6, 2 * Ly / 6, 2 * Ly / 6)
    plot_circle(2 * Ly / 6, 4 * Ly / 6, 2 * Ly / 6)
    plot_circle(Lx - Ly / 4, 2 * Ly / 6, Ly / 4)
    plot_circle(Lx - Ly / 4, 4 * Ly / 6, Ly / 4)
    ax.plot([0, 0], [2 * Ly / 6, 4 * Ly / 6], color='k')
    ax.plot([Lx, Lx], [2 * Ly / 6, 4 * Ly / 6], color='k')

    # Arcs
    angle = np.pi / 6
    p = angle / (np.pi / 2)
    # Point 1
    x1 = 2 * Ly / 6 + 2 * Ly / 6 * np.sin(angle)
    y1 = 2 * Ly / 6 - 2 * Ly / 6 * np.cos(angle)
    # Point 2
    x2 = Lx - Ly / 4 - Ly / 4 * np.sin(angle)
    y2 = 2 * Ly / 6 - Ly / 4 * np.cos(angle)

    a, b, c, d = FEMOL.domains.create_polynomial(x1, y1, x2, y2, p)
    x = np.linspace(x1, x2, 100)
    y = a * x ** 3 + b * x ** 2 + c * x + d  # 3nd order polynomial
    ax.plot(x, y, color='k')
    ax.plot(x, -y + Ly, color='k')

    # Soundhole
    plot_circle(2 * Lx / 3, Ly / 2, Ly / 6)
    pos_x = 2 * Lx / 3
    pos_y = Ly / 2
    r = Ly / 6
    x1 = np.linspace(pos_x - r, pos_x + r, 100)
    y1_1 = np.sqrt(r ** 2 - (x1 - pos_x) ** 2) + pos_y
    y1_2 = -np.sqrt(r ** 2 - (x1 - pos_x) ** 2) + pos_y
    ax.fill(x1, y1_1, color='k')
    ax.fill(x1, y1_2, color='k')

    # Settings
    ax.set_aspect('equal')
    ax.set_xlim(-1, Lx + 1)
    ax.set_ylim(-1, Ly + 1)
    ax.set_axis_off()
    plt.show()


def guitar_outline(Lx, Ly):
    """
    Plots the figure corresponding to a definition of the guitar domain
    :return: None
    """
    ax = plt.gca()

    # Polynomials
    angle = np.pi / 6
    p = angle / (np.pi / 2)
    # Point 1
    x1 = 2 * Ly / 6 + 2 * Ly / 6 * np.sin(angle)
    y1 = 2 * Ly / 6 - 2 * Ly / 6 * np.cos(angle)
    # Point 2
    x2 = Lx - Ly / 4 - Ly / 4 * np.sin(angle)
    y2 = 2 * Ly / 6 - Ly / 4 * np.cos(angle)

    a, b, c, d = FEMOL.domains.create_polynomial(x1, y1, x2, y2, p)
    x = np.linspace(x1, x2, 100)
    y = a * x ** 3 + b * x ** 2 + c * x + d  # 3nd order polynomial
    ax.plot(x, y, color='k')
    ax.plot(x, -y + Ly, color='k')

    # Circles
    plot_arc(0, x1, 2 * Ly / 6, 2 * Ly / 6, 2 * Ly / 6, 'upper')
    plot_arc(0, x1, 2 * Ly / 6, 4 * Ly / 6, 2 * Ly / 6, 'lower')
    plot_arc(x2, Lx, Lx - Ly / 4, 2 * Ly / 6, Ly / 4, 'upper')
    plot_arc(x2, Lx, Lx - Ly / 4, 4 * Ly / 6, Ly / 4, 'lower')
    ax.plot([0, 0], [2 * Ly / 6, 4 * Ly / 6], color='k')
    ax.plot([Lx, Lx], [2 * Ly / 6, 4 * Ly / 6], color='k')

    # Soundhole
    plot_circle(2 * Lx / 3, Ly / 2, Ly / 7)


def guitar_outline2(L):
    ax = plt.gca()
    ax.set_aspect('equal')
    # TODO : Use ellipse arcs
    ellipse1 = patches.Ellipse((0.25 * L, 0.38 * L), 0.50 * L, 0.76 * L, fill=False)
    ellipse4 = patches.Ellipse((0.8175 * L, 0.38 * L), 0.365 * L, 0.58 * L, fill=False)
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse4)
    # Top 1
    p = FEMOL.domains.create_polynomial(0.25 * L, 0.76 * L, 0.625 * L, (0.71225 - 0.1645 / 2) * L, 0)
    x = np.linspace(0.25 * L, 0.625 * L, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')

    # Top 2
    p = FEMOL.domains.create_polynomial(0.625 * L, 0.71225 - 0.1645 / 2, 0.8175 * L, 0.67, 0)
    x = np.linspace(0.625, 0.8175, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Bot 1
    p = FEMOL.domains.create_polynomial(0.625 * L, 0.04775 + 0.1645 / 2, 0.8175 * L, 0.09, 0)
    x = np.linspace(0.625, 0.8175, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Bot 2
    p = FEMOL.domains.create_polynomial(0.25 * L, 0, 0.625 * L, (0.04775 + 0.1645 / 2) * L, 0)
    x = np.linspace(0.25 * L, 0.625 * L, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Soundhole
    FEMOL.utils.plot_circle(0.673 * L, 0.38 * L, 0.175 * L / 2)


"""
File management
"""


def unique_time_string():
    """
    Returns a unique date_time string to create filenames
    """
    date = str(datetime.datetime.now().date())
    time = str(datetime.datetime.now().time())[:-7].replace(':', '_')
    return date + '_' + time


def count_lines(start, lines=0, header=True, begin_start=None):
    if header:
        print('{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'FILE'))
        print('{:->11}|{:->11}|{:->20}'.format('', '', ''))

    for thing in os.listdir(start):
        thing = os.path.join(start, thing)
        if os.path.isfile(thing):
            if thing.endswith('.py'):
                with open(thing, 'r') as f:
                    newlines = f.readlines()
                    newlines = len(newlines)
                    lines += newlines

                    if begin_start is not None:
                        reldir_of_thing = '.' + thing.replace(begin_start, '')
                    else:
                        reldir_of_thing = '.' + thing.replace(start, '')

                    if header:
                        print('{:>10} |{:>10} | {:<20}'.format(
                            newlines, lines, reldir_of_thing))

    for thing in os.listdir(start):
        thing = os.path.join(start, thing)
        if os.path.isdir(thing):
            lines = count_lines(thing, lines, header=False, begin_start=start)

    return lines


def project_lines():
    dirs = ['FEMOL/', 'Examples/', 'test/']
    lines = 0
    for d in dirs:
        lines += count_lines(d, header=False)

    return lines


"""
Vibration videos
"""


def modal_dance(mesh, eigen_vectors, filename):
    x, y = mesh.CORG.transpose()
    x = np.unique(x)
    y = np.unique(y)
    xg, yg = np.meshgrid(x, y)

    Dxs = []
    Dys = []
    Dts = []

    for i in range(len(eigen_vectors.transpose()) - 1):
        vector1 = eigen_vectors.transpose()[i]
        vector2 = eigen_vectors.transpose()[i + 1]

        Dx1 = vector1[np.arange(0, len(vector1), 2)]
        Dx1 = np.flip(Dx1.reshape(mesh.nodes_y, mesh.nodes_x), axis=0)

        Dx2 = vector2[np.arange(0, len(vector2), 2)]
        Dx2 = np.flip(Dx2.reshape(mesh.nodes_y, mesh.nodes_x), axis=0)

        Dy1 = vector1[np.arange(1, len(vector1), 2)]
        Dy1 = np.flip(Dy1.reshape(mesh.nodes_y, mesh.nodes_x), axis=0)

        Dy2 = vector2[np.arange(1, len(vector2), 2)]
        Dy2 = np.flip(Dy2.reshape(mesh.nodes_y, mesh.nodes_x), axis=0)

        Dt1 = np.sqrt(Dx1 ** 2 + Dy1 ** 2)
        Dt2 = np.sqrt(Dx2 ** 2 + Dy2 ** 2)

        Dxs.append(np.linspace(Dx1, Dx2, 20))
        Dys.append(np.linspace(Dy1, Dy2, 20))
        Dts.append(np.linspace(Dt1, Dt2, 20))

    fig, ax = plt.subplots()
    fig.set_facecolor("k")
    ax.set_axis_off()
    ax.set_aspect('equal')

    # Create the artists
    ims = []
    scale = 100
    for Dx, Dy, Dt in zip(Dxs, Dys, Dts):
        for dx, dy, dt in zip(Dx, Dy, Dt):
            im = plt.pcolor(xg + dx * scale, yg + dy * scale, dt[:-1, :-1], cmap='inferno')
            ims.append([im])

    # Animate the artists
    ani = animation.ArtistAnimation(fig, ims, interval=80, blit=True)

    # Save the animation
    ani.save(filename + '.mp4')


def first_4_modes(mesh, filename, eigen_vectors, scale=40):
    x, y = mesh.CORG.transpose()
    x = np.unique(x)
    y = np.unique(y)
    xg, yg = np.meshgrid(x, y)
    vectors = eigen_vectors.transpose()[:4]

    Dxss, Dyss, Dtss = [], [], []
    for vector in vectors:
        Dx = vector[np.arange(0, len(vector), 2)]
        Dx = np.flip(Dx.reshape(mesh.nodes_y, mesh.nodes_x), axis=0)
        Dy = vector[np.arange(1, len(vector), 2)]
        Dy = np.flip(Dy.reshape(mesh.nodes_y, mesh.nodes_x), axis=0)
        Dt = np.sqrt(Dx ** 2 + Dy ** 2)

        Dxs = np.append(np.linspace(-Dx, Dx, 20), np.linspace(Dx, -Dx, 20), axis=0)
        Dys = np.append(np.linspace(-Dy, Dy, 20), np.linspace(Dy, -Dy, 20), axis=0)
        Dts = np.abs(np.append(np.linspace(-Dt, Dt, 20), np.linspace(Dt, -Dt, 20), axis=0))

        Dxss.append(Dxs)
        Dyss.append(Dys)
        Dtss.append(Dts)

    fig, axes = plt.subplots(2, 2)
    fig.set_facecolor("k")
    for axs in axes:
        for ax in axs:
            ax.set_axis_off()
            ax.set_aspect('equal')
    plt.tight_layout()

    # Create the artists
    ims = []
    for _ in range(5):
        for i in range(len(Dxss[1])):
            im1 = axes[0, 0].pcolor(xg + Dxss[0][i] * scale, yg + Dyss[0][i] * scale, Dtss[0][i][:-1, :-1],
                                    cmap='inferno')
            im2 = axes[0, 1].pcolor(xg + Dxss[1][i] * scale, yg + Dyss[1][i] * scale, Dtss[1][i][:-1, :-1],
                                    cmap='inferno')
            im3 = axes[1, 0].pcolor(xg + Dxss[2][i] * scale, yg + Dyss[2][i] * scale, Dtss[2][i][:-1, :-1],
                                    cmap='inferno')
            im4 = axes[1, 1].pcolor(xg + Dxss[3][i] * scale, yg + Dyss[3][i] * scale, Dtss[3][i][:-1, :-1],
                                    cmap='inferno')
            ims.append([im1, im2, im3, im4])

    # Animate the artists
    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True)

    # Save the animation
    ani.save(filename + '.mp4')


"""
Modal analysis 
"""


def MAC(v1, v2):
    """
    Modal Assurance Criterion between two modal vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: mac
    Taken from :
    Pastor, M., Binda, M., & Harčarik, T. (2012). Modal Assurance Criterion.
    Procedia Engineering, 48, 543‑548. https://doi.org/10.1016/j.proeng.2012.09.551
    """
    mac = abs(v1.T @ v2) ** 2 / ((v1.T @ v1) * (v2.T @ v2))
    return mac


def MAC_mat(modes1, modes2):
    """
    Modal assurance criterion between two sets of modal vectors
    :param modes1: set one of modal vectors
    :param modes2: set two of modal vectors
    :return: Modal Assurance Criterion Matrix
    """
    mat = []
    for v1 in modes1:
        line = [MAC(v1, v2) for v2 in modes2]
        mat.append(line)
    return np.array(mat)


""" 
Special material tensors
"""


def elevated_isotropic_tensor(t, z, mtr):
    """
    Elevated plate bending 3x3 stiffness tensor
    """
    E = mtr.E
    mu = mtr.mu
    Ix = t ** 3 / 12 + t * z ** 2

    D = Ix * E / (1 - mu ** 2) * np.array([[1, mu, 0],
                                          [mu, 1, 0],
                                          [0, 0, (1 - mu) / 2], ])

    return D
