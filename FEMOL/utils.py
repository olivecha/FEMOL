import FEMOL
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
from scipy.interpolate import griddata, interp1d

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


def plot_ellipse_arc(center, a, b, theta_start, theta_stop):
    """ Plot an ellipse arc with dimensions a, b from theta start to theta stop at center"""
    ax = plt.gca()
    a, b = a / 2, b / 2
    theta = np.linspace(theta_start, theta_stop)
    r = (a * b) / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    ax.plot(x, y, color='k')


def plot_arc2(sta, c, sto, flip1=-1, flip2=1):
    r = np.sqrt((sta[0] - c[0]) ** 2 + (sta[1] - c[1]) ** 2)
    A1 = np.arctan2(sta[1] - c[1], sta[0] - c[0])
    A2 = np.arctan2(sto[1] - c[1], sto[0] - c[0])
    T = -np.linspace(A1, A2)
    x = flip1 * r * np.cos(T) + c[0]
    y = flip2 * r * np.sin(T) + c[1]
    ax = plt.gca()
    ax.plot(x, y, color='k')


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
    # Ellipse arc 1 (left side)
    a, b, c = 0.50 * L, 0.76 * L, (0.25 * L, 0.38 * L)
    FEMOL.utils.plot_ellipse_arc(c, a, b, np.pi / 2, 3 * np.pi / 2)
    # Ellipse arc 2 (right side)
    a, b, c = 0.365 * L, 0.58 * L, (0.8175 * L, 0.38 * L)
    FEMOL.utils.plot_ellipse_arc(c, a, b, -np.pi / 2, np.pi / 2)
    # Top 1
    p = FEMOL.domains.create_polynomial(0.25 * L, 0.76 * L, 0.625 * L, (0.71225 - 0.1645 / 2) * L, 0)
    x = np.linspace(0.25 * L, 0.625 * L, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Top 2
    p = FEMOL.domains.create_polynomial(0.625 * L, 0.71225*L - 0.1645*L / 2, 0.8175 * L, 0.67*L, 0)
    x = np.linspace(0.625*L, 0.8175*L, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Bot 1
    p = FEMOL.domains.create_polynomial(0.625 * L, 0.04775*L + 0.1645*L / 2, 0.8175 * L, 0.09*L, 0)
    x = np.linspace(0.625*L, 0.8175*L, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Bot 2
    p = FEMOL.domains.create_polynomial(0.25 * L, 0, 0.625 * L, (0.04775 + 0.1645 / 2) * L, 0)
    x = np.linspace(0.25 * L, 0.625 * L, 100)
    y = p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]
    ax.plot(x, y, color='k')
    # Soundhole
    FEMOL.utils.plot_circle(0.673 * L, 0.38 * L, 0.175 * L / 2)
    # Fix limits
    ax.set_xlim(0-0.01, L+0.01)


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

"""
Mesh data manipulation
"""

def angle(point, center):
    """Return the angle of a point on a circle at center"""
    return np.arctan2(point[0] - center[0], point[1] - center[1])


def distance(point, center):
    """Return the distance of a point to a center point"""
    return np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)


def points_area(points, subdivision=20):
    """
    Find the area containing the points
    :param points: a point scatter
    :param subdivision: subdivisions used by the algorithm
    :return: area
    """
    # convert to numpy
    points = np.array(points)
    center = (np.mean(points[:, 0]), np.mean(points[:, 1]))
    # Sort the points
    angles = [angle((xi, yi), center) for xi, yi in points[:, :2]]
    pts_s = points.copy()[np.argsort(angles), :]
    angles = np.sort(angles)
    # radial integration divided in 100
    distances = []
    step = pts_s.shape[0]//subdivision
    i = 0
    while True:
        try:
            distances.append(np.max([distance(point, center) for point in pts_s[i:i+step]]))
            i += step
        except ValueError:
            break

    area = 0
    T = (2*np.pi)/subdivision
    for i, _ in enumerate(distances[:-1]):
        hyp = min(distances[i], distances[i+1])
        b = max(distances[i], distances[i+1])
        h = hyp*np.sin(T)
        area += (b*h)/2

    return area


def interpolate_vector(old_vector, old_mesh, new_mesh, N_dof=6):
    """ Interpolate a displacement vector on a new mesh"""
    # empty array for the new vector
    new_vector = np.zeros(new_mesh.points.shape[0]*6)
    # Interpolate each degree of freedom
    for i in range(N_dof):
        # Interpolate at the new mesh points
        vi_linear = griddata(old_mesh.points[:, :2], old_vector[i::N_dof], new_mesh.points[:, :2], method='linear')
        vi_near = griddata(old_mesh.points[:, :2], old_vector[i::N_dof], new_mesh.points[:, :2], method='nearest')
        vi = vi_linear
        # Use the nearest value where the linear value is NaN (boundary)
        vi[np.isnan(vi_linear)] = vi_near[np.isnan(vi_linear)]
        new_vector[i::N_dof] = vi
    return new_vector


def interpolate_point_data(old_data, old_mesh, new_mesh):
    """ interpolate point data onto a new mesh"""
    # Interpolate at the new mesh points
    data_linear = griddata(old_mesh.points[:, :2], old_data, new_mesh.points[:, :2], method='linear')
    data_near = griddata(old_mesh.points[:, :2], old_data, new_mesh.points[:, :2], method='nearest')
    new_data = data_linear
    # Use the nearest value where the linear value is NaN (boundary)
    new_data[np.isnan(data_linear)] = data_near[np.isnan(data_linear)]
    return new_data


def analyse_mesh(meshfile, eigvalfile=None, mode=None):
    # Load the data
    mesh = FEMOL.mesh.load_vtk(meshfile)
    if eigvalfile is not None:
        eigvals = np.load(eigvalfile)
    else:
        eigvals = np.ones(len(mesh.cell_data.keys()))
    if mode is None:
        mode = '??'
    # Plot the moda of vibration
    fig, ax = plt.subplots()
    plt.sca(ax)
    mesh.plot.point_data('m1_Uz')
    ax.set_title(f'Mode of vibration {mode}')
    # Plot the optimization results
    fig, axs = plt.subplots(5, 4, figsize=(16, 20))
    for key, eig, ax in zip(mesh.cell_data.keys(), eigvals, axs.flatten()):
        plt.sca(ax)
        mesh.cell_to_point_data(key)
        mesh.plot.point_data(key, cmap='Greys')
        FEMOL.utils.guitar_outline2(L=1)
        title = f'TOM results {key} eigfreq {int(np.round(eig))} Hz'
        ax.set_title(title)


def plot_soundboard_deflexion(mesh):
    """ Function to plot the soundboard deflexion analysis"""

    def align_yaxis_np(ax1, ax2):
        """Align zeros of the two axes, zooming them out by same ratio"""
        axes = np.array([ax1, ax2])
        extrema = np.array([ax.get_ylim() for ax in axes])
        tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
        # Ensure that plots (intervals) are ordered bottom to top:
        if tops[0] > tops[1]:
            axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

        # How much would the plot overflow if we kept current zoom levels?
        tot_span = tops[1] + 1 - tops[0]

        extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
        extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
        [axes[i].set_ylim(*extrema[i]) for i in range(2)]

    x_points = mesh.points[np.isclose(mesh.points[:, 1], (0.76 / 2)), 0]
    idxs = np.argsort(x_points)
    displacement = mesh.point_data['Uz'][np.isclose(mesh.points[:, 1], (0.76 / 2))][idxs]
    x_points = x_points[idxs]

    f = interp1d(x_points, displacement, kind='quadratic')
    x1 = np.linspace(0, 0.6)
    x2 = np.linspace(0.78, 1)
    T1 = -np.degrees(np.arctan2(f(x1)[1:] - f(x1)[:-1], x1[1:] - x1[:-1]))
    T2 = -np.degrees(np.arctan2(f(x2)[1:] - f(x2)[:-1], x2[1:] - x2[:-1]))
    T_max = np.max(np.abs(np.hstack([T1, T2])))

    fig, ax1 = plt.subplots()
    ax1.plot(x1, f(x1) * 1000, color='#743F0B', label='deflexion')
    ax1.plot(x2, f(x2) * 1000, color='#743F0B')
    ax1.plot([0, 1], [0, 0], '--', color='0.6')
    ax1.set_ylim(-2, 5)
    ax1.plot([], [], linestyle='--', color='k', label='angle')
    ax1.set_xlabel('normalized distance across soundboard')

    ax2 = ax1.twinx()
    ax2.plot(x1[1:], T1, linestyle='--', color='k', label='angle')
    ax2.plot(x2[1:], T2, linestyle='--', color='k')

    ax1.legend()
    ax1.set_ylabel('Deflexion (mm)')
    ax2.set_ylabel('Normal angle (degrees)')
    ax1.grid('on')
    ax1.set_xlim(0, 1)
    align_yaxis_np(ax1, ax2)

    return T_max


def topology_info(mesh, h_min=0, scale=0.480):
    """ Function printing the topology info from a mesh"""
    Mf = flax_mass(mesh)
    Vcore = core_volume(mesh, h_min=h_min)
    Mpla = Vcore * 1.25 * 0.2
    print(f'pla mass: {np.around(Mpla, 1)} g')
    Mcarbon = carbon_mass(mesh, h_min=h_min)
    print(f'ratio brace/board {np.around((Mcarbon + Mpla) / Mf, 2)}')


def core_volume(mesh, h_min=0.002, scale=0.480):
    """ Function computing the volume of a core of a topology result"""
    h = mesh.cell_data['zc']['quad']
    h[h < h_min] = 0
    A = mesh.element_areas()['quad'] * (scale ** 2)
    print(f'core volume: {np.around(np.sum(h * A) * (100 ** 3), 2)} cm^3')
    return np.sum(h * A) * (100 ** 3)


def flax_mass(mesh, t=0.0025, scale=0.480):
    """ Function computing the flax mass of a board from a mesh"""
    A = mesh.element_areas()['quad'] * (scale ** 2)
    flax = FEMOL.materials.general_flax()
    V = np.sum(A * t)  # m3
    M = V * flax.rho
    return 1000 * M


def carbon_mass(mesh, t=0.00025, h_min=0, scale=0.480):
    """ Fonction computing the carbon mass from a topology result"""
    A = mesh.element_areas()['quad'] * (scale ** 2)
    h = mesh.cell_data['zc']['quad']
    flax = FEMOL.materials.general_carbon()
    V = np.sum(A[h > h_min] * t)  # m3
    M = V * flax.rho
    M *= 1000
    print(f'carbon mass: {np.around(M, 1)} g')
    return M


def add_top_brace_to_zc(mesh, zc, hc):
    """ Add a top brace to the core height cell data of a mesh"""
    domain = FEMOL.domains.top_brace(L=1)
    for i, element in enumerate(mesh.cells[mesh.contains[0]]):
        if np.array([domain(*coord[:2]) for coord in mesh.points[element]]).all():
            zc[mesh.contains[0]][i] = hc
    return zc

