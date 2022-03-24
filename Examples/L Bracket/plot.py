import FEMOL
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def contour_stress_plot(mesh):
    """ Plot the stress distribution on the mesh as a contour plot"""
    # get the data
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.point_data['Sv']

    # create figure
    fig, ax = plt.subplots(figsize=(6,6))
    # plot the contour levels
    cont = plt.tricontour(x, y, z, 15, colors='b')
    # plot the bracket outline
    FEMOL.misc.plot_L_outline2()
    # first bar hiding the contour
    plt.bar(0.001, 0.1, 0.118 - 0.025, bottom=0.032 + 0.0005, color='1', align='edge', zorder=20)
    # Corner circle
    Rc =  0.025
    x = 0.118 - 0.0005
    y = 0.032+ 0.0005
    c = [x - Rc, y + Rc]
    cir = plt.Circle(c, Rc, color='1', zorder=20)
    ax.add_patch(cir)
    # Second bar hiding contours
    R = 0.152 - 0.135
    plt.bar(0.001, 0.135 - R - 0.001, 0.135 - R - 0.0015, bottom=c[1], color='1', align='edge', zorder=20)
    # circle on the top hole
    Rc =  0.025
    x = 0.118 + 0.001
    y = 0.032 + 0.001
    c = [x - Rc, y + Rc]
    cir = plt.Circle([0.135, 0.160], 0.009, color='1', zorder=20)
    ax.add_patch(cir)
    # remove axes lines
    ax.set_axis_off()
    