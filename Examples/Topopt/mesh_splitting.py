mesh = FEMOL.mesh.load_vtk('soundboard_TOM_T31.vtk')
zc = mesh.point_data['zc']
good_nodes = np.arange(mesh.points.shape[0])[zc>0.005]
good_cells = [cell for cell in mesh.plot.all_tris if all(ci in good_nodes for ci in cell)]
cell_groups = []

while True:
    try:
        node_groups = []
        current_group = [good_cells[0]]
        good_cells.pop(0)
        # For each cell in the current_group
        for cell_1 in current_group:
            # If a cell in good cells shares two nodes with the current group add it
            for i, cell_2 in enumerate(good_cells):
                intersect = len(set(cell_1).intersection(cell_2))
                if intersect == 2:
                    current_group.append(cell_2)
                    # Remove it from all the good cells
                    _ = good_cells.pop(i)
        if len(current_group)>2:
            cell_groups.append(current_group)
    except IndexError:
        break
#good_cells = np.array([cell for cell in mesh.cells['quad'] if all(ci in good_nodes for ci in cell)])
all_mesh = []
fig, axs = plt.subplots(len(cell_groups)//2 + len(cell_groups)%2, 2, figsize=(8, 14))

for i, group in enumerate(cell_groups):
    plt.sca(axs.flatten()[i])
    test_mesh = FEMOL.mesh.Mesh(mesh.points, {'triangle':np.array(group)})
    test_mesh.point_data['zc'] = zc[np.unique(group)]
    test_mesh.plot.point_data('zc')
    FEMOL.utils.guitar_outline2(L=0.48)
    axs.flatten()[i].set_title(i)
    all_mesh.append(test_mesh)
    
for i in range(len(axs.flatten()) - len(cell_groups)):
    axs.flatten()[-(i+1)].set_axis_off()
    
for ci in np.array(all_mesh)[[1, 2, 6]]:
    ci.plot.point_data('zc')
FEMOL.utils.guitar_outline2(L=0.480)