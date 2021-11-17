import gmsh
import meshio
import FEMOL
import numpy as np

# Create a circle of points
R = 1
C = (0, 0)
T = np.linspace(0, 2*np.pi, 36)
points = np.array([C[0] + R*np.cos(T), C[1] + R*np.sin(T)]).T

# Initialize gmsh
gmsh.initialize()
gmsh.model.add("c11")

# Add the points keep the ids
p_is = []
for point in points:
    i = gmsh.model.geo.addPoint(*point, 0)
    p_is.append(i)

# add the lines keep the ids
l_is = []
for i in range(len(p_is) - 1):
    l = gmsh.model.geo.addLine(p_is[i], p_is[i + 1])
    l_is.append(l)
l = gmsh.model.geo.addLine(p_is[-1], p_is[0])
l_is.append(l)

# Curve loop from line ids
cl = gmsh.model.geo.addCurveLoop(l_is)
# Plane surface from curve loop
pl = gmsh.model.geo.addPlaneSurface([cl])
# Something
gmsh.model.geo.synchronize()

# To generate quadrangles instead of triangles, we can simply add
gmsh.model.mesh.setRecombine(2, pl)

gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
gmsh.model.mesh.recombine()
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

gmsh.write("c11.vtk")
gmsh.finalize()

mesh = meshio.read('c11.vtk')

mesh = FEMOL.Mesh(mesh.points, mesh.cells_dict)
mesh.get_quality()

print('min quality : ', max(mesh.cell_data['quality']['quad']))
if 'triangle' in mesh.contains:
    print(mesh.cells['triangle'].shape[0], 'triangles')
else:
    print(0, 'triangles')

mesh.plot.quad_tris()
