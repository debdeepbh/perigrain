import pygmsh

P = [
        [0.0, 0.0],
        [1.0, -0.2],
        [1.1, 1.2],
        [0.1, 0.7],
    ]
# with pygmsh.geo.Geometry() as geom:
geom=pygmsh.geo.Geometry()
p1 = geom.add_polygon(
        P,
        mesh_size=0.1,
)
s1 = geom.add_physical(p1.surface, 'surface1')
# print(geom.__dict__.keys())
# print(poly.__dict__.keys())
# geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)
mesh = geom.generate_mesh()
print(mesh)
