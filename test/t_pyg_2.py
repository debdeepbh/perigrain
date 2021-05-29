import pygmsh


P = [
        [0.0, 0.0],
        [1.0, -0.2],
        [1.1, 1.2],
        [0.1, 0.7],
        ]
with pygmsh.geo.Geometry() as geom:
    p1 = geom.add_polygon(
            P,
            mesh_size=0.1,
    )
    # mesh = geom.generate_mesh()
    mesh = geom.generate_mesh()
print(mesh)

# ll = []
# with pygmsh.geo.Geometry() as geom:
    # geom.add_polygon(
        # [
            # [0.0, 0.0],
            # [1.0, -0.2],
            # [1.1, 1.2],
            # [0.1, 0.7],
        # ],
        # mesh_size=0.1,
    # )
    # ll.append(geom)
    # print(geom.
    # mesh = geom.generate_mesh()
    # print(mesh)
# print(ll)
# print(ll[0].generate_mesh())
