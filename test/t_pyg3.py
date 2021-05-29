import pygmsh
geom = pygmsh.built_in.Geometry(
    characteristic_length_min=0.1,
    characteristic_length_max=0.1,
    )

def create_fin_surface(geom, h, w, l, cr, lcar=100):
    f = 0.5*w
    y = [-f,-f+cr, +f-cr, +f]
    z = [0.0, h-cr, h]
    f = 0.5 * cr
    x = [-f, f]
    points = []
    points.append(geom.add_point((x[0], y[0], z[0]), lcar=lcar))
    points.append(geom.add_point((x[0], y[0], z[1]), lcar=lcar))
    points.append(geom.add_point((x[0], y[1], z[1]), lcar=lcar))
    points.append(geom.add_point((x[0], y[1], z[2]), lcar=lcar))
    points.append(geom.add_point((x[0], y[2], z[2]), lcar=lcar))
    points.append(geom.add_point((x[0], y[2], z[1]), lcar=lcar))
    points.append(geom.add_point((x[0], y[3], z[1]), lcar=lcar))
    points.append(geom.add_point((x[0], y[3], z[0]), lcar=lcar))

    lines = []
    lines.append(geom.add_line(points[0], points[1]))
    lines.append(geom.add_circle_arc(points[1], points[2], points[3]))

    lines.append(geom.add_line(points[3], points[4]))
    lines.append(geom.add_circle_arc(points[4], points[5], points[6]))
    lines.append(geom.add_line(points[6], points[7]))
    lines.append(geom.add_line(points[7], points[0]))

    line_loop=geom.add_line_loop(lines)
    surface=geom.add_plane_surface(line_loop)
    vol = geom.extrude(surface, translation_axis=[l, 0, 0])
    return vol

h_fin = 25
w_fin = 10
l_fin = 100
x_fin = -0.5*l_fin
corner_radius = 1

surf = create_fin_surface(geom, h=h_fin, w=w_fin, l=l_fin, cr=corner_radius)
print(geom.get_code())
