# Additional scripts to generate plots etc

# Scripts

# Arrangements
* `arrangement.py` generates the 2D arrangement of circles with and without nodal circle correction 

# 2D

## Bulk
* `scripts/2d_bulk_disk.sh` generates a list of particles (location and radius) from a mesh, applies downward force (gravity).
* `scripts/2d_bulk_plus.sh` generates a list of particles (location and radius) from a mesh, applies downward force (gravity).
* `scripts/2d_bulk_plus_resume_test.sh` continues wall compression test after particles have settled (used: `meshsize = 1e-3/8`)

**Note:**
1. Setting `nl_bdry_only = 1` in the config files implies the contact detection is done only with the boundary nodes.

# 3D

## Collision
* `3d_collision_sphere.sh`: Elastic collision between two spheres (with and without damping)

**Note:**
1. Use `Mesh.CharacteristicLengthFactor = 0.5;` in `meshdata/3d_sphere_small.geo` for accurate collision (no rotation after collision)
1. Here, `normal_stiffness` has `delta^5` in the denominator and `R_c = delta/2.5`.

## Bulk
* `3d_bulk_sphere.sh` generates an arrangement of spheres within a cube and then lets them fall under gravity.


