# graph-cpp
Granular media simulation with fracture using peridynamics

# Installation

```bash
# system packages
sudo apt-get install -y libhdf5-serial-dev gmsh

# eigen library
wget -c https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz 
mkdir lib
tar -xvf eigen-3.3.9.tar.gz -C lib/
rm eigen-3.3.9.tar.gz

# python packages
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

# generate the executable
make 
```

# Experiment setup and running
* Specify the experiment in `gen_setup.py`. This uses dictionaries:
  * `exp_dict.py`
  - `shape_dict.py`
  - `material_dict.py`

* Generate the setup: 
```
python3 gen_setup.py
```
This creates `setup.png` and `meshdata/all.h5` containing _all_ the information to start the simulation.

* Copy `meshdata/all.h5` to `data/hdf5/all.h5` using
```
make getfresh_py
```

* Specify the timesteps and simulation parameters in `config/main.conf`. Execute the compiled binary:
```
make ex2
```
(`ex3` for 3D)

* Generate plots from the data produced in `output/hdf5/` using
```
python3 plot_current.py <initial_index> <final_index>
```
or
```
make genplot
```

# Dependencies
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (Specify the location of the `Eigen` library path in `makefile` if using other versions)
* `gmsh` to generate mesh:
* To read `hdf5` files in linux using `h5dump`:
* Python packages
    * `matplotlib`, `numpy` as usual
    * mesh related python dependencies (`pip3 install`): `pygmsh`, `gmsh`, `meshio`
    * optimization using `gekko`
    * parallel processing using `multiprocessing`, `pathos`


# Code concept

* **Fracture:** The default value for `break_bonds` is `0` for each particle. Set it to `1` manually if you want breakage. Apart from  setting `P.break_bonds`, need to set `TL.enable_fracture = 1` to start breaking.

* **Boundary nodes:** While creating 3d surfaces, if you extrude a surface, the original surface is also considered to be a boundary, even though another extruding makes it a non-boundary. 

* The `Shape` class contains info regarding the mesh _before_ a mesh is generated. This includes a polygon, a list of line segments which intersect non-convex bonds, and if the mesh is to be loaded from a `.msh` file, the filename with path.

* The output of `genmesh()` can be considered a mesh (called here a `ReturnValue`) which is a set of information regarding the mesh. All the member variables of this `ReturnValue` (mesh) is conveyed (linearly) to generate a `Particle`.

* Multiple `Particle` classes: in python codes, we have `Particle` class to hold mostly mesh properties and `Particle_brief` to hold plotting related info. In C++, the Particle class is mainly for Newtonian motion.

* The `Experiment` class holds a particle array, a `Wall` or `Wall3d` (if `allow`ed) and a `Contact`

## Damping and friction
For vertical wall-particle collision:
* Damping: oscillation i the vertical direction. The oscillation is centered at zero and decays eventually.
* [ ] Friction: causes oscillation in the horizontal direction (and the oscillation does not decay)
The oscillation is NOT centered at zero
The more the coefficient of friction, the larger the oscillation

## Adding input options through the config file
* Add modifications to `run_timeloop()`
* Add extra class elements to corresponding classes such `TL`, `CN`, `Wall` etc
* Add variables to `ConfigVal` class 
* Read variables through `ConfigVal.readfiles()`
* Transfer the variable from ConfigVal to other classed using `ConfigVal.apply()`

## Simulated new scenarios other than the default one
* Create new alternative of `run_timeloop()` and call is conditionally using a config file option like `timeloop_type = run_timeloop` from `simulate2d.cpp`
* Alternatively, create new version of `simulate2d.cpp` and call that.

# Particle properties

| breakable 	| movable 	| stoppable |
|:--------:	|:-------:	|:---------:|
| bonds break	|timestep updates| feels contact forces |

* `movable=0` implies `breakable=0`
* `stoppable=0` implies `breakable=0`
* peridynamic force in not computed when `movable=0` or `stoppable=0`
* If you want the particle to move only via external force, set `0,1,0`

# Notes regarding mesh generated via `.geo` files
**Note:** In `.geo` files, do **NOT** include points that are _not_ used in generating mesh. 
Listing unused point `Point(p)` in the `.geo` file creates a `.msh` file that has `p` in it but `p` is not used as any element vertex. Therefore, `genmesh()` produces a node with _zero_ volume, that crashes the simulation by producing `-nan` values.

We have two choices so far:
 - Add the unused point as mesh point using:
```
Point{p} in Volume {1};
```
for 3D (`Sphere(1)` is a `Volume`), or 
```
Point{p} in Surface {1};
```
for 2D.
 - Do not include the point in the `.geo` file.

**Note:** Over-damping, i.e. `damping_ratio > 1` introduces extra oscillation. So, avoid it.


# Todo
- [ ] Apply fixed torque on wheel, 2D
	
- [ ] Saving bulk data from all timesteps into h5 file with given options to compute different quantities
	- [ ] Generate h5 files with options
	- [ ] Plot from h5 files with options
- [ ] Coffee 
	- [ ] [running] changed `rho_scale` to `0.7` from `0.6`
- [ ] Compression without gravity? (pre-computation of gravity takes too long)
* [ ] Compute the total kinetic energy for plotting. Does fracture (without damping and friction) dissipate the kinetic energy?
* [ ] [rewrite - speed boost attempt] For each particle `i` for each node `j` in particle-`i`, compute all forces on node `j` at once by looping over nodes of all particles, all walls and all self-contact nodes. This way, we can avoid going through the `ij` loops 3 times. Moreover, we can attempt to update the symmetric node as well. Is all this possible while using a parallelization?
* [ ] [Better to edit (a copy of) timeloop.cpp] How to simulate: particle first settles, then things are dropped on them
	- an override of total particles!
* [ ] To get a convex curve like the others test on `2d_bulk_small`
	- [?] Equal grain size (maybe the disparity in cnot computation for scaled grains is causing extra/less force)
	- [x] [didn't make any difference] Bigger grain size
	- [x] [didn't make any difference] Allow other values of friction (and damping)?
	- [ ] Compute the wall reaction force from the particles
	   - [ ] From the distance from the wall
	   - [ ] From averaged contact like Kawamoto
	- [?] Allow the right wall to move to maintain a constant pressure (Kawamoto)
	- [ ] Does rearrangement of grains really produce a dip in the force?
	- [ ] Compute ratio of sigma_1 and sigma_3 vs the difference 
	- [x] [Important. Seen evidence where increasing scaling without increasing delta leads to excessive jump upon impact.] While scaling particles, if delta remains constant, the neighbors need to be recomputed. On the other hands, if the neighbors are kept the same, delta needs to be modified and hence cnot and snot changes too.

* [ ] Export total runtime to file
* [ ] Create a plot class with `append` to plot more files and `superimpose` to plot within the same picture
* [x] combine functions like saving to file, printing etc in `timeloop.cpp` to reduce code repetition 
* [ ] Why is `run_timeloop_compact()` slower than `run_timeloop()`??

Rigo code: try a rectangle breaking due to its own weight, melting and cracking at the same time.
lsdemperi code: add the initial displacement, velocity, and acceleration as zero.
Pressure test: `2d_pressure`
    - [ ] Note down the total timesteps for the whole simulation
    - [ ] Copy latest wall file to help resume in `2d_pressure`
    - [ ] Possibly run with much more particles, what should be the wall stopping point (needs a lot of resuming, develop the setup to resume easily)
* [x] Find out why Ha-Bobaru $c_0$ constant does not work in `material_dict`
It works under the following conditions:
	* same rho, nu, E as the peridem material parameter with (delta=1e-3)
	* sodalime parameters with delta=2e-3
	
- [ ] Don't save particle force anymore, (since it can be derived from acc)
- [ ] Remove nonconvex bonds from 3D shapes
- [ ] Saving NbdArr or connectivity
- [ ] Multiple wall contact
- [ ] State based
- [ ] Speed up plot generation with R(?) or VTK or VisPy (GPU plotting) or mayavi
- [ ] GPU based optimization
- [ ] 3D particle arrangement generation: vispy?
  - [ ] Plotting the arrangement spheres
  - [ ] How to change meshsize of particle arrangement mesh without editing the .geo file?
  - [ ] Optimization for nodal sphere radii

- [x] Save connectivity data for resuming
	- [x] convert NbdArr to connectivity in C++
	- [x] Save connectivity to file
	- [x] Load connectivity on resume
	- [x] [unnecessary] Add config option to load connectivity on resume (default: yes, if breaking enabled)
	- [x] Python loading connectivity from saved h5 files
	- [x] Plotting bonds
	- [x] Computing damage
	- [x] Test
* [?] [`2d_bar_pull`] Elastic bar
	* Compute force on the boundary of the object, and divide by the number of nodes 
	[x] force gradient
* [x] Is it possible to reduce gravity after the particles have settled?
Observation: as soon as the gravity is reduced, the particles bounce back up due to spring force
    * [x] (Works! to keep the particles static) Attempt: reduce gravity and increase rho such that gravitational force (`mg`) is the same i.e, set `rho_scale=G/g` where `g` is the new gravity, `G` is the old gravity
- [?] [basic] Self-contact
- [?] (did it, but didn't test) Friction debug in 3d wall as well, 
- [x] Parallelize NArr and boundary node computation in python
- [x] resume from current wall status
- [x] Reading moving wall in python while plotting
- [x] Wall can have velocity and hence can affect the damping and friction with the wall
- [x] Formalize the wall velocities and conventions
- [x] Use nonlocal boundary nodes only as input from config (useful for 2d nofracture)
- [x] friction modification for particle-to-particle contact
- [x] [other walls] Moving wall in C++
- [x] Object movable and breakable
- [x] Check of ball slides over meshed plank
- [x] Does fixing the friction change the graphs of wall contact?
- [ ] Organizational:
	- [x] `allow_damping` and `allow_friction` is read from the config file. No need to include that in the `all.h5` anymore.
	- [ ] Break `all.hd5` into two parts: one with all the original info for each particle, another with initial data, which can be replaced while resuming
	- The advantage of saving the original mesh info (including `bdry_edges`) is that we can draw edges at each time point, in case there is no breakage. Or, in case of breakage, when the bond associated with the edge is broken, we omit plotting that bond. This will produce a much better picture in the end result.
	- [ ] Install script for python package `pygmsh` and `gmsh` using `pip3` (which installs in the local directory?)
	- [ ] Merge the branch of `graph-cpp` named `par` to `main`
	- [x] Create directories `mperidem/data/hdf5` and `mperidem/data/csv`

- [x] Compute the total contact force on wall boundaries, averaged(?)
- [x] Move `timesteps`, `dt`, and `modulo` to a plaintext file and read
- [x] 3D wall interaction implementation
- [x] Implement neighborhood computation based on _given_ contact ratio as input
[x] [Will _remember_ to keep .geo files free of unused points] (Keeping the `Point(1) = {0, 0, 0, lc}` in the file `mperidem/geo/3d_sphere_small.geo` produces an extra mesh element. Removing this point produces correct number of mesh elements `171` instead of `172`) Investigate why extra point (center) in `.geo` file messed up the whole simulation. Is the volume begin computed fairly?
	 Observation: `octave` library `msh3m_gmsh()` ignores this point while creating a mesh. So, octave produces `171` elements every time, whether or not `Point(1)` is included.
- [x] 3D:
    - [x] Material properties for 3d
    - [x] 3D reading hdf5 in C++
    - [x] 3D reading hdf5 in python and plotting
    - [x] Plot exp with CurrPos instead of Pos?
    - [x] Plot setup experiment in 3d
	- [x] Wall3d class saving to file
	- [x] Plot wall
	- [x] Plot bonds in 3d
	- [x] Plot bdry edge
    - [x] Save setup png in 3d
    - [x] Aspect ratio of z axis using matplotlib
    - [x] Reading .msh file
    - [x] Extract tetrahedron
    - [x] Compute the volume of tetrahedron
    - [x] boundary edges? are now triangles
    - [?] [Works only if] extract boundary nodes: the extruded intermediated layers get considered as boundary nodes (e.g. disk_w_hole) 
    - [x] Plot mesh with matplotlib in 3d
- [x] [deleting the line with `Point(1)` in `.geo` file solved it] Huge difference in `total internal force` at timestep `397` from `396`. Check two parts of the force: `peridynamic` and `contact force`
- [x] Use `load_setup` classes in `hdf_plot`
- [?] [Seems to work] Automatic boundary in 2D when wall is present
- [x] Python-exported files do not produce accurate results in C++, but matlab-generated files do. What is going on?
- [x] Check that output of python code is read by C++ properly (it probably doesn't)
- [x] Python - extracting a list boundary nodes in mesh, generate nodes that are `R_c`-distance away from the boundary. When to do it? The particles are agnostic about `R_c`.
- [x] Python - saving to hdf5
- [x] Plot in python from Hdf5
- [x] Contact based on boundary nodes, and the same with the wall
- [x] (ans: minimal) Pass functions by pointer to class instead of the class, to speed up (check if true)
- [x] Write matrices to binary format in matlab and read in c++ in binary to reduce size.
- [x] Resuming ability

Preserving row vs column distinction of a vector comes with the extra baggage that quantities like position of the nodes have to be vector of vectors of Vector2d.

That means, the position of a node will be a 1-dimensional vector of Vector2d, instead of just a Vector2d. In that case, the position of the i-th node will be called using `Particle.position[i][0]` instead of `Particle.position[i]`


# Choosing the right datatype for storing vectors

The goal is to store vector-valued information about the nodes in a graph-like datatype. A graph-like datatype is like a matrix but with variable row size. Example from Matlab would be a `cell`.
We would like apply vector operators on the elements of the graph-like datatype.

* To store the vector-valued information of each node, I tried
1. `vector<double>` of size 2: the vector operations have to be defined through operator overloading individually.
1. `Vector2d` from `Eigen` library: cannot do coordinate-wise multiplication of vectors using `*`
1. `Array<double, 1, 2>`: the current choice, has all the features that I need.

* To store the vector-valued information of _all_ nodes:
(let `V2` be the datatype for a single vector)
1. `std::vector<V2>`: element-wise multiplication are not automatic. Need to define individually.
1. `std::valarray<V2>`: element-wise multiplication are already defined. However, a few issues:
  - Sum of two `valarray<double>` is **NOT** `valarray<double>`! You need to typecast it or pre-define a variable as `valarray<double>` and store the sum in it.
  - Cannot multiply a `valarray<double>` by 2. You need to multiply by `2.0`. (i.e. multiplication of `valarray<double>` by `int` is not allowed)
  * Cannot multiply `double` with `valarray<valarray<double>>` without overloading. Same as `std::vector` though.
  * Cannot erase elements like `.erase()` with `std::vector`
  * Apparently, nobody uses it anymore and is mostly substituted by `std::vector`

[Features](https://en.cppreference.com/w/cpp/numeric/valarray)
  * Can do assignments like `data[data > 5] = -1` (Make sure to use `5.0` when data is `valarray<double>`)
  * Can apply function to each element of the valarray:
```
    std::valarray<int> v = {1,2,3,4,5,6,7,8,9,10};
    v = v.apply([](int n)->int {
                    return std::round(std::tgamma(n+1));
                });
```

# Updating the neighborhood array:

* As bonds break, elements are to be removed from the neighborhood array. With datatype `vector`, it can be done dynamically using `NbdArr.erase()`

# Points to remember
* Make sure the NbdArr loaded in C++ contains serials that start from 1, so we need to subtract 1 to make it compatible 
* I noticed a potentially catastrophic issue with floating point arithmetic: 
```
((p_y - p_x) + (u_y - u_x)).norm() 	!=  ((p_y + u_y) - (p_x + u_x)).norm()
```
even though
```
((p_y - p_x) + (u_y - u_x)) 	==  ((p_y + u_y) - (p_x + u_x))
```
Here, `norm()` is norm function from the `Eigen` library.
It seems that the difference in the norm is about `3e-19`, which causes trouble when we multiply the difference by a large number like `cnot = 1e19`.
The first expression (i.e. sum of difference) seems to be more accurate for small differences, i.e. when `u_y ~ u_x`.
Is it NOT due to an issue with `Eigen` norm. I implemented my own `norm()` function and computed all the norms in the same way, and I still get the aforementioned discrepancy.

Example:
```
double u =  5.7;
double p_x = 0.1;
double a = (p_x + u) - u;
std::cout << "diff: " << p_x - a << std::endl;
```
produces an error of about `2e-16`, which is pretty large compared to the machine-epsilon of `double`.

**Solution:** Use `long double`. The datatype `double` is accurate up to `16` significant digits. So, numbers larger than 16 digits are read incorrectly.
[Here](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) is an interesting explanation where the provide example of something like this happening.

A [post](https://scicomp.stackexchange.com/questions/26370/improve-numeric-stability-of-subtraction-in-c) explains it very well in comparison to Matlab's trick to avoid such error.

## Tests

**Setup:** timesteps = 1800, modulo = 100, sodalime_prenotch crack test

1. Computing `xi_norm` a-priori and calling `xi_norm[i][j]` vs computing `xi_norm[i][j].norm()` during runtime:

**Ans: Pre-computing norm is better since re-computation of many of the similar norms are avoided.**


| pre-computing | compute norm at runtime | 
|:--------------|:------------------------|
| 12 s          | 14 s                    | 


1. Computing the peridynamic force using `CurrPos` vs `Pos` and `xi`:

**Ans: using `CurrPos` is slightly faster.** But round-off error makes it unusable.

| Using `CurrPos` | Using `disp` and `xi` | 
|:----------------|:----------------------|
| 11 s            | 12 s                  | 


1. Bond breaking withing the peridynamic force computation VS separately

**Ans: computing within the peridynamic force computation function is slightly faster, since we do not need to travel though one single for loop.**

| Within `i` loop of peridynamic force computation | Via  separate function | 
|:-------------------------------------------------|:-----------------------|
| 11 s                                             | 12 s                   | 

1. `openmp` for `for` loops:

**Ans: using `openmp` is significantly faster than serial implementation.**

In `get_peridynamic_force()`, basic for parallelization over the outer loop:

| Serial | Parallel over the outer loop in `i` | 
|:-------|:------------------------------------|
| 12 s   | 8 s                                 | 

Using `schedule(dynamic)` increases the runtime by 1s.

1. `parallel for` in peridynamic force computation VS per particle

**Per-particle parallelization only (i.e. no nested parallelization) is better when there are many particles**

**For a small number of particles, per-particle parallelization is very slow, since the nested for loops cannot allow parallelization, or takes too long for the overhead.**

I cannot do nested loops efficiently. Maybe there is a room for improvement by doing nested parallelization.


| total particles | particle loop is parallel | particle node `i` is parallel | 
|:----------------|:--------------------------|:------------------------------|
| 1 (#i= 14000)   | 24 s                      | 12 s                          | 
| 53 (#i=51)      | 17 s                      | 27 s                          | 


1. `static` vs `dynamic` scheduling in parallel particle loop

total_particles_univ = 53; nnodes = 51; timesteps = 50000; modulo=500
no contact present
No nested parallelization

| dynamic | static | 
|:--------|:-------|
| 15 s    | 17 s   | 


1. Parallel for different areas of the code

total_particles_univ = 53; nnodes = 51; time steps = 50000; modulo=500
contact, friction, damping present
No nested parallelization

** peridynamic + contact force computation per particle in parallel (with dynamic scheduling) is the fastest, not with parts, provided no more segfault detected **

**Observation:** best performance when
 * Single outer loop exits
 * Higher the load per loop the better
 * (not sure why, try both) dynamic for variable load per iteration
 * No segfault: other threads are not trying to write or when on thread is reading
 * `ccls` is not running (and thus wasting processing power)
**

| Parallel part                               | time      |
| :----------------------------------         | :-------- |
| Initial time update                         | 48 s      |
| Save file per particle                      | 37 s      |
| Peridynamic force computation only          | 48 s      |
| Last time updates only (v, a)               | 50 s      |
| Contact force update only                   | 24 s      |
| Contact force + last time updates           | segfault  |
| Contact force + peridynamic force           | 20 s, segfault      |
| Contact force + peridynamic force (dynamic) | 17 s      |
| No parallelization at all                   | 33 s      |

Possible explanation (check): The segfault is due to the attempt to use the velocity while computing the damping force, and at the same time writing the velocity value by another thread.


1. 
4011 particles, xeon processor 
timesteps = 10000; modulo = 100
runtime = 6980 s

1.

4011 particles, xeon processor, independent parallelizations over
 * CurrPos, disp
 * peridynamic + contact + wall_contact
 * force, acc, vel update
 * Write to disk
Writing only CurrPos, and force.norm() to file
timesteps: 50000, modulo=200; counter: 250
No bond breaking
Runtime: 31791s = 8.83 h

1. Computation of contact forces using boundary nodes only VS all nodes

total_particles_univ = 53; nnodes = 51; time steps = 50000; modulo=500
contact, friction, damping present


| Boundary nodes only | All nodes | 
|:--------------------|:----------|
| 40 s                | 49 s      | 

1. Compute total boundary force by value vs by reference (both for boundary nodes)

**Minimal benefit to move from by-value to by-reference, (and avoiding initializing a vector<Matrix<double, 1, dim>> by 0**

| By value   | By reference   |
| :--------- | :------------- |
| 42 s, 41 s | 39 s, 40 s     |


1. Output data size: csv VS hdf5

** HUGE advantage in using hdf5 format instead of csv**

53 particles, 51 nodes, modulo = 200, counter = 50

| csv                      | hdf5  | 
|:-------------------------|:------|
| 12 x3 = 36 mb (expected) | 15 mb | 


1. 3D simulation

Total particles: 506
Number of nodes on each particle (`3d_sphere_small`): 171
Timesteps = 10000
modulo = 100
dt = 0.02/1e5

Parallel in Xeon processor

runtime = 714s ~ 12 m
plotting time = 203s ~ 3 m


| nodes	    | particles | timesteps | modulo | runtime      |  plotting time |
|:----------|:----------|:----------|:-------|:-------------|:---------------|
| 177       | 506       | 10000     | 100    | 714s ~ 12m   | 203s ~ 3m      |
| 199       | 502       | 10000     | 100    | 973s ~ 16m   | 218s ~ 3.6m    |
| 177       | 506       | 20000     | 200    | 2154s ~ 36m  | 200s ~ 3m      |
| 199       | 502       | 40000     | 400    | 6716s ~ 1.8h | 216s ~ 3.6m    |

1. 3D simulation

Total particles: 1204 (without trimming the smaller ones)
timesteps = 40000
modulo = 400
dt = 0.2/1e5
**Failed: Particles blow up after 13-14 modulo**

runtime = 7 hrs
plotting time = 9 mins

1. 2D simulation: 2d_bulk with plus0.2 (meshsize = 1/8)

| nodes	    | particles | timesteps          | modulo | runtime      |  plotting time |
|:----------|:----------|:-------------------|:-------|:-------------|:---------------|
| 135       | 1490      |10000               | 100    | 2919s ~ 48m  | 258s ~ 4.3m    |
| 135       | 1490      |10000 (10000-20000) | 100    | 3399s ~ 56m  | 246s ~ 4.3m    |
| 135       | 1490      |20000 (20000-40000) | 100    | 8075 ~ 2.24h | 502s ~ 8.3m    |
| 135       | 1490      |20000 (40000-60000) | 100    | 8214 ~ 2.28h | 514s ~ 8.5m    |
| 135       | 1490      |40000 (60000-100000)| 100    | 16684 ~ 4.63h| 960s ~ 16m     |
| 135       | 1490      |50000(100000-150000)| 100    | 20796 ~ 5.7h | 1446s~ 24m     |
| :--- :|
|  --- 	 |		 |150000            |  100  |	16h  |		1h 	|
| 135       | 1490      |50000(150000-200000)| 100    |         5.9h |        27m     |
| 135       | 1490      |     160000         | 800    |         17 h |       8.5m     |


1. 2D simulation: 2d_bulk with square `small_disk(steps=4)` (meshsize = 1/8)

Wall starting height for compression: 0.01 = 10e-3

| nodes	    | particles | timesteps          | modulo | runtime      |  plotting time |
|:----------|:----------|:-------------------|:-------|:-------------|:---------------|
| 118       | 1490      |     160000         | 800    | 53629s ~14 h |       3.8m     |
| 118       | 1490      |      80000         | 200    | 29666s ~11.6h|       x.xm     |


1. 2D simulation: 2d_bulk with `perturbed_disk(steps=16, seed=1)` (meshsize = 1/5)

Wall starting height for compression: x.xx = xxe-x

| nodes	    | particles | timesteps          | modulo | runtime      |  plotting time |
|:----------|:----------|:-------------------|:-------|:-------------|:---------------|
| 142       | 1490      |     80000          | 800    | 38128s ~10.5h|       x.xm     |
| 142       | 1490      |     80000          | 400    | xxxxxs ~xx.xh|       x.xm     |

1. 2d simulation: 2d_bulk_diffmesh_nogravity (meshsize  = 1/5)
without fracture (without saving connectivity): 1800s
