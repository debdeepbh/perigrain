//+
radius = 1e-3;
lc = 0.5e-3;

//Point(1) = {0, -0, 0, lc};
//+
SetFactory("OpenCASCADE");
// mesh size factor = 1 => 171 nodes (v 3.6.0) 199 nodes (v 4.1.7)

//Mesh.CharacteristicLengthFactor = 3;
Mesh.CharacteristicLengthFactor = 1;
//Mesh.CharacteristicLengthFactor = 0.75;
//Mesh.CharacteristicLengthFactor = 0.5;
//Mesh.CharacteristicLengthFactor = 0.25;
//Sphere(1) = {0, 0, 0, radius, -Pi/2, Pi/2, 2*Pi};
Sphere(1) = {0, 0, 0, radius};
