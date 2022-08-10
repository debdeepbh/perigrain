radius = 3e-3;

SetFactory("OpenCASCADE");
//Mesh.CharacteristicLengthFactor = 1;
Mesh.CharacteristicLengthFactor = 0.5;

Circle(1) = {0, 0, 0, radius};
Circle(2) = {0, 0, 0, radius/2};

Curve Loop(1) = {1};
Curve Loop(2) = {2};
Plane Surface(1) = {1, 2};
