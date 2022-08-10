rad = 1e-3;
lc = 0.5e-3;

SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthFactor = 0.1;
//+
Point(1) = {0*rad, 0, 0, lc};
//+
Point(2) = {3*rad, 0, 0, lc};
//+
Line(1) = {1, 2};
