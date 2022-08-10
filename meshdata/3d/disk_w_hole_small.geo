//+
lc = 1e-3;

z_low = 0.2*lc;

//Point(1) = {0, -0, z_low, lc};
//+
SetFactory("OpenCASCADE");
//Mesh.CharacteristicLengthFactor = 0.5;
Mesh.CharacteristicLengthFactor = 0.4;
Circle(1) = {0, -0, z_low, 1*lc, 0, 2*Pi};
//+
Circle(2) = {0, 0.5*lc, z_low, 0.4*lc, 0, 2*Pi};
//+
Line Loop(1) = {1};
//+
Line Loop(2) = {2};

//+
Plane Surface(1) = {1, 2};
//+

// Extrude only once to obtain correct boundary nodes in python
Extrude {0, 0, z_low*2} {
  Surface{1}; 
}
