//+
lc = 1e-3;
Point(1) = {-0, -0, 0, lc};
//+
SetFactory("OpenCASCADE");

//Mesh.CharacteristicLengthFactor = 1;
Mesh.CharacteristicLengthFactor = 0.25;

scaling = 10e-3;

Circle(1) = {-0, -0, 0, 1*scaling, 0, 2*Pi};
//+
Circle(2) = {-0, -0, 0, 0.9*scaling, 0, 2*Pi};

//+
Line Loop(1) = {1};
//+
Line Loop(2) = {2};
//+
Plane Surface(1) = {1, 2};
//+
Extrude {0, 0, 1*scaling} {
  Surface{1}; 
}

//+
Extrude {0, 0, -0.1*scaling} {
  Surface{1};
}
//+
Line Loop(11) = {2};
//+
Plane Surface(8) = {11};
//+
Extrude {0, 0, -0.1*scaling} {
  Surface{8}; 
}
