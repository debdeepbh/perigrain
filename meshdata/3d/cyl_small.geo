//+
scaling = 10e-3;

lc = 1e-3;

Point(1) = {-0, -0, 0, lc};
//+
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthFactor = 0.1;

Circle(1) = {-0, -0, 0, 0.8*scaling, 0, 2*Pi};
//+
Line Loop(1) = {1};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, -0.1*scaling} {
  Surface{1}; 
}
