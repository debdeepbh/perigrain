//+
SetFactory("OpenCASCADE");

//x_min = -20e-3;
//y_min = -20e-3;
//z_min = -20e-3;
//x_max = 20e-3;
//y_max = 20e-3;
//z_max = 20e-3;

// clearance away from the wall: contact_rad*(1+epsilon)
clearance = (1e-3/(2.5))*1.001;
c = clearance/2;

x_min = -20e-3 + c;
y_min = -20e-3 + c;
z_min = -20e-3 + c;
x_max = 20e-3 - c;
y_max = 20e-3 - c;
z_max = 20e-3 - c;

//Mesh.CharacteristicLengthFactor = 2;
Mesh.CharacteristicLengthFactor = 1;
//Mesh.CharacteristicLengthFactor = 0.5;
//Mesh.CharacteristicLengthFactor = 0.25;

Point(1) = {x_min, y_min, z_min};
Point(2) = {x_max, y_min, z_min};
Point(3) = {x_max, y_max, z_min};
Point(4) = {x_min, y_max, z_min};
Point(5) = {x_min, y_min, z_max};
Point(6) = {x_max, y_min, z_max};
Point(7) = {x_max, y_max, z_max};
Point(8) = {x_min, y_max, z_max};

Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};
Line(5)  = {5, 6};
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};
//+
Curve Loop(1) = {5, 6, 7, 8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, -11, -2, 10};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {11, 7, -12, -3};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {8, -9, -4, 12};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {9, 5, -10, -1};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {1, 2, 3, 4};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {6, 5, 4, 1, 2, 3};
//+
Volume(1) = {1};
