x_min = -1;
x_max = 1;
//+

Point(1) =  {x_min, x_min, x_min};
Point(2) =  {x_max, x_min, x_min};
Point(3) =  {x_max, x_max, x_min};
Point(4) =  {x_min, x_max, x_min};
Point(5) =  {x_min, x_min, x_max};
Point(6) =  {x_max, x_min, x_max};
Point(7) =  {x_max, x_max, x_max};
Point(8) =  {x_min, x_max, x_max};

//+
//Surfaces

// [Z[0],Z[1],Z[2],Z[3]],
// [Z[4],Z[5],Z[6],Z[7]], 
// [Z[0],Z[1],Z[5],Z[4]], 
// [Z[2],Z[3],Z[7],Z[6]], 
// [Z[1],Z[2],Z[6],Z[5]],
// [Z[4],Z[7],Z[3],Z[0]]

//            [Z[0], Z[1]],
//            [Z[1], Z[2]],
//            [Z[2], Z[3]],
//            [Z[3], Z[0]],
//
//            [Z[4], Z[5]],
//            [Z[5], Z[6]],
//            [Z[6], Z[7]],
//            [Z[7], Z[4]],
//
//            [Z[0], Z[4]],
//            [Z[1], Z[5]],
//            [Z[2], Z[6]],
//            [Z[3], Z[7]]
//

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) =  {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

//+
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Curve Loop(3) = {4, 9, -8, -12};
Curve Loop(4) = {12, -7, -11, 3};
Curve Loop(5) = {11, -6, -10, 2};
Curve Loop(6) = {10, -5, -9, 1};
//+
//Plane Surface(1) = {1};
//Plane Surface(2) = {2};
//Plane Surface(3) = {3};
//Plane Surface(4) = {4};
//Plane Surface(5) = {5};
//Plane Surface(6) = {6};
//+
Extrude {0, 0, 1} {
  Curve{7}; Curve{6}; Curve{5}; Curve{8}; 
}
//+
Extrude {0, 0, -1} {
  Curve{3}; Curve{4}; Curve{1}; Curve{2}; 
}
//+
Extrude {1, 0, 0} {
  Curve{2}; Curve{11}; Curve{6}; Curve{10}; 
}
//+
Extrude {-1, 0, 0} {
  Curve{4}; Curve{12}; Curve{8}; Curve{9}; 
}
//+
Extrude {0, 1, 0} {
  Curve{12}; Curve{3}; Curve{11}; Curve{7}; 
}
//+
Extrude {0, -1, 0} {
  Curve{10}; Curve{1}; Curve{9}; Curve{5}; 
}
//+
Curve Loop(7) = {13, 25, 21, 17};
//+
Plane Surface(109) = {7};
//+
Curve Loop(8) = {101, 105, -93, -97};
//+
Plane Surface(110) = {8};
//+
Curve Loop(9) = {37, 41, 29, 33};
//+
Plane Surface(111) = {9};
//+
Curve Loop(10) = {81, 77, -89, -85};
//+
Plane Surface(112) = {10};
//+
Curve Loop(11) = {61, 73, -69, -65};
//+
Plane Surface(113) = {11};
//+
Curve Loop(12) = {53, -49, -45, 57};
//+
Plane Surface(114) = {12};
//+
Surface Loop(1) = {32, 84, 80, 68, 72, 28, 24, 108, 96, 60, 56, 20, 16, 92, 112, 88, 52, 114, 48, 44, 111, 40, 100, 110, 104, 76, 113, 64, 36, 109};
//+
Volume(1) = {1};
