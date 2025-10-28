std_msgs/Header header

string message
uint8 barId # i.e. color
geometry_msgs/Point[] endcaps
uint8[] ids

geometry_msgs/Point[] bar_pts

std_msgs/MultiArrayDimension[] covDim
float64[] covariances
float64[] scores