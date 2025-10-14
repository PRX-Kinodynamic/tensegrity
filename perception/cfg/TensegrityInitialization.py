#!/usr/bin/env python
PACKAGE = "perception"

from dynamic_reconfigure.parameter_generator_catkin import *

gen = ParameterGenerator()

gen.add("x",    double_t,    0, "x value (pixels)",   0)
gen.add("y",    double_t,    0, "y value (pixels)",   0)
gen.add("z",    double_t,    0, "z value (meters)",   0)

gen.add("Save",   bool_t,   0, "Save to file",  False)

exit(gen.generate(PACKAGE, "TensegrityInitialization", "TensegrityInitialization"))