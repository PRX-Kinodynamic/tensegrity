# Tensegrity
Tensegrity-related code (using ROS Noetic)

## Dependencies:
GTSAM
```
git clone https://github.com/borglab/gtsam
cd gtsam
git checkout 4.2.0 # Don't use a newer one
mkdir build
# use ccmake .. for more options. installing in local dir to avoid permissions issues
cd build -DCMAKE_PREFIX_INSTALL=../install -DGTSAM_USE_SYSTEM_EIGEN=ON .. 
make -j8
make install
export GTSAM_DIR=$(pwd)/install/lib/cmake/GTSAM # <- add to .bashrc
```

TORCH
Download libtorch according to OS / Cuda: https://pytorch.org
```
cd <Path/to/Torch/zip>
unzip <torch.zip>
export Torch_DIR=$(pwd)/libtorch # <-- Add it to bashrc!

```


## Compile
Clone this repository inside src:
```
cd <HOME>/catkin_ws/src/
git clone https://github.com/PRX-Kinodynamic/tensegrity.git
cd ..
catkin_make
export TENSEGRITY_ROS=$(pwd)/ # <-- Add it to bashrc!

```

## Tests
Running tests for all packages:
```
catkin_make               # Compile and generate msgs
catkin_make run_tests     # Run tests 
catkin_test_results       # Check for failures
```