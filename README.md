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
cd build
make -j8 -DCMAKE_PREFIX_INSTALL=../install -DGTSAM_USE_SYSTEM_EIGEN=ON .. 
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
## Prerequisites
Install torch:
```
cd PATH/TO/LIBTORCH/INSTALL # cd into where to install libtorch
wget https://download.pytorch.org/libtorch/cu128/libtorch-VERSION.zip # Go to https://pytorch.org/ to get the latest libtorch version (c++)
unzip libtorch-*
echo "export Torch_DIR=/PATH/TO/LIBTORCH/INSTALL/libtorch/" >> ~/.bashrc # Add env variable to bashrc. Might need to source

```

## Tests
Running tests for all packages:
```
catkin_make               # Compile and generate msgs
catkin_make run_tests     # Run tests 
catkin_test_results       # Check for failures
```

##Running State Estimation
In separate tabs:
```
roscore
rosrun perception image_publisher.py
roslaunch interface camera_info_publisher.launch camera_info_file:=/common/home/pm708/Desktop/catkin_edgar/src/tensegrity/estimation/config/camera_info.yaml
```
