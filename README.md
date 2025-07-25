# Tensegrity
Tensegrity-related code (using ROS Noetic)

## Usage
Assuming there is a ros (catkin_ws) directory:
```
$ pwd
<HOME>/catkin_ws/src
```

Clone this repository inside src:
```
cd <HOME>/catkin_ws/src/
git clone https://github.com/PRX-Kinodynamic/tensegrity.git
cd ..
catkin_make
```



## Tests
Running tests for all packages:
```
catkin_make               # Compile and generate msgs
catkin_make run_tests     # Run tests 
catkin_test_results       # Check for failures
```