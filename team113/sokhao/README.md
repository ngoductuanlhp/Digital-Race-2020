## Overview
This is source code to build an autonomous car on simulation env for FPT-Digital Race 
## Build source
Step 1: Build image team113
````sh
docker build -t team113 .
````

Step 2: Run container team113_container
````sh
docker run -it --rm --name=team113_container --gpus=all --network=host -v "$(pwd):/catkin_ws/src/team113" team113:latest bash
````

Step 3: In terminal running Docker container: Make ros package and run code
````sh
cd catkin_ws
catkin_make
roslaunch team113 team113.launch
````

Step 4: In host compute: Open new terminal and run Simulator 

Team name: team113\
IP: ws://127.0.0.1:9090

