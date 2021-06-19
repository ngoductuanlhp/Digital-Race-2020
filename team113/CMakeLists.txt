cmake_minimum_required(VERSION 2.8.3)
project(team113)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs
  DEPENDS system_lib
)

# include_directories(
#   include
#   ${catkin_INCLUDE_DIRS}
# )

# add_library(${PROJECT_NAME}
#   src/car.cpp
#   src/img_publisher.cpp
#   # src/log_data.cpp
# )

# add_executable(${PROJECT_NAME}_node src/main.cpp)
# add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
# target_link_libraries(
#     ${PROJECT_NAME}_node
#     ${PROJECT_NAME}
#     ${catkin_LIBRARIES}
#     ${OpenCV_LIBS}
#     ${Boost_LIBRARIES}
# )

# install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME}
#    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#  )

# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
#   PATTERN ".svn" EXCLUDE
# )
