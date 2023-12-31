cmake_minimum_required(VERSION 2.8.3)
project(twist_generator)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
        autoware_build_flags
        roscpp
        std_msgs
        geometry_msgs    
        autoware_msgs
        amathutils_lib
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES twist_generator
 CATKIN_DEPENDS 
        autoware_build_flags
        roscpp
        std_msgs
        geometry_msgs    
        autoware_msgs
        amathutils_lib
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)
# 

add_executable(vehicle_status_converter src/vehicle_status_converter.cpp src/vehicle_status_converter_node.cpp)
add_dependencies(vehicle_status_converter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vehicle_status_converter ${catkin_LIBRARIES})

#############
## Install ##
#############

## Install executables and/or libraries
install(TARGETS vehicle_status_converter
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## Install project namespaced headers
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
        PATTERN ".svn" EXCLUDE)

install(DIRECTORY include/twist_generator/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})


#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
        find_package(rostest REQUIRED)
        add_rostest_gtest(vehicle_status_converter-test test/test_vehicle_status_converter.test 
                        test/src/test_vehicle_status_converter.cpp
                        src/vehicle_status_converter.cpp)
        target_link_libraries(vehicle_status_converter-test ${catkin_LIBRARIES})
endif()