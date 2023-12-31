cmake_minimum_required(VERSION 2.8.3)
project(freespace_planner)

find_package(autoware_build_flags REQUIRED)
find_package(
  catkin REQUIRED COMPONENTS
    astar_search
    autoware_msgs
    geometry_msgs
    nav_msgs
    roscpp
    tf
    visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS
  astar_search
  autoware_msgs
  geometry_msgs
  nav_msgs
  roscpp
  tf
  visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(
  astar_navi
  src/astar_navi/astar_navi.cpp
  src/astar_navi/astar_navi_node.cpp
)

target_link_libraries(
  astar_navi
  ${catkin_LIBRARIES}
)

add_dependencies(
  astar_navi
  ${catkin_EXPORTED_TARGETS}
)

install(
  TARGETS astar_navi
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
