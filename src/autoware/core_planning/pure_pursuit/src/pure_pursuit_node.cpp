/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ROS Includes
#include <ros/ros.h>

// User defined includes
#include <pure_pursuit/pure_pursuit_core.h>

/**
 * pure_pursuit 纯跟踪节点
 * 纯跟踪控制节点需要接收规划的路径点序列、车辆当前位姿、车辆当前速度、节点中参数配置 => 发布轮胎转角和速度控制指令
*/
int main(int argc, char** argv)
{
  ros::init(argc, argv, "pure_pursuit");
  waypoint_follower::PurePursuitNode ppn;
  ppn.run();

  return 0;
}
