/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
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

#include "op_trajectory_generator_core.h"
#include "op_ros_helpers/op_ROSHelpers.h"


namespace TrajectoryGeneratorNS
{

TrajectoryGen::TrajectoryGen()
{
    bInitPos = false;  // 标志位，表示是否初始化车辆位置
    bNewCurrentPos = false;  // 标志位，表示是否有新的车辆当前位置信息
    bVehicleStatus = false;  // 标志位，表示是否有车辆状态信息
    bWayGlobalPath = false;  // 标志位，表示是否有全局路径信息

	ros::NodeHandle _nh;
	UpdatePlanningParams(_nh);

	// 获取坐标变换信息
	tf::StampedTransform transform;
	PlannerHNS::ROSHelpers::GetTransformFromTF("map", "world", transform);
	m_OriginPos.position.x  = transform.getOrigin().x();
	m_OriginPos.position.y  = transform.getOrigin().y();
	m_OriginPos.position.z  = transform.getOrigin().z();

    // 初始化ROS发布器
	pub_LocalTrajectories = nh.advertise<autoware_msgs::LaneArray>("local_trajectories", 1);
	pub_LocalTrajectoriesRviz = nh.advertise<visualization_msgs::MarkerArray>("local_trajectories_gen_rviz", 1);

    // 初始化ROS订阅器，订阅初始化位置、当前位置、车辆速度信息、全局路径信息等
	sub_initialpose = nh.subscribe("/initialpose", 1, &TrajectoryGen::callbackGetInitPose, this);
	sub_current_pose = nh.subscribe("/current_pose", 10, &TrajectoryGen::callbackGetCurrentPose, this);

	// 根据速度源参数选择相应的速度信息订阅器
	int bVelSource = 1;
	_nh.getParam("/op_trajectory_generator/velocitySource", bVelSource);
	if(bVelSource == 0)
		sub_robot_odom = nh.subscribe("/odom", 10,	&TrajectoryGen::callbackGetRobotOdom, this);
	else if(bVelSource == 1)
		sub_current_velocity = nh.subscribe("/current_velocity", 10, &TrajectoryGen::callbackGetVehicleStatus, this);
	else if(bVelSource == 2)
		sub_can_info = nh.subscribe("/can_info", 10, &TrajectoryGen::callbackGetCANInfo, this);
	// 订阅全局路径信息
	sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array", 1, &TrajectoryGen::callbackGetGlobalPlannerPath, this);
}

TrajectoryGen::~TrajectoryGen()
{
}

void TrajectoryGen::UpdatePlanningParams(ros::NodeHandle& _nh)
{
	_nh.getParam("/op_trajectory_generator/samplingTipMargin", m_PlanningParams.carTipMargin);
	_nh.getParam("/op_trajectory_generator/samplingOutMargin", m_PlanningParams.rollInMargin);
	_nh.getParam("/op_trajectory_generator/samplingSpeedFactor", m_PlanningParams.rollInSpeedFactor);
	_nh.getParam("/op_trajectory_generator/enableHeadingSmoothing", m_PlanningParams.enableHeadingSmoothing);

	_nh.getParam("/op_common_params/enableSwerving", m_PlanningParams.enableSwerving);
	if(m_PlanningParams.enableSwerving)
		m_PlanningParams.enableFollowing = true;
	else
		_nh.getParam("/op_common_params/enableFollowing", m_PlanningParams.enableFollowing);

	_nh.getParam("/op_common_params/enableTrafficLightBehavior", m_PlanningParams.enableTrafficLightBehavior);
	_nh.getParam("/op_common_params/enableStopSignBehavior", m_PlanningParams.enableStopSignBehavior);

	_nh.getParam("/op_common_params/maxVelocity", m_PlanningParams.maxSpeed);
	_nh.getParam("/op_common_params/minVelocity", m_PlanningParams.minSpeed);
	_nh.getParam("/op_common_params/maxLocalPlanDistance", m_PlanningParams.microPlanDistance);

	_nh.getParam("/op_common_params/pathDensity", m_PlanningParams.pathDensity);
	_nh.getParam("/op_common_params/rollOutDensity", m_PlanningParams.rollOutDensity);
	if(m_PlanningParams.enableSwerving)
		_nh.getParam("/op_common_params/rollOutsNumber", m_PlanningParams.rollOutNumber);
	else
		m_PlanningParams.rollOutNumber = 0;

	_nh.getParam("/op_common_params/horizonDistance", m_PlanningParams.horizonDistance);
	_nh.getParam("/op_common_params/minFollowingDistance", m_PlanningParams.minFollowingDistance);
	_nh.getParam("/op_common_params/minDistanceToAvoid", m_PlanningParams.minDistanceToAvoid);
	_nh.getParam("/op_common_params/maxDistanceToAvoid", m_PlanningParams.maxDistanceToAvoid);
	_nh.getParam("/op_common_params/speedProfileFactor", m_PlanningParams.speedProfileFactor);

	_nh.getParam("/op_common_params/smoothingDataWeight", m_PlanningParams.smoothingDataWeight);
	_nh.getParam("/op_common_params/smoothingSmoothWeight", m_PlanningParams.smoothingSmoothWeight);

	_nh.getParam("/op_common_params/horizontalSafetyDistance", m_PlanningParams.horizontalSafetyDistancel);
	_nh.getParam("/op_common_params/verticalSafetyDistance", m_PlanningParams.verticalSafetyDistance);

	_nh.getParam("/op_common_params/enableLaneChange", m_PlanningParams.enableLaneChange);

	_nh.getParam("/op_common_params/width", m_CarInfo.width);
	_nh.getParam("/op_common_params/length", m_CarInfo.length);
	_nh.getParam("/op_common_params/wheelBaseLength", m_CarInfo.wheel_base);
	_nh.getParam("/op_common_params/turningRadius", m_CarInfo.turning_radius);
	_nh.getParam("/op_common_params/maxSteerAngle", m_CarInfo.max_steer_angle);
	_nh.getParam("/op_common_params/maxAcceleration", m_CarInfo.max_acceleration);
	_nh.getParam("/op_common_params/maxDeceleration", m_CarInfo.max_deceleration);

	m_CarInfo.max_speed_forward = m_PlanningParams.maxSpeed;
	m_CarInfo.min_speed_forward = m_PlanningParams.minSpeed;

}

void TrajectoryGen::callbackGetInitPose(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
{
	if(!bInitPos)
	{
		m_InitPos = PlannerHNS::WayPoint(msg->pose.pose.position.x+m_OriginPos.position.x,
				msg->pose.pose.position.y+m_OriginPos.position.y,
				msg->pose.pose.position.z+m_OriginPos.position.z,
				tf::getYaw(msg->pose.pose.orientation));
		m_CurrentPos = m_InitPos;
		bInitPos = true;
	}
}

void TrajectoryGen::callbackGetCurrentPose(const geometry_msgs::PoseStampedConstPtr& msg)
{
	m_CurrentPos = PlannerHNS::WayPoint(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z, tf::getYaw(msg->pose.orientation));
	m_InitPos = m_CurrentPos;
	bNewCurrentPos = true;
	bInitPos = true;
}

void TrajectoryGen::callbackGetVehicleStatus(const geometry_msgs::TwistStampedConstPtr& msg)
{
	m_VehicleStatus.speed = msg->twist.linear.x;
	m_CurrentPos.v = m_VehicleStatus.speed;
	if(fabs(msg->twist.linear.x) > 0.25)
		m_VehicleStatus.steer = atan(m_CarInfo.wheel_base * msg->twist.angular.z/msg->twist.linear.x);
	UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
	bVehicleStatus = true;
}

void TrajectoryGen::callbackGetCANInfo(const autoware_can_msgs::CANInfoConstPtr &msg)
{
	m_VehicleStatus.speed = msg->speed/3.6;
	m_VehicleStatus.steer = msg->angle * m_CarInfo.max_steer_angle / m_CarInfo.max_steer_value;
	UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
	bVehicleStatus = true;
}

void TrajectoryGen::callbackGetRobotOdom(const nav_msgs::OdometryConstPtr& msg)
{
	m_VehicleStatus.speed = msg->twist.twist.linear.x;
	m_VehicleStatus.steer += atan(m_CarInfo.wheel_base * msg->twist.twist.angular.z/msg->twist.twist.linear.x);
	UtilityHNS::UtilityH::GetTickCount(m_VehicleStatus.tStamp);
	bVehicleStatus = true;
}

void TrajectoryGen::callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg)
{
	if(msg->lanes.size() > 0)
	{
		bool bOldGlobalPath = m_GlobalPaths.size() == msg->lanes.size();

		m_GlobalPaths.clear();

		for(unsigned int i = 0 ; i < msg->lanes.size(); i++)
		{
			PlannerHNS::ROSHelpers::ConvertFromAutowareLaneToLocalLane(msg->lanes.at(i), m_temp_path);

			PlannerHNS::PlanningHelpers::CalcAngleAndCost(m_temp_path);
			m_GlobalPaths.push_back(m_temp_path);

			if(bOldGlobalPath)
			{
				bOldGlobalPath = PlannerHNS::PlanningHelpers::CompareTrajectories(m_temp_path, m_GlobalPaths.at(i));
			}
		}

		if(!bOldGlobalPath)
		{
			bWayGlobalPath = true;
			std::cout << "Received New Global Path Generator ! " << std::endl;
		}
		else
		{
			m_GlobalPaths.clear();
		}
	}
}

// 这是一个局部路径规划的ROS节点，主要的功能是根据全局规划路径生成车辆的局部轨迹
void TrajectoryGen::MainLoop()
{
	ros::Rate loop_rate(100);

	PlannerHNS::WayPoint prevState, state_change;

	while (ros::ok())
	{
		ros::spinOnce();

		if(bInitPos && m_GlobalPaths.size()>0)
		{
			// 清空存储全局路径的各个段的容器
			m_GlobalPathSections.clear();
			// 遍历全局路径的每一段，提取车辆前方视野范围内的路径点，并存储在m_GlobalPathSections中
			for(unsigned int i = 0; i < m_GlobalPaths.size(); i++)
			{
				t_centerTrajectorySmoothed.clear();
				PlannerHNS::PlanningHelpers::ExtractPartFromPointToDistanceDirectionFast(m_GlobalPaths.at(i), m_CurrentPos, m_PlanningParams.horizonDistance ,
						m_PlanningParams.pathDensity ,t_centerTrajectorySmoothed);

				m_GlobalPathSections.push_back(t_centerTrajectorySmoothed);
			}
			// 定义一个用于调试的存储采样点的容器
			std::vector<PlannerHNS::WayPoint> sampledPoints_debug;
			// 调用局部轨迹生成函数，生成车辆的局部轨迹
			m_Planner.GenerateRunoffTrajectory(m_GlobalPathSections, m_CurrentPos,
								m_PlanningParams.enableLaneChange,
								m_VehicleStatus.speed,
								m_PlanningParams.microPlanDistance,
								m_PlanningParams.maxSpeed,
								m_PlanningParams.minSpeed,
								m_PlanningParams.carTipMargin,
								m_PlanningParams.rollInMargin,
								m_PlanningParams.rollInSpeedFactor,
								m_PlanningParams.pathDensity,
								m_PlanningParams.rollOutDensity,
								m_PlanningParams.rollOutNumber,
								m_PlanningParams.smoothingDataWeight,
								m_PlanningParams.smoothingSmoothWeight,
								m_PlanningParams.smoothingToleranceError,
								m_PlanningParams.speedProfileFactor,
								m_PlanningParams.enableHeadingSmoothing,
								-1 , -1,
								m_RollOuts, sampledPoints_debug);

			// 定义存储局部轨迹的消息
			autoware_msgs::LaneArray local_lanes;
			// 遍历生成的轨迹，将其转换为Autoware中的Lane消息，并添加到local_lanes中
			for(unsigned int i=0; i < m_RollOuts.size(); i++)
			{
				for(unsigned int j=0; j < m_RollOuts.at(i).size(); j++)
				{
					autoware_msgs::Lane lane;
					PlannerHNS::PlanningHelpers::PredictConstantTimeCostForTrajectory(m_RollOuts.at(i).at(j), m_CurrentPos, m_PlanningParams.minSpeed, m_PlanningParams.microPlanDistance);
					PlannerHNS::ROSHelpers::ConvertFromLocalLaneToAutowareLane(m_RollOuts.at(i).at(j), lane);
					lane.closest_object_distance = 0;
					lane.closest_object_velocity = 0;
					lane.cost = 0;
					lane.is_blocked = false;
					lane.lane_index = i;
					local_lanes.lanes.push_back(lane);
				}
			}
			// 发布局部轨迹消息
			pub_LocalTrajectories.publish(local_lanes);
		}
		else
			// 如果初始位置未初始化或者全局路径为空，则订阅全局路径
			sub_GlobalPlannerPaths = nh.subscribe("/lane_waypoints_array", 	1,		&TrajectoryGen::callbackGetGlobalPlannerPath, 	this);
		// 定义存储可视化轨迹的消息
		visualization_msgs::MarkerArray all_rollOuts;
		// 将生成的轨迹转换为可视化消息
		PlannerHNS::ROSHelpers::TrajectoriesToMarkers(m_RollOuts, all_rollOuts);
		// 发布可视化轨迹消息
		pub_LocalTrajectoriesRviz.publish(all_rollOuts);

		loop_rate.sleep();
	}
}

}
