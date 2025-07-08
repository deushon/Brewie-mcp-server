# MCP Functions

This is a list of functions that can be used in the ROS MCP Server.

## get_topics
- **Purpose**: Retrieves the list of available topics from the robot's ROS system.
- **Returns**: List of topics (List[Any])

## pub_twist
not relevant for AiNex, dellited
- **Purpose**: Sends movement commands to the robot by setting linear and angular velocities.
- **Parameters**:
  - `linear`: Linear velocity (List[Any])
  - `angular`: Angular velocity (List[Any])

## pub_twist_seq
not relevant for AiNex, dellited
- **Purpose**: Sends a sequence of movement commands to the robot, allowing for multi-step motion control.
- **Parameters**:
  - `linear`: List of linear velocities (List[Any])
  - `angular`: List of angular velocities (List[Any])
  - `duration`: List of durations for each step (List[Any])
 
## sub_image -> get_image
changed to auto open file in windows
- **Purpose**: Receive images from the robot's point of view or of the surrounding environment.
- **Parameters**:
  - `save_path`: By default, the image is saved to the ``Downloads`` folder.

## pub_jointstate
not relevant for AiNex, dellited
- **Purpose**: Publishes a custom JointState message to the `/joint_states` topic.
- **Parameters**:
  - `name`: List of joint names (list[str])
  - `position`: List of joint positions (list[float])
  - `velocity`: List of joint velocities (list[float])
  - `effort`: List of joint efforts (list[float])

## sub_jointstate
not relevant for AiNex, dellited
- **Purpose**: Subscribes to the `/joint_states` topic and returns the latest JointState message as a formatted JSON string.
- **Returns**: JointState message (str)

## make_step
New!
- **Purpose**: Moving AiNex by it's kinematic module.
- **Parameters**:
  - `x`: float(-1;1)
  - `y`: float(-1;1)
 
## run_action
New!
- **Purpose**: Launch pre-prepared actions in the AiNex application
- **Parameters**:
  - `action_name`: str.
 
## get_available_actions
- **Purpose**: Retrieves the list of available pre-prepared actions.
- **Returns**: List of action files from ActionGroups (List[Str])
