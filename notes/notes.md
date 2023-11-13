# Autonomous Driving Agent


## Physics Modeling

Three coordinates in Carla:
- world/map coordinates
- sensor coordinates
- vehicle coordinates


world coordinate or inertial coordinate system: righ-handed coordinate system with x-axis pointing east, y-axis pointing north, and z-axis pointing up.
- heading, pitch, roll all follows right-handed rule to define positive direction
  - heading: rotation around z-axis, positive direction is using the right hand thumb point to the positive z-axis, and the curling fingers point to the positive rotation direction
  - pitch: rotation around y-axis, positive direction is using the right hand thumb point to the positive y-axis, and the curling fingers point to the positive rotation direction
  - roll: rotation around x-axis, positive direction is using the right hand thumb point to the positive x-axis, and the curling fingers point to the positive rotation direction



## Vehicle Control

Consider a simple control example: given next waypoint and target velocity, and current transform of the ego vehicle, how to control the vehicle to reach the waypoint and speed?

### PID Controller

The simplest way is to use two PID controllers, one for steering angle and one for throttle/brake. The steering angle is calculated by the cross track error (CTE) and the target velocity is used to calculate the throttle/brake.

**Steering angle PID controller**: typically known as Lateral Control.

The cross track error is the distance between the vehicle and the reference trajectory. The reference trajectory is usually a polynomial function of the road centerline. 

`Transform`