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



`Transform`