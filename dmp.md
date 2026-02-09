To test dmp node, run the dmp node 

```
ros2 run colab_dress dmp_node
```
move the franka to home position


```
ros2 topic pub --once /dmp/arm_poses geometry_msgs/msg/PoseArray "{
  header: {
    frame_id: 'base'
  },
  poses: [
    {position: {x: 0.45, y: 0.045, z: 0.35}, orientation: {x: -0.7071, y: 0.7071, z: 0.0, w: 0.0}},
    {position: {x: 0.45, y: 0.145, z: 0.35}, orientation: {x: -0.7071, y: 0.7071, z: 0.0, w: 0.0}},
    {position: {x: 0.45, y: 0.345, z: 0.35}, orientation: {x: -0.7071, y: 0.7071, z: 0.0, w: 0.0}}
  ]
}"
```