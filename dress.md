Write a finite state machine type code to perform dressing with the instructions below.

The whole objective of the code is that the robot has to dress the person approaching it when they have neutral emotion and while paying attention to the camera. 

For that to happen the following steps need to be taken in the order,
1. Action recogniser should recognise whether the person is approaching and extending the hand in the workspace of Franka. Once approached and extended the hand they can recede at anytime, in that case the dressing should stop.
2. engagement detector should look at the emotion of the person, the emotion should always be happy, neutral else the dressing should stop. 
3. Once the above critera are met, the pose estimator should take a reliable pose and generate a trajectory using the dmp and publish a trajectory to be sent to the Franka robot.
4. If at any point the pose changes (eg., the shoulder moves) a new trajectory to reach the new shoulder pose should be published, we are using delayed goal DMP so that should be easy. 
5. 