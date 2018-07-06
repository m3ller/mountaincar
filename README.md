# mountaincar
Attempting to get the network to learn to win OpenAI's MountainCar game. 

Game premise: a car initially starts at the bottom of a valley and the goal is to get this car from the valley to the top of the mountain. The car does not have enough power to reach the top of the mountain directly, instead it needs to build momentum in the valley before attempting to travel to the top.

## Code Overview
Solve game with reinforcement learning. There are two neural networks that are being optimized at the same time - the policy and the value networks.

Policy network: determines the action to take given the current environment input. (eg. Given that the car is moving at velocity x, located at height y, should the car be pushed left, right, or be given no push at all?)

Value network: determines the value of an action with respect to the entire game. (eg. A seemingly "bad" action in the short-term may be a cruical and worthwhile action in the long-term.)


