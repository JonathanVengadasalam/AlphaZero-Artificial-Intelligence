# AlphaZero Artificial Intellignence
This project is an implementation of an artificial intelligence inspired by the [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0) "AlphaGo Zero cheat sheet" develloped by Google 

## Code :
1. The artificial intelligence (ai) is implemented in the class "AI" in the module [artificial_intelligence/ai.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/artificial_intelligence/ai.py). For a given position of a game state (called environnement), the ai gives the best next move. To found this next move, it use one of these 2 methods below and others hyperparameters (iteration, formula, selection...) :
   - Monte Carlo Tree Search (mcts) (function : "ai.montecarlo_treesearch"), it doesn't use neural network but use monte carlo method to build the research tree ([wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)).
   - Neural Network Tree Search (nnts) (function : "ai.neuralnetwork_treesearch"), I modified the classical mcts function so that it integrates neural network, the function use the network to evaluate the positions and build the research tree. The network gives to results :
     - the policy : the probabilities of each next move.
     - the value : the probability that this position is winning.

2. The games are in the folder [environnements](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/tree/master/environnements) (env). I implement the game Connect 4 in the module [envConnect.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/environnements/envConnect.py) and I use this game in this project to improve my neural network model. You can implement your one game like chess or draughts but be carefull that your class contains the function [clone, convertmove, domove, getmoves, value, update_y, x, y].

3. The module [manager/functions.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/manager/functions.py) gives to the user functions to collect data and results by playing several game ai versus ai (function : "selfplay"). The data is used to the machine learning and the results is used to test the network performance.

## Machine Learning :
The network model is builded with the module [neural_network_models/modelBuilder.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/neural_network_models/modelBuilder.py). I build a deep convolutional neural network with 8 residual layers and each convolution has 64 filters with a kernel size of 2.

I train the network in 2 steps. The main functions are in the [run.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/run.py).

1. The ai using mcts method self-play several times to get a dataset of x (about 750 000 positions) and y, then the model fit the data. This pretrain is necessary for the model to be more efficient for the next step when it will train by reinforcement. Find this step in the function "pretraining_for_connect_game".

2. Then, the ai use nnts method with the network to selplay. After getting data, the model fit it and get the new network. To test if the new network is better, the ai with the new network play 200 game against the ai with the old network. The new network is validated with an unilateral z-test.


