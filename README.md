# AlphaZero Artificial Intellignence
An implementation of an artificial intelligence inspired by the "AlphaGo Zero cheat sheet" develloped by Google.

## About The Project
This project is an implementation of an algorythme for playing 2 players board games like tic tac toe, connect4, gomoku. Inspired by the Google [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0), the algorythme uses deep neural network trained by self-playing.

## Getting Started
### Prerequisites
* numpy `py -m pip install numpy`
* tensorflow `pip install --upgrade tensorflow`

### Usage
1. `from functions import *`
2. unzip the mnist dataset in the folder data `unzip()`
3. load dataset `x_notscaled, x, y = load_data()`
4. plot the different unsupevized functions and get the results
   ```sh
   pca(x, y, x_notscaled)
   projected = tsne(x, y, x_notscaled)
   km_clustering(x, y)
   ac_clustering(x, y, projected)
   ```

To start training a model for Connect game : `python run.main()`

## Code description :
![ai/manager/environnement relations](https://github.com/JonathanVengadasalam/Artificial-Intelligence/blob/master/images/ai%20manager%20env%20relatiship.png)
1. The artificial intelligence (ai) is implemented to play 2 player games played each turn. It is coded in the class "AI" in the module [artificial_intelligence/ai.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/artificial_intelligence/ai.py). For a given position of a game state (called environnement), the ai gives the best next move. To found this next move, it use one of these 2 methods below and others hyperparameters (iteration, formula, selection...) :
   - Monte Carlo Tree Search (mcts) (function : `ai.montecarlo_treesearch`), it doesn't use neural network but use monte carlo method to build the research tree ([wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)).
   - Neural Network Tree Search (nnts) (function : `ai.neuralnetwork_treesearch`), I modified the classical mcts function so that it integrates neural network, the function use the network to evaluate the positions and build the research tree. The network gives to results :
     - the policy : the probabilities of each next move.
     - the value : the probability that this position is winning.

2. The games are in the folder [environnements](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/tree/master/environnements) (env). I have implemented the game Connect 4 in the module [envConnect.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/environnements/envConnect.py) and I use this game in this project to improve my neural network model. You can implement your one game like chess or draughts but be carefull that your class contains the function [clone, convertmove, domove, getmoves, value, x, y].

3. The module [manager/functions.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/manager/functions.py) gives to the user functions to collect data and results by playing several game ai versus ai (function : `run.selfplay`). The data is used to the machine learning and the results is used to test the network performance.

## Machine Learning :
![model/user/manager relations](https://github.com/JonathanVengadasalam/Artificial-Intelligence/blob/master/images/model%20user%20manager%20relationship.png)

The network model is builded with the module [neural_network_models/modelBuilder.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/neural_network_models/modelBuilder.py). I build a deep convolutional neural network with 8 residual layers and each convolution has 128 filters with a kernel size of 3. The model try to predict the policy and the value from the game actual and 5 previous positions and the player who just moved.

I train the network in 2 steps. The main functions are in the [run.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/run.py).

1. The ai using mcts method self-play several times to get a dataset of x (about 750 000 positions) and y, then the model fit the data. This pretrain is necessary for the model to be more efficient for the next step when it will train by reinforcement. Find this step in the function `run.pretraining_for_connect_game`.

2. Then, the ai self-play 4000 times by using neural-network-tree-search method and stochastic method to select move. After getting data, the model fit it and get the new network (function : `run.selftraining_for_connect`). To test if the new network is better then old, the ai with the new network play 200 game against the ai with the old network. The new network is validated it passes the unilateral z-test (function : `run.evaluate_network`). 


