# AlphaZero Artificial Intellignence
This project is an implementation of an artificial intelligence inspired by the [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0) "AlphaGo Zero cheat sheet" develloped by Google 

##Code:

1. The artificial intelligence (ai) is in the class "AI" in the module [artificial_intelligence/ai.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/artificial_intelligence/ai.py). For a given position of a game state, the ai gives the best next move. To found this next move, it can use 2 main method :
 - Monte Carlo Tree Search (function : "ai.montecarlo_treesearch"), it doesn't use neural network but use monte carlo method to build the research tree ([wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)).
 - Neural Network Tree Search (function : "ai.neuralnetwork_treesearch"), I modified the classical mcts function so that it integrates neural network, the function use the network to evaluate the positions and build the research tree. The network gives to results :
   - the policy : the probabilities of each next move.
   - the value : the probability that this position is winning.

With this 2 main method, the ai use others hyperparameters:
 - the iteration : the number of node that are added to build the tree.
 - the formula : the upper confidence bound function and the alpha parameter that balances the exploitation and the exploration.
 - the selection : choise the move deterministically (the node that has the highest visits) or stochastically.

2. The games are in [environnements](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/tree/master/environnements) (env). I implement the game Connect 4 in the module [envConnect](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/environnements/envConnect.py) and I use this game in this project to improve my neural network model. You can implement your one game like chess or draughts but be carefull that your class contains the function [clone, convertmove, domove, getmoves, value, update_y, x, y].

3. The module manager contains the [functions](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/manager/functions.py) that manage env and ai(s) to :
   - play against the ai (function : "play").
   - collect data and results by playing ai vs ai (function : "selfplay").
   - test the performance of an ai on a game position (function : "testposition").


<a/>

To try to obtain subsets of similar elements that correspond to the human recognition of numbers.
