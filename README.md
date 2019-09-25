# AlphaZero Artificial Intellignence
This project is an implementation of an artificial intelligence inspired by the [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0) "AlphaGo Zero cheat sheet" develloped by Google 

The artificial intelligence (ai) is the class "AI" in the module [artificial_intelligence/ai.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/artificial_intelligence/ai.py). For a given position of a game state, the ai gives the best next move. To found this next move, it can use 2 main method :
 - Monte Carlo Tree Search (function : "ai.montecarlo_treesearch"), it doesn't use neural network but use monte carlo method to build the research tree ([wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)).
 - Neural Network Tree Search (function : "ai.neuralnetwork_treesearch"), I modified the classical mcts function so that it integrates neural network, the function use the network to evaluate the positions and build the research tree. The network gives to results :
   - the policy : the probabilities of each next move.
   - the value : the probability that this position is winning.

The games are in [environnement](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/tree/master/environnements). I implement the game Connect 4 in the module [envConnect](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/environnements/envConnect.py) and I use this game in this project to improve my neural network model.
<a/>

To try to obtain subsets of similar elements that correspond to the human recognition of numbers.
