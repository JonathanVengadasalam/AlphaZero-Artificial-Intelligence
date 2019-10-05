# Neural Network Models
The network models are inspired by the works of DeepMind. You can see the network architecture in their [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0). The network takes in inputs the game state and gives in outputs the next move probabilities and the value of the game state, then this outputs are used to fill the monte-carlo-tree-search.

The game state is presented as a table of numerical value in 3 dimensions.

The module `neural_network_models.py` need to install TensorFLow library, see the link : https://www.tensorflow.org/install/pip.

The function `build_model` builds a deep residual network. so as you can see in the lineit needs the input size (height, width, depth)
