# Neural Network Models
The network models are inspired by the works of DeepMind. You can see the network architecture in their [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0). 

## Inputs
The model takes in inputs the game state that is presented as an array of numerical value in 3 dimensions.

![game state](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/images/game%20state%20architecture.png)

## 2 Dimensional Convolution

![2D convolution](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/images/convolution.png)
The module `neural_network_models.py` need to install TensorFLow library, see the link : https://www.tensorflow.org/install/pip.

The function `build_model` builds a deep residual network. so as you can see in the lineit needs the input size (height, width, depth)
