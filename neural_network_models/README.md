# Neural Network Models
The network models are inspired by the works of DeepMind. You can see the network architecture in their [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0).

The neural network is used to evaluate an environnement state (value prediction) and direct the tree research (policy prediction). It takes in input the environnement state and gives in outputs the value and the policy. The network graph is builded in [modelBuilder.py](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/neural_network_models/modelBuilder.py) in the function `build_model`. The graph is composed of a common core and 2 branches for the outputs.

![model](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/images/model.png)

## Inputs
The input is an array of numerical value in 3 dimensions.

![game state](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/images/game%20state%20architecture.png)

## Layers and Functions
 - [Fully Connected](https://en.wikipedia.org/wiki/Artificial_neural_network) : `Dense` is a connected neural layer.

 - [2 Dimensional Convolution](https://en.wikipedia.org/wiki/Convolutional_neural_network) : `Conv2D`
![2D convolution](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/images/2DConvolution.png)

 - [Batch Normalization](https://en.wikipedia.org/wiki/Batch_normalization) : `BatchNormalization` takes in input the 3D array tab(x, y, z). For i from 1 to n, the function gets the 2D arrays tab(., ., i). For each tab(., ., i), it calculates its mean and its standard deviation  and computes `tab(., ., i) = (tab(., ., i) - mean(i))/standard_deviation(i)`. It return the transformed tab(x, y, z).

 - [Relectified Linear Unit Activation](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) : `Actiavation("relu")` takes in input the 3D array, for each coordinates i, j, k of the array the function computes `f(tab(i, j, k)) = max(0, tab(i, j, k))` and return the array.

The module `neural_network_models.py` need to install TensorFLow library, see the link : https://www.tensorflow.org/install/pip.
