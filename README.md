# AlphaZero Artificial Intellignence
An implementation of an artificial intelligence using deep neural network to play board game.

## About The Project
This project is an implementation of an algorythme for playing 2 players board games like tic tac toe, connect4, gomoku. Inspired by the Google "AlphaGo Zero cheat sheet" [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0), the algorythme uses deep neural network trained by self-playing. See more about the code description and the machine learning, in [code_description.md](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/code_description.md).

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
