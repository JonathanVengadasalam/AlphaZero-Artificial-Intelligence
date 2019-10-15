# AlphaZero Artificial Intellignence
An implementation of an artificial intelligence using deep neural network to play board game.

## About The Project
This project is an implementation of an algorythme for playing 2 players board games like tic tac toe, connect4, gomoku. Inspired by the Google "AlphaGo Zero cheat sheet" [paper](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0), the algorythme uses deep neural network trained by self-playing. In this repository, I trained network model for the connect4 game. See more about the code description and the machine learning, in [code_description.md](https://github.com/JonathanVengadasalam/AlphaZero-Artificial-Intelligence/blob/master/code_description.md).

## Getting Started
### Prerequisites
* numpy `py -m pip install numpy`
* tensorflow `pip install --upgrade tensorflow`

### Usage
`from main import *`

to play connect4 with provided model, run the following script from the directory:

`human_play()`

to train the provided model for the game connect4, run the following script:

`train()`

## Acknowledgements
* [TowardsDataScience](https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a)
