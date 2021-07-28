##### In this repo, I create a program to run the simple mobile phone game Snake: https://en.wikipedia.org/wiki/Snake_(video_game_genre) and attempt to train an agent to play the game.

The model used is an implementation of the classic DeepMind Atari game(2013), featuring 3 convolutional layers and 2 dense layers.

We feed the previous two frames to the model, so that it can infer direction the snake is travelling.

The agent is reward with a positive point for eating food, and a negative point for dying. 

Due to the scarcity of positive points under random simulation, best results were achieved by training only on games that featured at least one positive score (otherwise it reaches a local minima of going round in a circle to stop itself from dying).

After 4 million simulations, the cycle score(score per 200,000 games) has increased from -41229 (purely random) to 32000, with the positive cycle score (ignoring neg. points for dying increasing from 15048 to 13000

Note: to see scores, look at both 'performance metrics.json' files, as they were run sequentially, with the model learning carried over to the second run.
