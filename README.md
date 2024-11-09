# TicTacToe.AI
Q-learning agent that trains itself playing tic tac toe with another one

In this project, we develop a Q-learning agent that learns to play Tic-Tac-Toe autonomously by playing repeated games against another Q-learning agent. Q-learning is a type of reinforcement learning where an agent learns optimal actions based on rewards received from its environment. Here, the environment is the Tic-Tac-Toe game board, and the actions are the possible moves the agent can make.

Each agent starts with no knowledge of the game. As they play, they explore different moves, learning from wins, losses, and draws. The agents use a Q-table to store "Q-values," representing the quality of each possible action in a given board state. Over time, they adjust these values to maximize their chances of winning, minimizing unfavorable outcomes.

Through this self-play process, the agents reinforce successful strategies and discard less effective ones, gradually mastering the game. This method highlights how reinforcement learning can develop strategies through interaction and feedback rather than predefined rules. By the end of the training, our Q-learning agent can play competitively, demonstrating a learned understanding of Tic-Tac-Toe.

For better resuls please proceed more than 1 000 000 rounds, which can take many hours...
