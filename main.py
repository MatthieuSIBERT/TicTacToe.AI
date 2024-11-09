import numpy as np
import itertools as it
import csv
#tqdm is a package that allows you to display a progress bar for loops
from tqdm import tqdm

# In this project, we develop a Q-learning agent that learns to play Tic-Tac-Toe autonomously by playing repeated games against another Q-learning agent. Q-learning is a type of reinforcement learning where an agent learns optimal actions based on rewards received from its environment. Here, the environment is the Tic-Tac-Toe game board, and the actions are the possible moves the agent can make.

# Each agent starts with no knowledge of the game. As they play, they explore different moves, learning from wins, losses, and draws. The agents use a Q-table to store "Q-values," representing the quality of each possible action in a given board state. Over time, they adjust these values to maximize their chances of winning, minimizing unfavorable outcomes.

# Through this self-play process, the agents reinforce successful strategies and discard less effective ones, gradually mastering the game. This method highlights how reinforcement learning can develop strategies through interaction and feedback rather than predefined rules. By the end of the training, our Q-learning agent can play competitively, demonstrating a learned understanding of Tic-Tac-Toe.

# For better results please proceed more than 1 000 000 rounds, which can take many hours...

symbols = [0,1,2]
player = [1,2]
permutations = list(it.product(symbols, repeat=9))

# Removing impossible values where the difference between the number of 1s and 2s is greater than 1, cause this would mean a player has played twice in a row
toRemove = []

for perm in tqdm(permutations):
    if perm.count(symbols[1]) - perm.count(symbols[2]) > 1:
        toRemove.append(permutations.index(perm))
    if perm.count(symbols[2]) - perm.count(symbols[1]) > 1:
        toRemove.append(permutations.index(perm))
    pass

permutations = [perm for i, perm in enumerate(permutations) if i not in toRemove]

toRemove = []

for perm in tqdm(permutations):
    if perm.count(0) == 0:
        toRemove.append(permutations.index(perm))
    pass

permutations = [perm for i, perm in enumerate(permutations) if i not in toRemove]

permutations1 = permutations.copy()
permutations2 = permutations.copy()

# Removing where player1 does not have to play
toRemove = []
for perm in tqdm(permutations1):
    if perm.count(symbols[1]) > perm.count(symbols[2]):
        toRemove.append(permutations1.index(perm))
    pass

permutations1 = [perm for i, perm in enumerate(permutations1) if i not in toRemove]

q_values1 = np.zeros((len(permutations1),9))

# Removing where player2 does not have to play
toRemove = []
for perm in tqdm(permutations2):
    if perm.count(symbols[1]) < perm.count(symbols[2]):
        toRemove.append(permutations2.index(perm))
    pass

permutations2 = [perm for i, perm in enumerate(permutations2) if i not in toRemove]

q_values2 = np.zeros((len(permutations2),9))

#parameters for the Q-learning that you can change
learing_rate = 1
discount_factor = 0.9
exploration_rate = 0.5
rounds = 1000
life = 0

#variables to keep track of the number of victories of each player
victoryPlayer1 = 0
victoryPlayer2 = 0

# For each parties
for round in tqdm(range(rounds)):
    game = np.array([0,0,0,0,0,0,0,0,0])
    play = 0
    play1 = 0
    play2 = 0
    play3 = 0
    play4 = 0
    play5 = 0
    play6 = 0
    play7 = 0
    playerSelected = np.random.choice(player)
    history = np.zeros([9,9])
    actionHistory = [0,0,0,0,0,0,0,0,0]
    q_values_current_state = np.zeros(9)
    rewards1 = 0
    rewards2 = 0
    victory = 0
    line = 0

    # While the game is not over
    while play < 9:
        
        actionList = np.where(game == 0)[0]
        history[play] = game
   
        if playerSelected == 1:
            for i in range(len(permutations1)):
                if np.array_equal(permutations1[i], game):
                    line = i
                    q_values_current_state = q_values1[line]
                    break
        elif playerSelected == 2:
            for i in range(len(permutations2)):
                if np.array_equal(permutations2[i], game):
                    line = i
                    q_values_current_state = q_values2[line]
                    break

        # Choose the action
        if np.random.rand() < exploration_rate:
            action = np.random.choice(actionList)
        else:
            q_values_current_state0 = np.array(q_values_current_state)[actionList]
            max = np.argmax(q_values_current_state0)
            max = actionList[max]
            action = max
        # Making the move
        game[action] = playerSelected
        actionHistory[play] = action
        # Player1 winning condition
        if((game[0] == 1 and game[1] == 1 and game[2] ==1) or (game[3] == 1 and game[4] == 1 and game[5] ==1) or (game[6] == 1 and game[7] == 1 and game[8] ==1) or (game[0] == 1 and game[3] == 1 and game[6] ==1) or (game[1] == 1 and game[4] == 1 and game[7] ==1) or (game[2] == 1 and game[5] == 1 and game[8] ==1) or (game[0] == 1 and game[4] == 1 and game[8] ==1) or (game[2] == 1 and game[4] == 1 and game[6] ==1)):
            rewards1 = 1
            rewards2 = -1
            victory = 1
        # Player2 winning condition
        if((game[0] == 2 and game[1] == 2 and game[2] == 2) or (game[3] == 2 and game[4] == 2 and game[5] == 2) or (game[6] == 2 and game[7] == 2 and game[8] == 2) or (game[0] == 2 and game[3] == 2 and game[6] == 2) or (game[1] == 2 and game[4] == 2 and game[7] == 2) or (game[2] == 2 and game[5] == 2 and game[8] == 2) or (game[0] == 2 and game[4] == 2 and game[8] == 2) or (game[2] == 2 and game[4] == 2 and game[6] == 2)):
            rewards1 = -1
            rewards2 = 1
            victory = 1
        # End of the game
        if victory ==1:
            # If player1 wins
            if playerSelected == 1:
                victoryPlayer1 = victoryPlayer1 + 1
                if play == 4:
                    # Update the q_values for the last action of player1
                    q_values1[line][action] = q_values1[line][action] + learing_rate * rewards1
                    # Update the q_values for the last action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[3]):
                            play3 = i
                            break
                    q_values2[play3][actionHistory[3]] = q_values2[play3][actionHistory[3]] + learing_rate * rewards2
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[2]):
                            play2 = i
                            break
                    q_values1[play2][actionHistory[2]] = q_values1[play2][actionHistory[2]] + learing_rate  * (discount_factor * rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[1]):
                            play1 = i
                            break
                    q_values2[play1][actionHistory[1]] = q_values2[play1][actionHistory[1]] + learing_rate  * (discount_factor *  rewards2 - life)
                    # Uptade the q_values for the very first action of player1
                    q_values1[0, actionHistory[0]] = q_values1[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards1 - life * 2)
                elif play == 5:
                    # Update the q_values for the last action of player1
                    q_values1[line][action] = q_values1[line][action] + learing_rate * rewards1
                    # Update the q_values for the last action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[4]):
                            play4 = i
                            break
                    q_values2[play4][actionHistory[4]] = q_values2[play4][actionHistory[4]] + learing_rate * rewards2
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[3]):
                            play3 = i
                            break
                    q_values1[play3][actionHistory[3]] = q_values1[play3][actionHistory[3]] + learing_rate  * (discount_factor * rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[2]):
                            play2 = i
                            break
                    q_values2[play2][actionHistory[2]] = q_values2[play2][actionHistory[2]] + learing_rate  * (discount_factor *  rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[1]):
                            play1 = i
                            break
                    q_values1[play1][actionHistory[1]] = q_values1[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Uptade the q_values for the very first action of player2
                    q_values2[0, actionHistory[0]] = q_values2[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards2 - life * 2)
                elif play == 6:
                    # Update the q_values for the last action of player1
                    q_values1[line][action] = q_values1[line][action] + learing_rate * rewards1
                    # Update the q_values for the last action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[5]):
                            play5 = i
                            break
                    q_values2[play5][actionHistory[5]] = q_values2[play5][actionHistory[5]] + learing_rate * rewards2
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[4]):
                            play4 = i
                            break
                    q_values1[play4][actionHistory[4]] = q_values1[play4][actionHistory[4]] + learing_rate  * (discount_factor * rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[3]):
                            play3 = i
                            break
                    q_values2[play3][actionHistory[3]] = q_values2[play3][actionHistory[3]] + learing_rate  * (discount_factor *  rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[2]):
                            play2 = i
                            break
                    q_values1[play2][actionHistory[2]] = q_values1[play2][actionHistory[2]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[1]):
                            play1 = i
                            break
                    q_values2[play1][actionHistory[1]] = q_values2[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Uptade the q_values for the very first action of player
                    q_values1[0, actionHistory[0]] = q_values1[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * discount_factor * rewards1 - life * 3)
                elif play == 7:
                    # Update the q_values for the last action of player1
                    q_values1[line][action] = q_values1[line][action] + learing_rate * rewards1
                    # Update the q_values for the last action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[6]):
                            play6 = i
                            break
                    q_values2[play6][actionHistory[6]] = q_values2[play6][actionHistory[6]] + learing_rate * rewards2
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[5]):
                            play5 = i
                            break
                    q_values1[play5][actionHistory[5]] = q_values1[play5][actionHistory[5]] + learing_rate  * (discount_factor * rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[4]):
                            play4 = i
                            break
                    q_values2[play4][actionHistory[4]] = q_values2[play4][actionHistory[4]] + learing_rate  * (discount_factor *  rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[3]):
                            play3 = i
                            break
                    q_values1[play3][actionHistory[3]] = q_values1[play3][actionHistory[3]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[2]):
                            play2 = i
                            break
                    q_values2[play2][actionHistory[2]] = q_values2[play2][actionHistory[2]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[1]):
                            play1 = i
                            break
                    q_values1[play1][actionHistory[1]] = q_values1[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * discount_factor * rewards1 - life * 3)
                    # Uptade the q_values for the very first action of player2
                    q_values2[0, actionHistory[0]] = q_values2[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * discount_factor * rewards2 - life * 3)
                elif play == 8:
                    # Update the q_values for the last action of player1
                    q_values1[line][action] = q_values1[line][action] + learing_rate * rewards1
                    # Update the q_values for the last action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[7]):
                            play7 = i
                            break
                    q_values2[play7][actionHistory[7]] = q_values2[play7][actionHistory[7]] + learing_rate * rewards2
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[6]):
                            play6 = i
                            break
                    q_values1[play6][actionHistory[6]] = q_values1[play6][actionHistory[6]] + learing_rate  * (discount_factor * rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[5]):
                            play5 = i
                            break
                    q_values2[play5][actionHistory[5]] = q_values2[play5][actionHistory[5]] + learing_rate  * (discount_factor *  rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[4]):
                            play4 = i
                            break
                    q_values1[play4][actionHistory[4]] = q_values1[play4][actionHistory[4]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[3]):
                            play3 = i
                            break
                    q_values2[play3][actionHistory[3]] = q_values2[play3][actionHistory[3]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[2]):
                            play2 = i
                            break
                    q_values1[play2][actionHistory[2]] = q_values1[play2][actionHistory[2]] + learing_rate  * (discount_factor * discount_factor * discount_factor * rewards1 - life * 3)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[1]):
                            play1 = i
                            break
                    q_values2[play1][actionHistory[1]] = q_values2[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * discount_factor * rewards2 - life * 3)
                    # Uptade the q_values for the very first action of player1
                    q_values1[0, actionHistory[0]] = q_values1[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * discount_factor * discount_factor * rewards1 - life * 4)
            # If player2 wins
            elif playerSelected == 2:
                victoryPlayer2 = victoryPlayer2 + 1
                if play == 4:
                    # Update the q_values for the last action of player2
                    q_values2[line][action] = q_values2[line][action] + learing_rate * rewards2
                    # Update the q_values for the last action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[3]):
                            play3 = i
                            break
                    q_values1[play3][actionHistory[3]] = q_values1[play3][actionHistory[3]] + learing_rate * rewards1
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[2]):
                            play2 = i
                            break
                    q_values2[play2][actionHistory[2]] = q_values2[play2][actionHistory[2]] + learing_rate  * (discount_factor * rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[1]):
                            play1 = i
                            break
                    q_values1[play1][actionHistory[1]] = q_values1[play1][actionHistory[1]] + learing_rate  * (discount_factor *  rewards1 - life)
                    # Uptade the q_values for the very first action of player2
                    q_values2[0, actionHistory[0]] = q_values2[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards2 - life * 2)
                elif play == 5:
                    # Update the q_values for the last action of player2
                    q_values2[line][action] = q_values2[line][action] + learing_rate * rewards2
                    # Update the q_values for the last action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[4]):
                            play4 = i
                            break
                    q_values1[play4][actionHistory[4]] = q_values1[play4][actionHistory[4]] + learing_rate * rewards1
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[3]):
                            play3 = i
                            break
                    q_values2[play3][actionHistory[3]] = q_values2[play3][actionHistory[3]] + learing_rate  * (discount_factor * rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[2]):
                            play2 = i
                            break
                    q_values1[play2][actionHistory[2]] = q_values1[play2][actionHistory[2]] + learing_rate  * (discount_factor *  rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[1]):
                            play1 = i
                            break
                    q_values2[play1][actionHistory[1]] = q_values2[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Uptade the q_values for the very first action of player1
                    q_values1[0, actionHistory[0]] = q_values1[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards1 - life * 2)
                elif play == 6:
                    # Update the q_values for the last action of player2
                    q_values2[line][action] = q_values2[line][action] + learing_rate * rewards2
                    # Update the q_values for the last action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[5]):
                            play5 = i
                            break
                    q_values1[play5][actionHistory[5]] = q_values1[play5][actionHistory[5]] + learing_rate * rewards1
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[4]):
                            play4 = i
                            break
                    q_values2[play4][actionHistory[4]] = q_values2[play4][actionHistory[4]] + learing_rate  * (discount_factor * rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[3]):
                            play3 = i
                            break
                    q_values1[play3][actionHistory[3]] = q_values1[play3][actionHistory[3]] + learing_rate  * (discount_factor *  rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[2]):
                            play2 = i
                            break
                    q_values2[play2][actionHistory[2]] = q_values2[play2][actionHistory[2]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[1]):
                            play1 = i
                            break
                    q_values1[play1][actionHistory[1]] = q_values1[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Uptade the q_values for the very first action of player2
                    q_values2[0, actionHistory[0]] = q_values2[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards2 - life * 2)
                elif play == 7:
                    # Update the q_values for the last action of player2
                    q_values2[line][action] = q_values2[line][action] + learing_rate * rewards2
                    # Update the q_values for the last action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[6]):
                            play6 = i
                            break
                    q_values1[play6][actionHistory[6]] = q_values1[play6][actionHistory[6]] + learing_rate * rewards1
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[5]):
                            play5 = i
                            break
                    q_values2[play5][actionHistory[5]] = q_values2[play5][actionHistory[5]] + learing_rate  * (discount_factor * rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[4]):
                            play4 = i
                            break
                    q_values1[play4][actionHistory[4]] = q_values1[play4][actionHistory[4]] + learing_rate  * (discount_factor *  rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[3]):
                            play3 = i
                            break
                    q_values2[play3][actionHistory[3]] = q_values2[play3][actionHistory[3]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[2]):
                            play2 = i
                            break
                    q_values1[play2][actionHistory[2]] = q_values1[play2][actionHistory[2]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[1]):
                            play1 = i
                            break
                    q_values2[play1][actionHistory[1]] = q_values2[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Uptade the q_values for the very first action of player1
                    q_values1[0, actionHistory[0]] = q_values1[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards1 - life * 2)
                elif play == 8:
                    # Update the q_values for the last action of player2
                    q_values2[line][action] = q_values2[line][action] + learing_rate * rewards2
                    # Update the q_values for the last action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[7]):
                            play7 = i
                            break
                    q_values1[play7][actionHistory[7]] = q_values1[play7][actionHistory[7]] + learing_rate * rewards1
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[6]):
                            play6 = i
                            break
                    q_values2[play6][actionHistory[6]] = q_values2[play6][actionHistory[6]] + learing_rate  * (discount_factor * rewards2 - life)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[5]):
                            play5 = i
                            break
                    q_values1[play5][actionHistory[5]] = q_values1[play5][actionHistory[5]] + learing_rate  * (discount_factor *  rewards1 - life)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[4]):
                            play4 = i
                            break
                    q_values2[play4][actionHistory[4]] = q_values2[play4][actionHistory[4]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[3]):
                            play3 = i
                            break
                    q_values1[play3][actionHistory[3]] = q_values1[play3][actionHistory[3]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Update the q_values for a previous action of player2
                    for i in range(len(permutations2)):
                        if np.array_equal(permutations2[i], history[2]):
                            play2 = i
                            break
                    q_values2[play2][actionHistory[2]] = q_values2[play2][actionHistory[2]] + learing_rate  * (discount_factor * discount_factor * rewards2 - life * 2)
                    # Update the q_values for a previous action of player1
                    for i in range(len(permutations1)):
                        if np.array_equal(permutations1[i], history[1]):
                            play1 = i
                            break
                    q_values1[play1][actionHistory[1]] = q_values1[play1][actionHistory[1]] + learing_rate  * (discount_factor * discount_factor * rewards1 - life * 2)
                    # Uptade the q_values for the very first action of player2
                    q_values2[0, actionHistory[0]] = q_values2[0, actionHistory[0]] + learing_rate * ( discount_factor * discount_factor * rewards2 - life * 2)
            break    
        # If the game is not over reset some parameters
        play = play + 1
        playerSelected = 3 - playerSelected

print("Player1 wins: ", victoryPlayer1)
print("Player2 wins: ", victoryPlayer2)

with open('q_values1.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(q_values1)

with open('permutations1.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(permutations1)

with open('q_values2.csv', 'w',newline="") as f:
    writer = csv.writer(f)
    writer.writerows(q_values2)

with open('permutations2.csv', 'w',newline="") as f:
    writer = csv.writer(f)
    writer.writerows(permutations2)