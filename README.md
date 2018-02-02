# Reinforcement Learning: Tic Tac Toe
<img src="https://github.com/saryazdi/Reinforcement_Learning-Tic_Tac_Toe/blob/master/TIC_TAC_TOE_Game.jpg?raw=true" width="400" align="right" alt="Computer Hope">
This code is written in <b>PYTHON 3</b>.</br>
<b>Dependencies</b></br>
-numpy</br>
-random</br>
-termcolor (for printing colored text)

<h2>Introduction</h2>

<h3>Basic Introduction:</h3> This Python code trains a model to play Tic Tac Toe. The model learns to play Tic Tac Toe by playing the game against itself for several thousand times. During these games, the model tries to learn the best moves to take in order to win (Reinforcement Learning). After the model is trained, the user can play Tic Tac Toe against the model.

<h3>More Specific Introduction:</h3> The model used is a single neuron, because Tic Tac Toe is a fairly simple game. Training is done using gradient descent. Values are assigned to each state of the game after the game is finished based on the explanation in the book "Machine Learning" by Dr. Tom Mitchell. The features that are used by the model are:</br>
-Number of open paths for the query move with 2 team members</br>
-Number of paths that the query move will block with atleast 1 enemy</br>
-Number of paths that the query move will block with 2 enemies

<h2>---- Code Instructions ----</h2>
<h2>Defined Functions:</h2>
<h3>-Board_Analysis(board,team):</h3> This function analyses the board and extracts features from it

<h3>-Endgame_Check(board):</h3> This function checks if the game has ended, and if so, who has won the game.

<h3>-Experiment_Generator():</h3> This function creates the initial board state when the model is playing against itself in the training phase.

<h3>-Best_Move(Move_Attributes,Weights):</h3> This function finds the best move to take by choosing the state with the maximum predicted value (forward propagation).

<h3>-Actual_Scores_Calc(Board_States, Winner, Weights):</h3> This function calculates the actual values of each state after game has ended.

<h3>-Predicted_Scores_Calc(...):</h3> This function calculates the values that our model predicted during the game, in order to calculate the error between the predicted values and the actual values.


<h3>-Update_Weight_Values(...):</h3> This function updates the weights of our neuron based on the error.

<h3>-Play_and_Learn(Num_Times,Initial_Weights,Learning_Rate,NumIteration):</h3> This function asks the model to play %Num_Times times against itself, while updating the weights %NumIteration times using Gradient Descent after each round of playing. The model updates weights based on the plays of both agents, i.e. both the agent that has lost the game and the agent that has won.

<h3>-Computer_Move(board, computer_team, Learned_Weights):</h3> This function causes the model to make a move while playing against a human based on the learned weights in the training phase.

<h3>-Human_Move(board, human_team):</h3> This function asks the human to decide on the move to play against the computer.
