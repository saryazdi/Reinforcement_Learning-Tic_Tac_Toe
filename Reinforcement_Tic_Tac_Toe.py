# -*- coding: utf-8 -*-
# Code by Soroosh Saryazdi, 2017
# Teaching the game of Tic Tac Toe to a computer using Reinforcement Learning
import numpy as np
import random as rm
from termcolor import colored
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LEARNING WEIGHTS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ Function #1 : Board Analysis (Feature Extraction) ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Board_Analysis(board,team): # This function analyses the board and extracts features from it
    options = (np.argwhere(board == 0))
    total_options = options.shape[0]
    Data_table = np.zeros((total_options, 7))
    counter = 0
    board *= team
    for moves in options:
        # OP = Number of open paths
        # OP1 = Number of open paths with 1 team member
        # OP2 = Number of open paths with 2 team members
        # B1 = Number of paths it will block with 1 enemy
        # B2 = Number of paths it will block with 2 enemies
        OP = 0
        OP1 = 0
        OP2 = 0
        B1 = 0
        B2 = 0
        i = moves[0]
        j = moves[1]
        # Row path
        OP += int(((board[i][0] != -1) & (board[i][1] != -1) & (board[i][2] != -1)) == True)  # row i
        OP1 += int(((board[i][0] + board[i][1] + board[i][2]) > 0) == True)  # row i
        OP2 += int(((board[i][0] + board[i][1] + board[i][2]) == 2) == True)  # row i
        B1 += int(((board[i][0] + board[i][1] + board[i][2]) < 0) == True)  # row i
        B2 += int(((board[i][0] + board[i][1] + board[i][2]) == -2) == True)  # row i
        # Column Path
        OP += int(((board[0][j] != -1) & (board[1][j] != -1) & (board[2][j] != -1)) == True)  # col j
        OP1 += int(((board[0][j] + board[1][j] + board[2][j]) > 0) == True)  # col j
        OP2 += int(((board[0][j] + board[1][j] + board[2][j]) == 2) == True)  # col j
        B1 += int(((board[0][j] + board[1][j] + board[2][j]) < 0) == True)  # col j
        B2 += int(((board[0][j] + board[1][j] + board[2][j]) == -2) == True)  # col j
        # Diagonal Path 1
        if ((moves[0] == 0)&(moves[1]==0))or((moves[0] == 2)&(moves[1]==2)):  # \ Path
            OP += int(((board[0][0] != -1) & (board[1][1] != -1) & (board[2][2] != -1)) == True)  # \
            OP1 += int(((board[0][0] + board[1][1] + board[2][2]) > 0) == True)  # \
            OP2 += int(((board[0][0] + board[1][1] + board[2][2]) == 2) == True)  # \
            B1 += int(((board[0][0] + board[1][1] + board[2][2]) < 0) == True)  # \
            B2 += int(((board[0][0] + board[1][1] + board[2][2]) == -2) == True)  # \
        # Diagonal Path 2
        if ((moves[0] == 0)&(moves[1]==2))or((moves[0] == 2)&(moves[1]==0)):  # / Path
            OP += int(((board[0][2] != -1) & (board[1][1] != -1) & (board[2][0] != -1)) == True)  # /
            OP1 += int(((board[0][2] + board[1][1] + board[2][0]) > 0) == True)  # /
            OP2 += int(((board[0][2] + board[1][1] + board[2][0]) == 2) == True)  # /
            B1 += int(((board[0][2] + board[1][1] + board[2][0]) < 0) == True)  # /
            B2 += int(((board[0][2] + board[1][1] + board[2][0]) == -2) == True)  # /
        # X Path
        if ((moves[0] == 1)&(moves[1]==1)):  # X Path
            OP += int(((board[0][0] != -1) & (board[1][1] != -1) & (board[2][2] != -1)) == True)  # \
            OP1 += int(((board[0][0] + board[1][1] + board[2][2]) > 0) == True)  # \
            OP2 += int(((board[0][0] + board[1][1] + board[2][2]) == 2) == True)  # \
            B1 += int(((board[0][0] + board[1][1] + board[2][2]) < 0) == True)  # \
            B2 += int(((board[0][0] + board[1][1] + board[2][2]) == -2) == True)  # \
            OP += int(((board[0][2] != -1) & (board[1][1] != -1) & (board[2][0] != -1)) == True)  # /
            OP1 += int(((board[0][2] + board[1][1] + board[2][0]) > 0) == True)  # /
            OP2 += int(((board[0][2] + board[1][1] + board[2][0]) == 2) == True)  # /
            B1 += int(((board[0][2] + board[1][1] + board[2][0]) < 0) == True)  # /
            B2 += int(((board[0][2] + board[1][1] + board[2][0]) == -2) == True)  # /
        # ATTRIBUTE ASSIGNMENT
        attr1 = 0 # Number of open paths OP \\** NOTE ** Not using this attribute gave a better result, hence I changed its value to zero
        attr2 = 0  # Number of open paths with atleast 1 team member OP1 \\** NOTE ** Not using this attribute gave a better result, hence I changed its value to zero
        attr3 = OP2/2  # Number of open paths with 2 team members
        attr4 = (B1-B2)/4  # Number of paths it will block with atleast 1 enemy
        attr5 = (B2)/2  # Number of paths it will block with 2 enemies
        Data_table[counter][0] = moves[0]
        Data_table[counter][1] = moves[1]
        Data_table[counter][2] = attr1
        Data_table[counter][3] = attr2
        Data_table[counter][4] = attr3
        Data_table[counter][5] = attr4
        Data_table[counter][6] = attr5
        counter += 1
    return Data_table
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ Function #2 : Check For Winner (Has game ended?) ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Endgame_Check(board):
    rows = board.sum(1)
    columns = board.sum(0)
    d0 = [[1,0,0],[0,1,0],[0,0,1]]*board
    d1 = [[0,0,1],[0,1,0],[1,0,0]]*board
    diagonal0 = d0.sum()
    diagonal1 = d1.sum()
    Winner = 0
    if (np.max(abs(rows))==3):
        if (np.max(rows)==3):
            Winner = 1
        else:
            Winner = -1
    if (np.max(abs(columns))==3):
        if (np.max(columns)==3):
            Winner = 1
        else:
            Winner = -1
    if (np.max(abs(diagonal0))==3):
        if (diagonal0==3):
            Winner = 1
        else:
            Winner = -1
    if (np.max(abs(diagonal1))==3):
        if (diagonal1==3):
            Winner = 1
        else:
            Winner = -1
    return Winner
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Function #3 : Experiment Generator ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Experiment_Generator():
    Loop = True
    while (Loop):
        Play_Num = np.random.random(1)
        Play_Num = np.round(Play_Num*8)
        if (Play_Num>0):
            O_Plays = np.round(Play_Num/2)
            X_Plays = Play_Num - O_Plays
            Select = rm.sample([0,1,2,3,4,5,6,7,8],int(Play_Num))
            X_Select = rm.sample(Select,int(X_Plays))
            O_Select = list(set(Select) - set(X_Select))
            board = np.zeros([1,9])
            board[0][X_Select] = 1
            board[0][O_Select] = -1
            board.resize([3,3])
            Any_Winners = Endgame_Check(board)
            if X_Plays > O_Plays:
                first = -1
            elif O_Plays > X_Plays:
                first = 1
            else:
                first = rm.sample([-1,1],1)[0]
            if Any_Winners == 0:
                Loop = False
        else:
            board = np.zeros([1,9])
            board.resize([3,3])
            first = rm.sample([-1,1],1)[0]
            Loop = False
    return board, first
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ Function #4 : Find best move (Forward Prop)  ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Best_Move(Move_Attributes,Weights):
    Moves_Available = Move_Attributes.shape[0] # The number of available move options
    Move_Scores = np.zeros([Moves_Available,1]) # Initiate the matrix of score for each move
    for Move in range(0,Moves_Available): # For every available move:
        Features = Move_Attributes[Move][2:] # Get features of that move
        Move_Scores[Move] = np.dot(Features,Weights) # Calculate score of that move (Features*Weights)
    Max_Score = np.max(Move_Scores) # Find the max score
    Move_Scores = np.squeeze(Move_Scores)
    Max_Locations = np.ndarray.tolist(np.squeeze(np.argwhere(Move_Scores==Max_Score)))  # Find the moves with max score
    if type(Max_Locations) is int:
        Max_Loc = Max_Locations
        Selected_Move = Move_Attributes[Max_Loc][0:2]
        Selected_Move_Attributes = Move_Attributes[Max_Loc,2:]
    else:
        Max_Loc = rm.sample(Max_Locations,1)  # If we have multiple moves with score equal to max score, choose one of those moves randomly
        Selected_Move = Move_Attributes[Max_Loc[0]][0:2] # The location of the best move (selected move) on the board
        Selected_Move_Attributes = Move_Attributes[Max_Loc[0],2:] # The attributes of the best move (selected move)
    return Selected_Move, Selected_Move_Attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ Function #5 : Calculate Actual Score Values  ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Actual_Scores_Calc(Board_States, Winner, Weights):
    No_Winners = (abs(Winner)==0)
    if No_Winners:
        Winner = 1
        W_Last_Score = 0
        L_Last_Score = 0
    else:
        W_Last_Score = 100
        L_Last_Score = -100
    # Winner Actual Scores:
    if (Board_States.ndim>1):
        Winner_Moves_Loc = np.squeeze(np.argwhere(Board_States[:,0] == Winner))
        W_States = Board_States[Winner_Moves_Loc,1:] # Winner moves
    else:
        Winner_Moves_Loc = np.array([0])
        W_States = Board_States[1:]
    W_States_Num = W_States.shape[0]
    W_Actual_Scores = np.zeros([W_States_Num])
    if (W_States.ndim>1):
        for W_Move in range(0,W_States_Num-1):
            W_Actual_Scores[W_Move] = np.dot(W_States[W_Move+1,:],Weights)
        if (Winner != 0):
            W_Actual_Scores[-1]=W_Last_Score
    else:
        W_Actual_Scores = np.array([W_Last_Score])
    # Loser Actual Scores:
    L_States = np.array([0])
    L_Actual_Scores = np.array([0])
    if (Board_States.ndim>1):
        Loser_Moves_Loc = np.squeeze(np.argwhere(Board_States[:,0] != Winner))
        L_States = Board_States[Loser_Moves_Loc,1:]
        L_States_Num = L_States.shape[0]
        L_Actual_Scores = np.zeros([L_States_Num])
        if (L_States.ndim>1):
            for L_Move in range(0,L_States_Num-1):
                L_Actual_Scores[L_Move] = np.dot(L_States[L_Move+1][:],Weights)  
            L_Actual_Scores[-1]=L_Last_Score
        else:
            L_Actual_Scores = np.array([L_Last_Score])
    # Final Actual Scores:
        Actual_Scores = np.hstack((W_Actual_Scores,L_Actual_Scores))
    else:
        Actual_Scores = W_Actual_Scores
    return Actual_Scores, W_States, L_States, W_Actual_Scores, L_Actual_Scores
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ Function #6 : Calculate Predicted Scores Values ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Predicted_Scores_Calc(Board_States, Weights, W_States, L_States, W_States_Num, L_States_Num, W_Actual_Scores, L_Actual_Scores):
    #Winner Predicted Scores:
    W_Predicted_Scores = np.zeros([W_States_Num])
    if (W_States.ndim>1):
        W_Predicted_Scores[1:] = W_Actual_Scores[0:-1]
        W_Predicted_Scores[0] = np.dot(W_States[0],Weights)
    else:
        W_Predicted_Scores = np.dot(W_States,Weights)     
    # Loser Predicted Scores:
    if (Board_States.ndim>1):
        L_Predicted_Scores = np.zeros([L_States_Num])
        if (L_States.ndim>1):
            L_Predicted_Scores[1:] = L_Actual_Scores[0:-1]
            L_Predicted_Scores[0] = np.dot(L_States[0],Weights)
        else:
            L_Predicted_Scores = np.dot(L_States,Weights)
            # Final Predicted Scores:
        Predicted_Scores = np.hstack((W_Predicted_Scores,L_Predicted_Scores))
        X = np.vstack((W_States,L_States)) # Move attributes of winner and loser in order
    else:
        Predicted_Scores = W_Predicted_Scores
        X = W_States # Move attributes of winner
    return Predicted_Scores, X
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ Function #7 : Update Weights (Gradient Descent) ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Update_Weight_Values(Weights, Learning_Rate, Memory_Attributes, Winner, NumIteration):
    Board_States = np.squeeze(Memory_Attributes[~np.all(Memory_Attributes == 0, axis=1)])
    for iteration in range(0,NumIteration):
        Actual_Scores, W_States, L_States, W_Actual_Scores, L_Actual_Scores = Actual_Scores_Calc(Board_States, Winner, Weights)
        W_States_Num = W_States.shape[0]
        L_States_Num = L_States.shape[0]
        Predicted_Scores, X = Predicted_Scores_Calc(Board_States, Weights, W_States, L_States, W_States_Num, L_States_Num, W_Actual_Scores, L_Actual_Scores)
        # ------ Update Weights ------
        XT = X.T # Transpose of X
        if (Actual_Scores.shape[0]>1):
            Grad_E_W = -1 * np.dot(XT,Actual_Scores-Predicted_Scores)
        else:
            Grad_E_W = -1 * (XT*(Actual_Scores-Predicted_Scores))
        Update_Value = (Learning_Rate*Grad_E_W)/Actual_Scores.shape[0]
        Update_Value = Update_Value.reshape(Weights.shape)
        Updated_Weights = Weights - Update_Value
        Updated_Weights=Updated_Weights.reshape(Weights.shape)
        Weights = Updated_Weights
    return Updated_Weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ Function #8 : Play against yourself and Learn ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Play_and_Learn(Num_Times,Initial_Weights,Learning_Rate,NumIteration):
    Weights = Initial_Weights
    for new_game in range(0,Num_Times):
        if (new_game == np.round(Num_Times/4)):
            print(colored(' [25% ','green'),end="")
        elif (new_game == np.round(Num_Times/2)):
            print(colored('50% ','green'),end="")
        elif (new_game == np.round((3*Num_Times)/4)):
            print(colored('75% ','green'),end="")
        elif (new_game == np.round(Num_Times-1)):
            print(colored('100%]','green'))
        board, team = Experiment_Generator() # Starting board state and first team to play
        options = (np.argwhere(board == 0))
        max_rounds = options.shape[0] # How many rounds will the game last maximum?
        Memory_Attributes = np.zeros([max_rounds,6]) # Initialize memory that stores states
        # Memory_Attributes: Holds attributes of move and team of move (to check if it was related to the winner or the loser)
        for plays in range(0,max_rounds):
            Move_Attributes = Board_Analysis(board,team) # Analyze board
            Selected_Move,Selected_Move_Attributes = Best_Move(Move_Attributes,Weights)
            index0 = int(Selected_Move[0])
            index1 = int(Selected_Move[1])
            board[index0][index1] = team # Play (select best move)
            Memory_Attributes[plays][:] = np.hstack((team,Selected_Move_Attributes))  # Store move attributes in memory + which team the move was for
            Winner = Endgame_Check(board)
            if Winner != 0:
                break
            team *= -1 # Change Teams
        Updated_Weights = Update_Weight_Values(Weights, Learning_Rate, Memory_Attributes, Winner, NumIteration)
        Weights = Updated_Weights
    return Weights



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ THE GAME (COMPUTER VS. HUMAN)  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~ Function: Computer's Play  ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Computer_Move(board, computer_team, Learned_Weights):
    New_Board = board
    Move_Attributes = Board_Analysis(board,computer_team) # Analyze board
    Selected_Move,Selected_Move_Attributes = Best_Move(Move_Attributes,Learned_Weights) # Find best move @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    index0 = int(Selected_Move[0])
    index1 = int(Selected_Move[1])
    New_Board[index0][index1] = computer_team # Play (select best move)
    return New_Board
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~ Function: Human's Play  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Human_Move(board, human_team):
    print('*****',colored('Your Turn!','green'),'*****\n')
    # ###### Print Board and take input ######
    # We create a dictionary for printing the board with X and O and number of tiles (XO_Dictionary) as opposed to printing
    # board with values of 0 and +1 and -1. For this purpose we create another board (Dict_Board_Values) and in it change
    # each 0 tile with its (10*index_number(starting from 1, not 0)). The reason for using 10*index_number and not the index_number
    # itself is because the number 1 is used in the dictionary for the value of X.
    New_Board = board.copy()
    New_Board.resize([1,9])
    New_Board = np.squeeze(New_Board)
    XO_Dictionary={human_team:colored('X', 'cyan',attrs=['underline']),(-1*human_team):colored('O', 'red',attrs=['underline']),10:1,20:2,30:3,40:4,50:5,60:6,70:7,80:8,90:9}
    Dict_Board_Values = np.squeeze(New_Board.copy())
    Available_Moves = np.array([])
    for i in range(0,9):
        if (New_Board[i]==0):
            Available_Moves = np.append(Available_Moves,i)
            Dict_Board_Values[i] = (i+1)*10
    for array in range(0,9):
        Val = XO_Dictionary[Dict_Board_Values[array]]
        if (array == 0) or (array == 1) or (array == 3) or (array == 4) or (array == 6) or (array == 7):
            if type(Val) is int:
                Val = colored(Val, 'grey')
            print(' ',Val,' |',end="")
        elif (array == 2) or (array == 5):
            if type(Val) is int:
                Val = colored(Val, 'grey')
            print(' ',Val,' ')
            print('------------------')
        else:
            if type(Val) is int:
                Val = colored(Val, 'grey')
            print(' ',Val,' ')
        del Val
    while (True):
        Human_Choice = int(input("Where do you want to place X?\n>> ")) -1
        if Human_Choice in Available_Moves:
            break
        print(colored('Move not allowed!','red'),end="")
    # ########################################
    New_Board[Human_Choice] = human_team
    New_Board.resize([3,3])
    print(colored('-----------------------------------','cyan'))
    return New_Board
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Num_Times = 10000
Initial_Weights = np.array([[0.01],[0.01],[0.01],[0.01],[0.01]])
Learning_Rate = 0.1
NumIteration = 1
print('\n\n\n\n\n\n\n\n')
print('***************************************************')
print('*******************',colored('WELCOME TO','cyan'),'********************')
print('*******',colored('ARTIFICIAL INTELLIGENCE TIC TAC TOE!','cyan'),'******')
print('***************************************************')
print('******* Written by: Soroush Saryazdi, 2017. *******')
print('***************************************************')
print(colored('> Hi','green'))
print(colored('> Practicing... Please wait','green'),end="")
Learned_Weights = Play_and_Learn(Num_Times,Initial_Weights,Learning_Rate,NumIteration)

# 5,000 Plays Result :
# Learned_Weights = np.array([[0.01],[0.01],[121.96599],[36.513437],[39.39943]]) # 5000 Plays

#   200,000 plays result:
#   Learned_Weights [[  1.00000000e-02]
# [  1.00000000e-02]
# [  1.71161231e+02]
# [ -1.24281391e+02]
# [  8.11437798e+01]]

#print('Learned_Weights',Learned_Weights)

print(colored('> OK I\'m ready!','green'))
print(colored('> I\'m','green'),colored('O','red',attrs=['underline']),end="")
print(colored(', You\'re','green'),colored('X','cyan',attrs=['underline']),colored('. Goodluck!','green'))
First_Round = True
human_team = -1
computer_team = 1
Moves_Played = 0
while (True):
    if (First_Round):
        print(colored('-----------------------------------','cyan'))
        print(colored('-------- New Game Started! --------','cyan'))
        print(colored('-----------------------------------','cyan'))
        board = np.zeros([3,3])
        team = rm.sample([-1,1],1)[0]
        if (team==computer_team):
            print(colored('NOTIFICATION:>>>The computer went first!<<<','red'))
        Moves_Played = 0
        First_Round = False
    if (team == human_team):
        board = Human_Move(board,human_team)
        Moves_Played += 1
    elif (team == computer_team):
        board = Computer_Move(board, computer_team, Learned_Weights)
        Moves_Played += 1
    Winner_Team = Endgame_Check(board)
    Early_End = abs(Winner_Team)
    team = -1 * team
    print('')
    if (Early_End==1) or (Moves_Played==9):
        XO_Winner_Dictionary={human_team:colored('X', 'cyan',attrs=['underline']),(-1*human_team):colored('O', 'red',attrs=['underline']),0:' '}    
        Dict_Board = board.copy()
        Dict_Board.resize([1,9])
        Dict_Board = np.squeeze(Dict_Board)
        for array in range(0,9):
            Val = XO_Winner_Dictionary[Dict_Board[array]]
            if (array == 0) or (array == 1) or (array == 3) or (array == 4) or (array == 6) or (array == 7):
                if type(Val) is int:
                    Val = colored(Val, 'grey')
                print(' ',Val,' |',end="")
            elif (array == 2) or (array == 5):
                if type(Val) is int:
                    Val = colored(Val, 'grey')
                print(' ',Val,' ')
                print('------------------')
            else:
                if type(Val) is int:
                    Val = colored(Val, 'grey')
                print(' ',Val,' ')
            del Val
        print('')
        if (Winner_Team == human_team):
            print(colored('~~~~~~~~~~~~~~~~~~~~~~~~~','green'))
            print(colored('~~~~~~~~ You Won! ~~~~~~~','green'))
            print(colored('~~~~~~~~~~~~~~~~~~~~~~~~~','green'),end="")
        elif (Winner_Team == computer_team):
            print(colored('~~~~~~~~~~~~~~~~~~~~~~~~~','red'))
            print(colored('~~~~~~~ You Lost! ~~~~~~~','red'))
            print(colored('~~~~~~~~~~~~~~~~~~~~~~~~~','red'),end="")
        else:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('~~~~~~~~~ Draw! ~~~~~~~~~')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~',end="")
        Play_Again = input("Play again? [y n]\n>> ")
        if (Play_Again.lower() == 'n'):
            break
        else:
            print('\n\n\n\n\n\n\n\n\n\n\n')
            First_Round = True