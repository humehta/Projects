#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

import sys
# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] )

# Count total # of pieces on diagonals
def count_on_diagonal(board,row,col):
    sum = 0
    row1 = row2 = row3 = row4 = row
    col1 = col2 = col3 = col4 = col
    for i in range(0,N):
                    if row1 - 1 >= 0 and col1 + 1 < N:
                        if board[row1 - 1][col1 + 1] == 1:
                            sum += 1
                        row1 = row1 - 1
                        col1 = col1 + 1
                                                        
                    if row2 + 1 < N and col2 - 1 >= 0:
                        if board[row2 + 1][col2 - 1] == 1:
                            sum += 1
                        row2 = row2 + 1
                        col2 = col2 - 1
                            
                    if row3 + 1 < N and col3 + 1 < N:
                        if board[row3 + 1][col3 + 1] == 1:
                            sum += 1
                        row3 = row3 + 1
                        col3 = col3 + 1
                        
                    if row4 - 1 >= 0  and col4 - 1 >= 0:
                        if board[row4 - 1][col4 - 1] == 1:
                            sum += 1
                        row4 = row4 - 1
                        col4 = col4 - 1
                        
                    if sum > 0:
                        return True
    return False                 


# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    if problem_type == 'nrook':
        return "\n".join([ " ".join([ "X" if row==row_coord and col==col_coord else "R" if board[row][col]== 1 else "_" for col in range(N)]) for row in range(N)])

    if problem_type == 'nqueen':
        return "\n".join([ " ".join([ "X" if row==row_coord and col==col_coord else "Q" if board[row][col]== 1 else "_" for col in range(N) ]) for row in range(N)] )

    
# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece2(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

def successors2(board):
    succ = []
    for row in range(0,N):
        for col in range(0,N):

            if 1 in board[row] or count_on_col(board,col) >= 1 or board[row][col] == 1 :
                continue
    
            if  row == row_coord and col == col_coord:
                continue

            if problem_type == 'nqueen':
                if count_on_diagonal(board,row,col): 
                    continue 
            
            succ.append(add_piece2(board, row, col))
    return succ


# Check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N 

# Solve n-rooks!

def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors2( fringe.pop() ):
                        if is_goal(s):            
                            return (s)
                        fringe.append(s)                    		
    return False

# This is N, the size of the board. It is passed through command line arguments.
problem_type = sys.argv[1].lower()
N = int(sys.argv[2])
row_coord = int(sys.argv[3]) - 1
col_coord = int(sys.argv[4]) - 1


# Check for proper problem type    
# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N for i in range(N)]
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solve(initial_board)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")




