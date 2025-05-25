#!/usr/bin/env python

import gamerules
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import your player
from player import Player


class RNGPlayer(gamerules.Player):
    """Random player for testing"""

    def __init__(self, name):
        super().__init__("RNG player " + name)
        self.rounds = 0

    def getAction(self, board, startValue):
        """All logic going here"""
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


# Main execution
if __name__ == "__main__":
    p1 = Player("Taras Demchyna", "weights.pkl")

    # If environment variable is set - it is remote session
    if os.environ.get("PLAYER1_REMOTE", "no") == "yes":
        p1.serve_via_std()
        sys.exit(0)

    # Local testing mode
    p2 = RNGPlayer("Test Opponent")

    board = gamerules.Board()
    gameFinished = False

    players = [p1, p2]
    startValue = {p1: 1, p2: -1}  # player p1 starts first
    p1.newGame(True)
    p2.newGame(True)

    while not gameFinished:
        for player in players:
            action = player.getAction(copy.deepcopy(board), startValue[player])
            print("Player " + player.getName() + " : " + str(action))
            possibleActions = board.getPossibleActions()

            if action in possibleActions:
                board.updateBoard(action, startValue[player])

                gameFinished = board.checkVictory(action, startValue[player])
                if gameFinished:
                    print(player.getName() + " won")
                    break

                if len(board.getPossibleActions()) == 0:
                    print("No possible actions left")
                    gameFinished = True
                    break
            else:
                print(player.getName() + " lost - invalid move")
                gameFinished = True
                break

    player1 = p1 if startValue[p1] == 1 else p2
    player2 = p2 if startValue[p1] == 1 else p1
    board.showBoard()
    board.plotBoard(player1, player2)