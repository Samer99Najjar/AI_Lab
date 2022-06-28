from RoshamboPlayer import RoshamboPlayer
from random import randint

class Move:
    def __init__(self, move, score):
        self.move = move
        self.score = score


class Agent(RoshamboPlayer):
    def __init__(self):
        super().__init__()
        self.moves = []
        self.trial = 1000
        self.score=0
        self.fitness=0
        for i in range(1000):
            self.moves.append(randint(0,2))


    def storeMove(self, move, score):
        self.moves.append(Move(move, score))

    def nextMove(self):
        rsp = [0] * 3
        for move in self.moves:
            rsp[move.move] += 1

    def getName(self):
        return "My Agent"

    def getAuthor(self):
        return "MORSY & SAMER"
