import time
from Greenberg import *
from RoshamboPlayer import RoshamboPlayer
from iocaine import iocaine_agent
from Urza import *
from Megahal import *

class Observation:
    def __init__(self):
        self.step = 0
        self.lastOpponentAction = 0


class Iocaine(RoshamboPlayer):
    def __init__(self):
        super().__init__()
        self.observe = Observation()

    def newGame(self, trial):
        self.observe = Observation()

    def storeMove(self, move, score):
        self.observe.step += 1
        self.observe.lastOpponentAction = move

    def nextMove(self):
        return iocaine_agent(self.observe, 0)

    def getName(self):
        return "iocaine"

    def getAuthor(self):
        return "Robby"


class Greenberg(RoshamboPlayer):
    def __init__(self):
        super().__init__()
        self.opp_moves = []
        self.moves = 0

    def newGame(self, trial):
        self.opp_moves = []
        self.moves = 0

    def storeMove(self, move, score):
        self.moves += 1

    def nextMove(self):
        return player(self.moves, self.opp_moves)

    def getName(self):
        return "Greenberg"

    def getAuthor(self):
        return "Samer"


class Urza(RoshamboPlayer):
    def __init__(self):
        super().__init__()
        self.observe = Observation()

    def newGame(self, trial):
        self.observe = Observation()

    def storeMove(self, move, score):
        self.observe.step += 1
        self.observe.lastOpponentAction = move

    def nextMove(self):
        return urza_agent(self.observe, 0)

    def getName(self):
        return "Urza"

    def getAuthor(self):
        return "Urza Auther"


class MegaHal(RoshamboPlayer):
    def __init__(self):
        super().__init__()
        self.observe = Observation()

    def newGame(self, trial):
        self.observe = Observation()

    def storeMove(self, move, score):
        self.observe.step += 1
        self.observe.lastOpponentAction = move

    def nextMove(self):
        return megahal_agent(self.observe, 0)

    def getName(self):
        return "MegaHal"

    def getAuthor(self):
        return "Mega"
