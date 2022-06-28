from random import randint


class GAStruct:

    def __int__(self, N, randomStr):
        self.N = N
        self.NQueens = randomStr
        self.fitness = 0
        self.age = 0

    def __init__(self, string, fitness):
        self.N = string
        self.NQueens = fitness
        self.fitness = 0
        self.age = 0
        self.specynumber=0
        self.specynum = 0
        self.diversity = 0

        self.selfBest = ""
        self.velocity = ""
        self.personalbestfittnes = -1
        self.score = 0
        self.age = 0

    def get_str(self):
        return self.str

    def get_fitness(self):
        return self.fitness

    def set_str(self, string):
        self.str = string

    def set_fitness(self, fitness):
        self.fitness = fitness

    def __gt__(self, other):
        return self.getFitness() >= other.getFitness()

    def __lt__(self, other):
        return self.getFitness() < other.getFitness()
