import random


class BaldStruct:
    def __init__(self, target_size):
        self.fitness = -1
        self.learning_fitness = -1
        self.str = ""
        self.init_str(target_size)
        self.attempts_num = 0
        self.success = False

    def init_str(self, target_size):
        for i in range(target_size):
            rand = random.randrange(0, 4)
            if rand == 0 or rand == 1:
                self.str += '?'
            if rand == 2:
                self.str += '0'
            if rand == 3:
                self.str += '1'

