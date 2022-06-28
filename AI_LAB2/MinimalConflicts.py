import random
import time

MC_MAXITER = 16384


class MinimalConflict:

    def __init__(self, queens):
        self.board = random.sample(range(0, queens), queens)
        self.queens = queens

    def conflicts(self, col, row):
        count = 0
        for i in range(0, self.queens):
            if i == col:
                continue
            if self.board[i] == row or (abs(self.board[i] - row)) == abs(i - col):
                count += 1
        return count

    def all_conflicts(self):
        return [self.conflicts(i, self.board[i]) for i in range(0, self.queens)]

    def start(self):
        def random_position(li, filt):
            return random.choice([i for i in range(self.queens) if filt(li[i])])

        start = time.time()
        for i in range(MC_MAXITER):

            all_conflicts = self.all_conflicts()
            conflicts_sum = sum(all_conflicts)
            if conflicts_sum == 0:
                end = time.time()
                print("solution is:",self.board)
                print("time is : ", end - start)
                return self.board

            queen = random_position(all_conflicts, lambda flt: flt > 0)
            values = [self.conflicts(queen, i) for i in range(self.queens)]
            chosen_value = random_position(values, lambda flt: flt == min(values))
            self.board[queen] = chosen_value
            print (self.board)