import statistics

import Graph
from Dummies import *
from Agent import *
from random import uniform
from math import *
from Experts import *


class Tournament:
    def __init__(self):
        self.myarr = []
        for i in range(1000):
            self.myarr.append(i)
        self.maxIter = 1000
        self.participants = []
        self.score = []
        self.finalScore = []
        self.population = []
        self.buffer = []
        self.popsize = 30
        self.addPlayers()
        self.matchHistory = []
        self.start()
        self.init_population()
        self.run()
        self.print_sort()

    def init_population(self):
        for i in range(self.popsize):
            Citizen = Agent()
            self.population.append(Citizen)
            self.buffer.append(Citizen)

    def Agents_game(self):
        for i in range(self.popsize):
            finalscore = 0
            for player in range(len(self.participants)):
                finalscore += self.startGame2(self.population[i], player)
            self.population[i].score = finalscore

    def calc_fitness(self):
        for i in range(self.popsize):
            self.population[i].fitness = self.population[i].score
            self.population[i].score = 0

    def addPlayers(self):
        self.participants.append(AntiFlat())
        self.participants.append(Copy())
        self.participants.append(Freq())
        self.participants.append(Flat())
        self.participants.append(Foxtrot())
        self.participants.append(Bruijn81())
        self.participants.append(Pi())
        self.participants.append(Play226())
        self.participants.append(RndPlayer())
        self.participants.append(Rotate())
        self.participants.append(Switch())
        self.participants.append(SwitchALot())
        #self.participants.append(Iocaine())
        self.participants.append(Greenberg())
        self.participants.append(Urza())
        self.participants.append(MegaHal())
        for _ in range(len(self.participants)):
            self.score.append(0)
            self.finalScore.append(0)

    def start(self):
        for player1 in range(len(self.participants) - 1):
            player1History = []
            for player2 in range(player1 + 1, len(self.participants)):
                finalScore = self.startGame(player1, player2)
                if finalScore == 1:
                    self.finalScore[player1] += 1
                    self.finalScore[player2] -= 1
                else:
                    self.finalScore[player2] += 1
                    self.finalScore[player1] -= 1

                player1History.append(finalScore)
            self.matchHistory.append(player1History)

    def startGame2(self, player1, player2):
        self.participants[player2].newGame(1000)
        score1, score2 = 0, 0
        for i in range(1000):
            move1 = player1.moves[i]
            move2 = self.participants[player2].nextMove()
            if (move1 == 0 and move2 == 1) or (move1 == 1 and move2 == 2) or (move1 == 2 and move2 == 0):
                score2 += 1
                score1 -= 1
                self.participants[player2].storeMove(move2, 1)

            elif (move2 == 0 and move1 == 1) or (move2 == 1 and move1 == 2) or (move2 == 2 and move1 == 0):
                score1 += 1
                score2 -= 1

                self.participants[player2].storeMove(move2, -1)
            else:
                self.participants[player2].storeMove(move2, 0)

        self.score[player2] += score2
        if score2 > score1:
            return -1

        return 1

    def startGame(self, player1, player2):
        self.participants[player1].newGame(1000)
        self.participants[player2].newGame(1000)
        score1, score2 = 0, 0
        for _ in range(1000):
            move1 = self.participants[player1].nextMove()
            move2 = self.participants[player2].nextMove()
            if (move1 == 0 and move2 == 1) or (move1 == 1 and move2 == 2) or (move1 == 2 and move2 == 0):
                score2 += 1
                score1 -= 1
                self.participants[player2].storeMove(move2, 1)
                self.participants[player1].storeMove(move1, -1)

            elif (move2 == 0 and move1 == 1) or (move2 == 1 and move1 == 2) or (move2 == 2 and move1 == 0):
                score1 += 1
                score2 -= 1
                self.participants[player1].storeMove(move1, 1)
                self.participants[player2].storeMove(move2, -1)
            else:
                self.participants[player1].storeMove(move1, 0)
                self.participants[player2].storeMove(move2, 0)

        self.score[player1] += score1
        self.score[player2] += score2

        if score2 > score1:
            return -1

        return 1

    def mutualism_phase(self, best):
        for i in range(self.popsize):
            index = randint(0, self.popsize - 1)
            while index == i:
                index = randint(0, self.popsize - 1)
            BF1 = randint(1, 2)
            BF2 = randint(1, 2)
            mutual_vector = []
            for j in range(len(self.population[i].moves)):
                num = int((self.population[i].moves[j] + self.population[index].moves[j]) / 2)
                mutual_vector.append(num)
            newvec1, newvec2 = [], []

            for j in range(len(self.population[i].moves)):
                num1 = abs(int(self.population[i].moves[j] + uniform(0, 1) * (best.moves[j] - mutual_vector[j] * BF1)))
                num2 = abs(
                    int(self.population[index].moves[j] + uniform(0, 1) * (best.moves[j] - mutual_vector[j] * BF2)))
                newvec1.append(num1 % 3)
                newvec2.append(num2 % 3)

            finalscore = 0
            my_agent = Agent()
            my_agent.moves = newvec1
            for player in range(len(self.participants)):
                finalscore += self.startGame2(my_agent, player)
            if finalscore > self.population[i].fitness:
                self.population[i].moves = newvec1
                self.population[i].fitness = finalscore

            finalscore = 0
            my_agent2 = Agent()
            my_agent2.moves = newvec2
            for player in range(len(self.participants)):
                finalscore += self.startGame2(my_agent2, player)
            if finalscore > self.population[index].fitness:
                self.population[i].moves = newvec1
                self.population[i].fitness = finalscore

    def Commensalism_phase(self, best):
        for i in range(self.popsize):
            index = randint(0, self.popsize - 1)
            while index == i:
                index = randint(0, self.popsize - 1)
            newvec = []
            for j in range(len(self.population[i].moves)):
                num = abs(int(self.population[i].moves[j] + random.choice([-1, 1]) * (
                        best.moves[j] - self.population[index].moves[j])))
                newvec.append(num % 3)
            my_agent = Agent()
            my_agent.moves = newvec
            finalscore = 0
            for player in range(len(self.participants)):
                finalscore += self.startGame2(my_agent, player)
            if finalscore > self.population[i].fitness:
                self.population[i].moves = newvec
                self.population[i].fitness = finalscore

    def paratisim_phase(self):
        for i in range(self.popsize):
            index = randint(0, self.popsize - 1)
            while index == i:
                index = randint(0, self.popsize - 1)
            sampled_list = random.sample(self.myarr, 50)
            my_agent = Agent()
            my_agent.moves = self.population[index].moves
            for num in sampled_list:
                my_agent.moves[num] = randint(0, 2)
            finalscore = 0
            for player in range(len(self.participants)):
                finalscore += self.startGame2(my_agent, player)
            if finalscore > self.population[i].fitness:
                self.population[i].moves = my_agent.moves
                self.population[i].fitness = finalscore

    def fitness_sort(self, x):
        return x.fitness

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort, reverse=True)

    def print_best(self):
        print("sol = ", self.population[0].moves)
        print('Fitness = ', self.population[0].fitness)
        print()

    def run(self):
        self.Agents_game()
        self.calc_fitness()
        self.sort_by_fitness()
        best = self.population[0]
        for i in range(self.maxIter):
            self.sort_by_fitness()
            self.print_best()
            if self.population[0].fitness > best.fitness:
                best = self.population[0]
                if best.fitness == len(self.participants):
                    break
            self.mutualism_phase(best)
            self.Commensalism_phase(best)
            self.paratisim_phase()

        self.finalGame(best)

    def finalGame(self, player1):
        print("My Agent results:")
        avgs = []
        for player2 in range(len(self.participants)):
            scores = []
            for k in range(10):
                score1, score2 = 0, 0
                draw = 0
                self.participants[player2].newGame(1000)
                for i in range(1000):
                    move1 = player1.moves[i]
                    move2 = self.participants[player2].nextMove()
                    if (move1 == 0 and move2 == 1) or (move1 == 1 and move2 == 2) or (move1 == 2 and move2 == 0):
                        score2 += 1
                        self.participants[player2].storeMove(move2, 1)
                    elif (move2 == 0 and move1 == 1) or (move2 == 1 and move1 == 2) or (move2 == 2 and move1 == 0):
                        score1 += 1
                        self.participants[player2].storeMove(move2, -1)
                    else:
                        draw += 1
                        self.participants[player2].storeMove(move2, 0)
                # print(score1, draw, 1000 - draw - score1)
                scores.append(score1 + draw)
            avg = 0
            for score in scores:
                avg += score
            avg /= len(scores)
            avgs.append(avg)
            std = statistics.stdev(scores)
            print("Opp :", self.participants[player2].getName(), "||  Win Avg :", avg, "||  Win Stdev :", std)
            Graph.draw(scores, self.participants[player2].getName())
        print()

    def print_sort(self):
        seen = []
        print("My-Agent Score :", len(self.finalScore))
        for i in range(len(self.participants)):
            max = -200
            index = -1
            for j in range(len(self.participants)):
                if self.finalScore[j] > max and j not in seen:
                    max = self.finalScore[j]
                    index = j
            print(self.participants[index].getName(), " Score :", self.finalScore[index])
            seen.append(index)
