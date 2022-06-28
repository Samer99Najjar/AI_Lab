import random

import numpy.random

import matplotlib.pyplot as plt

from BaldStruct import BaldStruct


class Baldwin:
    def __init__(self, target_size):
        self.target = ""
        self.target_size = target_size
        self.popsize = 1000
        self.iter = 1000
        self.fitness_sum = 0
        self.population = []
        self.buffer = []
        self.correct = 0
        self.incorrect = 0
        self.learnedbits = 0
        self.intensity = 0
        self.local_search_algo = ''

    def init_target(self):
        for i in range(self.target_size):
            self.target += str(random.randrange(0, 2))
        print("Target --> ", self.target)

    def init_population(self):
        for i in range(self.popsize):
            citizen = BaldStruct(self.target_size)
            self.population.append(citizen)
            self.buffer.append(citizen)

    def local_search(self):
        self.correct = 0
        self.incorrect = 0
        for i in range(self.iter):
            for j in range(self.popsize):
                if not self.population[j].success:
                    new_str = ""
                    for k in range(len(self.population[i].str)):
                        c = self.population[j].str[k]
                        if c == '?':
                            new_str += str(random.randrange(0, 2))
                        else:
                            new_str += c
                        if i == 0 and c != '?':
                            if c == self.target[k]:
                                self.correct += 1
                            else:
                                self.incorrect += 1

                    if new_str == self.target:
                        self.population[j].success = True
                        self.population[j].attempts_num = i + 1

    def calc_fitness(self):
        self.fitness_sum = 0
        for i in range(self.popsize):
            n = self.iter - self.population[i].attempts_num
            if not self.population[i].success:
                n = 0
            self.population[i].fitness = 1 + (19 * n / 1000)
            self.fitness_sum += self.population[i].fitness
            # if self.population[i].success:
            # print(i, "- Fitness: ", self.population[i].fitness, "Attempts: ", self.population[i].attempts_num)

            self.population[i].success = False

    def calc_frequency(self):
        sum = 0
        for i in range(self.popsize):
            sum += self.population[i].attempts_num
        sum = sum / self.popsize
        for i in range(self.popsize):
            self.population[i].frequency = self.population[i].attempts_num / sum

    def mate(self):
        prob_arr = []
        self.learnedbits = 0
        for i in range(self.popsize):
            prob_arr.append(self.population[i].fitness / self.fitness_sum)
        for i in range(self.popsize):
            i1 = numpy.random.choice(self.population, p=prob_arr)
            i2 = numpy.random.choice(self.population, p=prob_arr)
            myStr = ''
            for j in range(len(self.target)):
                spos = random.randint(0, 1)
                if spos == 0:
                    myStr += i1.str[j]
                    if self.population[i].str[j] == '?' and i1.str[j] != '?':
                        self.learnedbits += 1
                else:
                    myStr += i2.str[j]
                    if self.population[i].str[j] == '?' and i2.str[j] != '?':
                        self.learnedbits += 1

            self.buffer[i].str = myStr
        self.population = self.buffer

    def draw_hist(self, points1, points2, points3, arr4):
        range = (0, 100)
        bins = 10

        plt.plot(arr4, points1, label="correctpos")
        plt.plot(arr4, points2, label="correctpos")
        plt.plot(arr4, points3, label="correctpos")
        # x-axis label
        plt.xlabel('fitness value')
        # frequency label
        plt.ylabel('No. of fit val')
        # plot title
        plt.title('My histogram')

        # function to show the plot
        plt.show()

    def run(self):
        self.init_target()
        self.init_population()
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        for i in range(10):
            self.local_search()
            self.calc_fitness()

            avg1 = self.correct / (self.popsize * len(self.target))
            avg2 = self.incorrect / (self.popsize * len(self.target))
            avg3 = self.learnedbits / (self.popsize * len(self.target))
            arr1.append(avg1)
            arr2.append(avg2)
            arr3.append(avg3)
            arr4.append(i)
            print("CORRECT POSITION AVG:",avg1, "---  INCORRECT POSITION AVG:", avg2, "---  LEARNED BITS", avg3, "\n")
            self.mate()
        self.draw_hist(arr1, arr2, arr3, arr4)
