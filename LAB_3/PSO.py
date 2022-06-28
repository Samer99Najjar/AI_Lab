from random import randint, random
import time
import copy
import functools
import operator

import numpy

MIN_ASCII = 32
MAX_ASCII = 122
popsize = 2048
elite_rate = 0.1
mutation = random() * 0.25
maxIter = 3000
iters = 3
runs=3


def init_sol(problem):  # TSP: nearest neighbor heuristic
    size = problem.size
    array = []
    city = randint(1, size)
    array.append(city)
    dictionary = {city: True}
    index = size
    index -= 1
    while index > 0:
        distanceArray = problem.distanceMatrix[city]
        minCity = 1
        minDistance = float('inf')
        for i in range(1, len(distanceArray)):
            distance = distanceArray[i]
            if 0 < distance < minDistance and not dictionary.get(i, False):
                minDistance = distance
                minCity = i
        array.append(minCity)
        dictionary[minCity] = True
        city = minCity
        index -= 1
    return array


class GAStruct:
    def __init__(self, string, str2, fitness):
        self.str = string
        self.fitness = fitness
        self.selfBest = []
        self.velocity = str2
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


class PSO:

    def __init__(self, CVRP):
        self.population = []
        self.CVRP = CVRP
        self.W = 1
        self.C1 = 1
        self.C2 = 3
        self.init_population()

    def init_population(self):
        for i in range(popsize):
            randStr1 = init_sol(self.CVRP)
            randStr2 = init_sol(self.CVRP)
            member1 = GAStruct(randStr1, randStr2, 0)
            self.population.append(member1)

    def calc_fitness(self):
        for i in range(popsize):
            fitness, path = self.CVRP.calcPathCost(self.population[i].str)
            arr = self.population[i].str
            for j in range(len(arr)):
                if j + 1 not in arr:
                    fitness += 1000
            self.population[i].fitness = fitness
            if self.population[i].personalbestfittnes == -1 or self.population[i].personalbestfittnes > fitness:
                self.population[i].personalbestfittnes = fitness
                self.population[i].selfBest = self.population[i].str



    def fitness_sort(self, x):
        return x.get_fitness()

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def update_parameters(self, t, N):  # update the PSO parameters
        self.W = 0.5
        self.C1 = -3 * (t / N) + 3.5
        self.C2 = 3 * (t / N) + 0.5

    def print_best(self, global_best, fit):
        print("Best: ", self.population[0].str, " (",self.population[0].fitness, ")")

    def run(self):
        self.calc_fitness()
        self.sort_by_fitness()

        myfitness = self.population[0].fitness
        global_best = self.population[0].str
        start = time.time()

        for index in range(int(maxIter)):

            self.update_parameters(index, maxIter)
            self.calc_fitness()
            self.sort_by_fitness()
            self.print_best(global_best, myfitness)

            if self.population[0].fitness < myfitness:
                global_best = self.population[0].str
                myfitness = self.population[0].fitness
            self.CVRP.bestFitness = myfitness
            self.CVRP.best = global_best

            for j in range(popsize):
                # in this loop we walk over all the parcials
                # we have POPSIZE parcials in our implementation
                arr1 = []
                arr2 = []
                rand1 = random()
                rand2 = random()
                for k in range(self.CVRP.size):
                    # here we calculate the new string for each parcial
                    # and update the velocity and position of it
                    # using the formulas we saw in the lecture
                    num1 = rand1 * self.C1 * (self.population[j].selfBest[k] - self.population[j].str[k])
                    num2 = rand2 * self.C2 * (global_best[k] - self.population[j].str[k])
                    num3 = self.W * (self.population[j].velocity[k])
                    num4 = num1 + num2 + num3
                    num6=(int(num4) % self.CVRP.size) +1
                    #while num6 in arr1:
                        #num6= randint(1,self.CVRP.size)
                    arr1.append(num6)

                    num5 = (arr1[k] + self.population[j].str[k])%self.CVRP.size+1
                    #while num5 in arr2:
                     ##   num5= randint(1,self.CVRP.size)
                    arr2.append(num5)

                self.population[j].velocity = arr1
                self.population[j].str = arr2
                #print(arr2)


    def cooperative_pso(self):
        global_best =  float('inf')
        self.init_population()
        self.calc_fitness()
        self.sort_by_fitness()
        for i in range(iters):

            local_array = []
            for j in range(runs):
                local_array.append((self.pso_run()))
            self.update_parameters(iters,runs)

            local_best = self.cross_over(local_array)

            local_fitness= self.calc_fitness(local_best)
            if local_fitness < global_best:
                self.CVRP.best = copy.deepcopy(local_best)
                global_best = local_fitness
                self.CVRP.bestFitness=local_fitness

