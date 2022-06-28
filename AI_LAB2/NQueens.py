import math

import numpy

from GAStruct import GAStruct
from random import random, shuffle, randint, choice


class NQueens:

    def __init__(self, N, POPSIZE, MAXITER, ELITRATE, MUTATION, pmx, mutateType):
        self.N = N
        self.GA_MUTATION = MUTATION
        self.PMX = pmx
        self.population = []
        self.nextPopulation = []
        self.susPopulation = []
        self.susInitArray = True
        self.susIndex = 0
        self.mutateType = mutateType
        self.numElitation = POPSIZE * ELITRATE
        self.GA_ELITRATE = ELITRATE
        self.GA_POPSIZE = POPSIZE
        self.GA_MAXITER = MAXITER
        self.diversity = []
        self.species = []
        self.pmax = 0.75
        self.generation_number = 0
        self.threshhold = 0.4
        self.k = 5

        self.init_population()

    def init_population(self):
        for j in range(self.GA_POPSIZE):
            randomStr = [i for i in range(0, self.N)]
            randomStr1 = [i for i in range(0, self.N)]

            shuffle(randomStr)
            shuffle(randomStr1)
            member1 = GAStruct(self.N, randomStr)
            member2 = GAStruct(self.N, randomStr1)
            self.population.append(member1)
            self.nextPopulation.append(member2)

    def genetic_diversity(self, index):
        l = 0
        for j in range(self.GA_POPSIZE):
            l += self.distance(self.population[index], self.population[j])
        self.population[index].diversity = l

    def selection_pressure(self):
        best_fit_number = 0
        best_fitness = self.population[0].fitness
        for i in range(self.GA_POPSIZE):
            if self.population[i].fitness == best_fitness:
                best_fit_number += 1
        print("selection pressure is: ", best_fit_number / self.GA_POPSIZE)

        return best_fit_number / self.GA_POPSIZE

    def calc_new_centers(self):
        for i in range(self.k):
            if len(self.classifier[i]) > 0:
                self.centers[i] = self.classifier[i][randint(0, len(self.classifier[i]) - 1)]
            else:
                self.centers[i] = self.population[i]
            self.classifier[i] = []

    def clustering_spec(self):
        self.classifier = []
        self.centers = []
        for i in range(self.k):
            self.classifier.append([])
            self.centers.append(self.population[i])
        for i in range(self.maxSteps):
            for j in range(self.GA_POPSIZE):
                dis = []
                for citizen in self.centers:
                    dis.append(self.distance(self.population[j], citizen))
                index = dis.index(min(dis))
                self.classifier[index].append(self.population[j])
                self.population[j].classy = index
            self.calc_new_centers()

    def Threshhold_spec(self):
        numberof_species = -1
        for i in range(self.GA_POPSIZE):
            flag = 0
            for j in range(len(self.species)):
                counter = 0
                for k in range(len(self.species[j])):
                    destence = self.distance(self.population[i], self.species[j][k])
                    if destence < self.threshhold:
                        counter += 1
                if counter == len(self.species[j]):
                    flag = 1
                    self.population[i].specynum = j
                    self.species[j].append(self.population[i])
                    break
            if flag == 0:
                new_array = []
                self.population[i].specynum = len(self.species)
                new_array.append(self.population[i])
                self.species.append(new_array)
                numberof_species += 1

    def distance(self, firstIndex, secondIndex):
        # values1 = self.population[firstIndex]
        # values2 = self.population[secondIndex]
        """Compute the Kendall tau distance."""
        values1 = firstIndex.NQueens
        values2 = secondIndex.NQueens
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = numpy.meshgrid(numpy.arange(n), numpy.arange(n))
        a = numpy.argsort(values1)
        b = numpy.argsort(values2)
        ndisordered = numpy.logical_or(numpy.logical_and(a[i] < a[j], b[i] > b[j]),
                                       numpy.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        # print(ndisordered / (n * (n - 1)))
        return ndisordered / (n * (n - 1))

    def fitness_sort(self, x):
        return x.get_fitness()

    def hill_climb(self, state):
        if state.fitness == 0:
            return state
        nearstates = self.calc_near_states(state)
        for city in nearstates:
            if city.fitness < state.fitness:
                state = city
                return state
            if state.fitness == 0:
                return state
        return state

    def random_walk(self, state, walksnumber):
        mystate = state
        for i in range(walksnumber):
            newstate = self.steepest_ascend(mystate)
            if newstate.fitness < mystate.fitness:
                mystate = newstate
        return mystate

    def k_gene_exchange(self):
        arr = []
        for i in range(self.N):
            arr.append(0)
        child = GAStruct(self.N, arr)
        for i in range(self.N):
            child.NQueens[i] = self.population[randint(0, self.GA_POPSIZE / 2)].NQueens[i]

        # print(child.NQueens)

        fitness = 0
        for j in range(self.N):
            fitness += self.calc_conflict(child.NQueens, j)
        child.fitness = fitness / 2
        return child

    def steepest_ascend(self, state):
        if state.fitness == 0:
            return state
        nearstates = self.calc_near_states(state)
        for city in nearstates:
            if city.fitness < state.fitness:
                state = city
            if state.fitness == 0:
                return state
        return state

    def calc_near_states(self, state):
        nearstates = []
        for i in range(self.N):
            for j in range(i):
                newArr = []
                for k in range(self.N):
                    newArr.append(state.NQueens[k])
                newstate = GAStruct(self.N, newArr)
                tmp = newstate.NQueens[i]
                newstate.NQueens[i] = newstate.NQueens[j]
                newstate.NQueens[j] = tmp
                nearstates.append(newstate)
        self.calc_fitness_states(nearstates)
        return nearstates

    def calc_fitness_states(self, nearstates):

        for i in range(len(nearstates)):
            fitness = 0
            for j in range(self.N):
                fitness += self.calc_conflict(nearstates[i].NQueens, j)
            nearstates[i].fitness = fitness / 2
            # print(nearstates[i].fitness)

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def calc_conflict(self, NQueens, j):
        # this functions counts how many queens are in conflict
        # it takes the board and the place of queen and check how many conflicts with this queen
        row = NQueens[j]
        col = j
        conflictNum = 0
        for i, k in zip(range(row), range(col)):
            if NQueens[k] == i:
                conflictNum += 1

        for i, k in zip(range(row + 1, self.N), range(col)):
            if NQueens[col - 1 - k] == i:
                conflictNum += 1

        for i, k in zip(range(row), range(col + 1, self.N)):
            if NQueens[k] == row - 1 - i:
                conflictNum += 1

        for i, k in zip(range(row + 1, self.N), range(col + 1, self.N)):
            if NQueens[k] == i:
                conflictNum += 1

        for i in range(self.N):
            if NQueens[i] == row and i != col:
                conflictNum += 1
        return conflictNum

    def calc_fitness(self):
        # here we choose to calculate the fitness
        # based on how many conflicts there is in the board
        # total_diver = 0
        for i in range(self.GA_POPSIZE):
            # self.genetic_diversity(i)
            # total_diver += self.population[i].diversity
            fitness = 0
            for j in range(self.N):
                fitness += self.calc_conflict(self.population[i].NQueens, j)
            self.population[i].fitness = fitness / 2
        # print("Diversity is: ", total_diver / self.GA_POPSIZE)

    def random_immigrant(self):
        esize = self.GA_POPSIZE * 0.7
        tsize = self.N
        for i in range(int(esize), self.GA_POPSIZE):
            i1 = randint(0, int(esize))
            self.population[i] = self.population[i1]
            if random() < self.GA_MUTATION:
                self.inversion_mutation(i)

    def linear_scaling(self, fitness, a=0.5, b=0):
        return a * fitness + b

    def roulette_spin(self, new_fitnesses):
        fitness_sum = sum(new_fitnesses)
        prob = [fitness / fitness_sum for fitness in new_fitnesses]
        return choice(len(new_fitnesses), p=prob)

    def RWS(self, population):
        new_fitnesses = [1 / self.linear_scaling(member.fitness + 1) for member in population]
        selection = self.roulette_spin(new_fitnesses)
        return selection

    def SUS(self, population):
        F = 0
        N = 360
        for i in range(self.GA_POPSIZE):
            F += population[i].fitness
        fitnessBarometer = F / N
        sqrtFit = 0
        start = randint(0, self.GA_MAXITER) % fitnessBarometer
        ptr = math.sqrt(random() * fitnessBarometer)
        for i in range(int(start), self.GA_POPSIZE):
            sqrtFit += i * fitnessBarometer
            if sqrtFit >= ptr:
                return i
        return -1

    def tournament(self, population, k):
        best = None
        for _ in range(k):
            member = population[randint(0, len(population) - 1)]
            if best == None:
                best = member
            elif best.fitness > member.fitness:
                best = member
        return best

    def print_best(self):
        print("Best :", end=' ')
        for i in range(self.N):
            print(self.population[0].NQueens[i], end=" ")
        print(" ||  Fitness :", self.population[0].fitness)

    def elitism(self):
        for i in range(self.numElitation):
            self.nextPopulation[i].fitness = self.population[i].fitness
            for j in range(self.N):
                self.nextPopulation[i].NQueens[j] = self.population[i].NQueens[j]

    def pmx(self):
        for i in range(int(self.numElitation), self.GA_POPSIZE):
            i1 = randint(0, self.GA_POPSIZE / 2)
            i2 = randint(0, self.GA_POPSIZE / 2)
            i3 = randint(0, self.N - 1)
            secondCitizenQueen = self.population[i2].NQueens[i3]
            for j in range(1, self.N):
                self.nextPopulation[i].NQueens[j] = self.population[i1].NQueens[j]
            for j in range(self.N):
                if self.population[i1].NQueens[j] == secondCitizenQueen:
                    self.nextPopulation[i].NQueens[j] = self.population[i1].NQueens[i3]
                    self.nextPopulation[i].NQueens[i3] = self.population[i1].NQueens[j]
                    break
            if self.mutateType == 0:
                if random() < self.GA_MUTATION:
                    self.inversion_mutation(i)
            else:
                if random() < self.GA_MUTATION:
                    self.random_mutation(i)

    def findFirstIndex(self, indicesArray):
        if len(indicesArray) == 0:
            return 0
        i = 1
        while i < self.N:
            if i not in indicesArray:
                return i
            i += 1
        return i

    def cx(self):
        # here we start from parent 1 then do the cycle and keep coping from parent 1
        # when the cycle is finished we do another cycle and copy from parent 2
        # keep on doing this 2 things until we moved over all indeces
        for i in range(int(self.numElitation), self.GA_POPSIZE):
            i1 = randint(0, self.GA_POPSIZE - 1)
            i2 = randint(0, self.GA_POPSIZE - 1)
            parent1 = self.population[i1].NQueens
            parent2 = self.population[i2].NQueens
            indicesArray = []
            child = []
            odd = False
            while len(indicesArray) < self.N:
                firstIndex = self.findFirstIndex(indicesArray)
                index = firstIndex
                while True and firstIndex < self.N:
                    indicesArray.append(index)
                    if odd:
                        child.append(parent1[index])
                        index = parent1.index(parent2[index])
                    else:
                        child.append(parent2[index])
                        index = parent1.index(parent2[index])
                    if index == firstIndex:
                        break
                odd = not odd
            self.nextPopulation[i].NQueens = child
            if self.mutateType == 0:
                if random() < self.GA_MUTATION:
                    self.inversion_mutation(i)
            else:
                if random() < self.GA_MUTATION:
                    self.random_mutation(i)

    def inversion_mutation(self, i):
        # choose 3 a part of string reverse it
        # put it back in random place
        member = self.population[i].NQueens
        i1 = randint(0, self.N - 3)
        i2 = randint(i1 + 1, self.N - 2)
        i3 = randint(i2, self.N - 1)
        self.population[i].NQueens = member[0:i1] + member[i2:i3] + member[i1:i2][::-1] + member[i3:]

    def random_mutation(self, i):
        # take part of the array shuffle it put it back
        member = self.population[i].NQueens
        i1 = randint(0, self.N - 3)
        i2 = randint(i1 + 1, self.N - 2)
        newHdak = member[i1:i2]
        shuffle(newHdak)
        self.population[i].NQueens = member[0:i1] + newHdak + member[i2:]

    def mate(self):
        if self.PMX:
            self.pmx()
        else:
            self.cx()

    def swap(self):
        temp = self.population
        self.population = self.nextPopulation
        self.nextPopulation = temp

    def part2_run(self, type):
        self.calc_fitness()
        for index in range(self.GA_MAXITER):
            for i in range(self.GA_POPSIZE):
                if type == 1:
                    self.population[i] = self.hill_climb(self.population[i])
                if type == 2:
                    self.population[i] = self.steepest_ascend(self.population[i])
                if type == 3:
                    self.population[i] = self.random_walk(self.population[i], 10)
                else:
                    self.population[i] = self.k_gene_exchange()

            self.sort_by_fitness()
            self.print_best()
            if self.population[0].fitness == 0:
                break

    def run(self):
        for index in range(self.GA_MAXITER):
            self.calc_fitness()
            self.sort_by_fitness()
            self.print_best()
            self.selection_pressure()
            if self.population[0].fitness == 0:
                break

            self.mate()
            self.swap()
