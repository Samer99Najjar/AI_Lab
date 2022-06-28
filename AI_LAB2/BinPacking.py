import sys

import numpy

from BinPackingStruct import BinPackingStruct
from random import randint, random, choice, randrange


class BinPacking:
    def __init__(self, c, POPSIZE, GA_ELITRATE, GA_MUTATION, maxSteps, items):
        self.C = c
        self.N = len(items)
        self.items = items
        self.centers = []
        self.indeces = []
        self.classifier = []
        self.maxSteps = maxSteps
        self.GA_POPSIZE = POPSIZE
        self.GA_ELITRATE = GA_ELITRATE
        self.GA_MUTATION = GA_MUTATION
        self.population = []
        self.buffer = []
        self.bestFound = BinPackingStruct()
        self.bestFound.updateN(c)
        self.init_population()
        self.diversity = []
        self.species = []
        self.pmax = 0.75
        self.generation_number = 0
        self.mutation_type = 'Adaptive'
        self.threshhold = 0.4
        self.k = 5

    def init_population(self):
        self.classifier = []
        for i in range(self.GA_POPSIZE):
            citizen = BinPackingStruct()
            citizen.updateN(self.N)
            for j in range(self.N):
                citizen.arr[j] = randint(0, self.N - 1)

            self.population.append(citizen)
            self.buffer.append(citizen)

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

    def distance(self, first, second):
        values1 = first.arr
        values2 = second.arr
        """Compute the Kendall tau distance."""
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = numpy.meshgrid(numpy.arange(n), numpy.arange(n))
        a = numpy.argsort(values1)
        b = numpy.argsort(values2)
        ndisordered = numpy.logical_or(numpy.logical_and(a[i] < a[j], b[i] > b[j]),
                                       numpy.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        # print("the distance is: ",ndisordered / (n * (n - 1)),"for indeces:",firstIndex,secondIndex)
        return ndisordered / (n * (n - 1))

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

    def calc_fitness(self):
        # in Bin packing we choose to calculate fitness this way
        # first we go over the packs
        # if a pack has weight than allowed we give huge penalty
        # else the farther the pack from the capacity get more penalty
        # which means if the pack has the weight of capacity -1 we give
        # very little penalty
        # total_diver=0
        for i in range(self.GA_POPSIZE):
            # self.genetic_diversity(i)
            # print(i)
            # total_diver+=self.population[i].diversity
            fitness = 0
            binPacks = []
            for j in range(self.N):
                sum = 0
                if self.population[i].arr[j] not in binPacks:
                    binPacks.append(self.population[i].arr[j])

                for k in range(self.N):
                    if self.population[i].arr[k] == j:
                        sum += self.items[k]

                if self.C >= sum:
                    fitness += self.C - sum
                else:
                    fitness += self.N * 10000
            fitness += len(binPacks) * 100
            self.population[i].binsNum = len(binPacks)
            self.population[i].fitness = fitness
        # print("Diversity is: ",total_diver/self.GA_POPSIZE)

    def fitness_sort(self, x):
        return x.fitness

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def elitism(self, population, buffer, esize):
        for i in range(esize):
            buffer[i].set_str(population[i].get_str())
            buffer[i].set_fitness(population[i].getfitness())
        return buffer

    def mutate(self, member):
        tsize = self.N
        ipos = randint(0, tsize - 1)
        delta = randint(0, self.N - 1)
        member.arr[ipos] = delta
        return member

    def random_walk(self, state, walksnumber):
        mystate = state
        for i in range(walksnumber):
            newstate = self.steepest_ascend(mystate)
            if newstate.fitness < mystate.fitness:
                mystate = newstate
        return mystate

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
            newstate = BinPackingStruct()
            newstate.arr = state.arr
            newstate.arr[i] = randint(0, self.N)
            nearstates.append(newstate)
        self.calc_fitness_states(nearstates)
        return nearstates

    def calc_fitness_states(self, nearstates):

        for i in range(len(nearstates)):
            fitness = 0
            binPacks = []
            for j in range(self.N):
                sum = 0
                if nearstates[i].arr[j] not in binPacks:
                    binPacks.append(nearstates[i].arr[j])

                for k in range(self.N):
                    if nearstates[i].arr[k] == j:
                        sum += self.items[k]

                if self.C >= sum:
                    fitness += self.C - sum
                else:
                    fitness += self.N * 10000
            fitness += len(binPacks) * 100
            nearstates[i].binsNum = len(binPacks)
            nearstates[i].fitness = fitness

    def random_immigrant(self):
        esize = self.GA_POPSIZE * 0.7
        tsize = self.N
        for i in range(int(esize), self.GA_POPSIZE):
            i1 = randint(0, int(esize))
            self.population[i] = self.population[i1]
            if random() < self.GA_MUTATION:
                self.inversion_mutation(i)

    def mate(self):
        esize = self.GA_POPSIZE * self.GA_ELITRATE
        tsize = self.N
        for i in range(int(esize), self.GA_POPSIZE):
            i1 = randint(0, self.GA_POPSIZE / 2)
            i2 = randint(0, self.GA_POPSIZE / 2)
            while self.population[i1].classy == self.population[i2].classy:
                i2 = randint(0, self.GA_POPSIZE / 2)
            self.distance(self.population[i1], self.population[i2])
            spos = randint(0, tsize - 1)
            for j in range(self.N):
                self.buffer[i].arr[j] = choice([self.population[i1].arr[j], self.population[i2].arr[j]])

            if self.mutation_type == 'Adaptive':
                self.pmax = self.pmax * self.pmax
                adaptive = 2 * self.pmax * self.pmax * pow(numpy.exp(1), self.generation_number * 0.5) / (
                        self.pmax + self.pmax * pow(numpy.exp(1), self.generation_number * 0.5))
                if random() < adaptive * self.GA_MUTATION:
                    self.inversion_mutation(i)

    def inversion_mutation(self, i):
        member = self.population[i].arr
        i1 = randint(0, self.N - 3)
        i2 = randint(i1 + 1, self.N - 2)
        i3 = randint(i2, self.N - 1)
        self.buffer[i].arr = member[0:i1] + member[i2:i3] + member[i1:i2][::-1] + member[i3:]

    def print_best(self):
        print("Best :", self.population[0].arr, " (", self.population[0].fitness, ")  \nnum of packs : ",
              self.population[0].binsNum)

    def swap(self):
        temp = self.population
        self.population = self.buffer
        self.buffer = temp

    def print_topbest(self):
        for i in range(5):
            print(i, ". ", self.population[i].arr, " --> ", self.population[i].fitness)

    def k_gene_exchange(self):

        child = BinPackingStruct()
        child.updateN(self.N)
        for i in range(self.N):
            child.arr[i] = self.population[randint(0, self.GA_POPSIZE / 2)].arr[i]

        fitness = 0
        binPacks = []
        for j in range(self.N):
            sum = 0
            if child.arr[j] not in binPacks:
                binPacks.append(child.arr[j])

            for k in range(self.N):
                if child.arr[k] == j:
                    sum += self.items[k]

            if self.C >= sum:
                fitness += self.C - sum
            else:
                fitness += self.N * 10000
        fitness += len(binPacks) * 100
        child.binsNum = len(binPacks)
        child.fitness = fitness
        return child

    def start_hill(self):
        self.bestFound.fitness = 20000000
        self.calc_fitness()
        for index in range(self.maxSteps):
            for i in range(self.GA_POPSIZE):
                self.population[i] = self.k_gene_exchange()

            self.sort_by_fitness()
            self.print_best()
            if self.population[0].fitness == 0:
                break



def start(self):
    self.bestFound.fitness = 20000000
    for index in range(self.maxSteps):
        self.calc_fitness()
        self.sort_by_fitness()
        self.update_best_found()
        # self.selection_pressure()
        # self.species = []
        # self.random_immigrant()
        # self.Threshhold_spec()
        # self.clustering_spec()
        # self.generation_number = index
        # self.print_best()
        if self.population[0].fitness == 0:
            break
        self.mate()
        self.swap()
        print(index + 1, "/", self.maxSteps)
    print("\nBEST FOUND IN", self.maxSteps, "STEPS:\n", self.bestFound.arr, " (", self.bestFound.fitness,
          ")  \nnum of packs : ", self.bestFound.binsNum)


def update_best_found(self):
    if self.population[0].fitness < self.bestFound.fitness:
        self.bestFound.arr = self.population[0].arr
        self.bestFound.fitness = self.population[0].fitness
        self.bestFound.binsNum = self.population[0].binsNum
    else:
        self.population[0].arr = self.bestFound.arr
        self.population[0].fitness = self.bestFound.fitness
        self.population[0].binsNum = self.bestFound.binsNum
