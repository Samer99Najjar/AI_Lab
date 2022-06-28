from BinPackingStruct import BinPackingStruct
from random import randint, random, choice


class BinPacking:
    def __init__(self, c, POPSIZE, GA_ELITRATE, GA_MUTATION, maxSteps, items):
        self.C = c
        self.N = len(items)
        self.items = items
        self.maxSteps = maxSteps
        self.GA_POPSIZE = POPSIZE
        self.GA_ELITRATE = GA_ELITRATE
        self.GA_MUTATION = GA_MUTATION
        self.population = []
        self.buffer = []
        self.bestFound = BinPackingStruct()
        self.bestFound.updateN(c)
        self.init_population()

    def init_population(self):
        for i in range(self.GA_POPSIZE):
            citizen = BinPackingStruct()
            citizen.updateN(self.N)
            for j in range(self.N):
                citizen.arr[j] = randint(0, self.N - 1)

            self.population.append(citizen)
            self.buffer.append(citizen)

    def calc_fitness(self):
        # in Bin packing we choose to calculate fitness this way
        # first we go over the packs
        # if a pack has weight than allowed we give huge penalty
        # else the farther the pack from the capacity get more penalty
        # which means if the pack has the weight of capacity -1 we give
        # very little penalty
        for i in range(self.GA_POPSIZE):
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

    def fitness_sort(self, x):
        return x.fitness

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def elitism(self, population, buffer, esize):
        for i in range(esize):
            buffer[i].set_str(population[i].get_str())
            buffer[i].set_fitness(population[i].getfitness())
        return buffer

    def mate(self):
        esize = self.GA_POPSIZE * self.GA_ELITRATE
        tsize = self.N
        for i in range(int(esize), self.GA_POPSIZE):
            i1 = randint(0, self.GA_POPSIZE / 2)
            i2 = randint(0, self.GA_POPSIZE / 2)
            spos = randint(0, tsize - 1)
            for j in range(self.N):
                self.buffer[i].arr[j] = choice([self.population[i1].arr[j], self.population[i2].arr[j]])
            if random() < self.GA_MUTATION:
                self.inversion_mutation(i)

    def inversion_mutation(self, i):
        member = self.population[i].arr
        i1 = randint(0, self.N - 3)
        i2 = randint(i1 + 1, self.N - 2)
        i3 = randint(i2, self.N - 1)
        self.buffer[i].arr = member[0:i1] + member[i2:i3] + member[i1:i2][::-1] + member[i3:]

    def mutate(self, member):
        tsize = self.N
        ipos = randint(0, tsize - 1)
        delta = randint(0, self.N - 1)
        member.arr[ipos] = delta
        return member

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

    def start(self):
        self.bestFound.fitness = 20000000
        for index in range(self.maxSteps):
            self.calc_fitness()
            self.sort_by_fitness()
            self.update_best_found()
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
