from random import randint, random, choices, sample
import Graph
from SortingNetwork import SortingNetwork
from SNVector import SNVector
''' MORE COMMENTS AND EXPLANATIONS IN FUNC -> |RUN| '''
GROUP_SIZE = 5000


class GenStruct:
    def __init__(self, N, arrSize):
        self.arr = []
        for _ in range(arrSize):
            self.arr.append(randint(1, N))
        self.fitness = -1


class GeneticAlgo:
    def __init__(self, N, minCN, maxCN, maxIter):
        self.popSize = 2048
        self.population = []
        self.vectors = []
        self.buffer = []
        self.N = N
        self.maxIter = maxIter
        self.minCN = minCN
        self.mutation = random() * 0.25
        self.arrSize = minCN
        self.maxCN = maxCN
        self.initPopulation()
        self.initVec()

    def initPopulation(self):
        for _ in range(self.popSize):
            member = SortingNetwork(self.N, self.minCN, self.maxCN)
            self.population.append(member)
            self.buffer.append(member)

    def initVec(self):
        for i in range(2 ** self.N):
            bnumber = int(("{:0%db}" % self.N).format(i))  # create binary number
            bnumber = '{numbers:0{width}d}'.format(width=self.N, numbers=bnumber)
            member = SNVector(self.N, list(bnumber))
            self.vectors.append(member)
        print("VEC DONE!")

    def calcFitness(self):
        for i in range(self.popSize):
            self.population[i].evaluate()
            self.buffer[i].evaluate()
        for i in range(len(self.vectors)):
            self.vectors[i].evaluate()

    def mate(self):
        esize = self.popSize * 0.1
        for i in range(int(esize), self.popSize):
            i1 = randint(0, self.popSize / 2)
            i2 = randint(0, self.popSize / 2)
            spos = randint(0, self.arrSize - 1)
            if len(self.population[i1].str) > len(self.population[i2].str):
                self.buffer[i].str = self.population[i1].str[0:spos] + self.population[i2].str[spos:]
            else:
                self.buffer[i].str = self.population[i2].str[0:spos] + self.population[i1].str[spos:]

            if random() < self.mutation:
                self.mutate(i)
        self.swap()

    def mutate(self, i):
        i1 = randint(0, self.arrSize - 2)
        while i1 % 2 == 1:
            i1 = randint(0, self.arrSize - 2)
        i2 = randint(0, self.arrSize - 2)
        while i2 % 2 == 1:
            i2 = randint(0, self.arrSize - 2)

        tmp = self.buffer[i].str[i1]
        self.buffer[i].str[i1] = self.buffer[i].str[i2]
        self.buffer[i].str[i2] = tmp

        tmp = self.buffer[i].str[i1 + 1]
        self.buffer[i].str[i1 + 1] = self.buffer[i].str[i2 + 1]
        self.buffer[i].str[i2 + 1] = tmp

    def swap(self):
        temp = self.population
        self.population = self.buffer
        self.buffer = temp

    def printBest(self):
        print("SOL = ", self.population[0].str)
        print('FITNESS = ', self.population[0].fitness)
        print('ARRAY SIZE = ', len(self.population[0].str))
        print()

    def sortByFitness(self):
        self.population.sort(key=self.fitnessSort)
        self.vectors.sort(key=self.fitnessSort)

    def fitnessSort(self, x):
        return x.fitness

    def assign(self, itration):
        if self.N < 10:
            for i in range(self.popSize):
                self.population[i].vectors = self.vectors
                if i < len(self.vectors):
                    self.vectors[i].networks = self.population
        else:
            if itration == 0:
                for i in range(self.popSize):
                    self.population[i].vectors = []
                    vectorsArr = sample(range(len(self.vectors)), k=GROUP_SIZE)
                    for j in vectorsArr:
                        self.population[i].vectors.append(self.vectors[j])
                        self.vectors[j].networks.append(self.population[i])
            else:
                for i in range(self.popSize):
                    self.population[i].vectors = []
                    vectorsArr = sample(range(int(len(self.vectors) / 2)), k=GROUP_SIZE)
                    for j in vectorsArr:
                        self.population[i].vectors.append(self.vectors[j])
                        self.vectors[j].networks.append(self.population[i])

    def run(self):
        fitnessArray = []
        sizeArray = []
        for i in range(self.maxIter):
            # we assign a group of vectors for each member (in case of k=6 the group is all the vectors)
            self.assign(i)
            # we calc the fitness for each vector/member
            self.calcFitness()
            # we sort the vectors/population
            self.sortByFitness()
            print("ITRATION : ", i, '/', self.maxIter)
            self.printBest()
            # FOR THE GRAPH
            fitnessArray.append(self.population[0].fitness)
            sizeArray.append(len(self.population[0].str))
            if self.population[0].fitness == self.minCN:
                break
            self.mate()
        Graph.draw(fitnessArray)
        Graph.draw(sizeArray, "Array Size")
