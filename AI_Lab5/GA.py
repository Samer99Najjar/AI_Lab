import random
import numpy as np
from IPython.core.display_functions import clear_output
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import Graph

RELU = 'relu'
TANH = 'tanh'


class NeuralNetwork:
    def __init__(self, depth, hidden, activate):
        self.depth = depth
        self.hidden = hidden
        self.activateFunction = activate


class Citizen:
    def __init__(self):
        citizen_array = []
        depth = random.randint(1, 10)
        for i in range(depth):
            tmp = random.randint(2, 200)
            citizen_array.append(tmp)

        if random.randint(0, 2) == 0:
            activate = RELU
        else:
            activate = TANH

        self.network = NeuralNetwork(depth, citizen_array, activate)
        self.fitness = -1
        self.reg = 0


class GA:
    def __init__(self, popSize, maxIter, X_train, Y_train, X_test, Y_test):
        self.population = []
        self.buffer = []
        self.popSize = popSize
        self.eliteRate = 0.1
        self.maxIter = maxIter
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.initPopulation()

    def initPopulation(self):
        for _ in range(self.popSize):
            self.population.append(Citizen())
            self.buffer.append(Citizen())

    def calcFitness(self):
        for i in range(self.popSize):
            cls = MLPClassifier(hidden_layer_sizes=self.population[i].network.hidden, max_iter=69000,
                                activation=self.population[i].network.activateFunction, solver='adam', random_state=1)
            cls.fit(self.X_train, self.Y_train)
            predict = cls.predict(self.X_test)
            cMat = confusion_matrix(predict, self.Y_test)
            sum = cMat.sum()
            dSum = cMat.trace()
            self.population[i].fitness = dSum / sum
            self.population[i].reg = 0

    def sortByFitness(self):
        self.population.sort(key=lambda x: - x.fitness)

    def mate(self):
        for i in range(self.popSize):
            i1 = random.randint(0, self.popSize - 1)
            i2 = random.randint(0, self.popSize - 1)
            while i2 == i1:
                i2 = random.randint(0, self.popSize - 1)
            spos = random.randint(0, min(self.population[i1].network.depth, self.population[i2].network.depth))
            self.buffer[i].network.hidden = self.population[i1].network.hidden[0: spos] + self.population[
                                                                                              i2].network.hidden[spos:]
            self.buffer[i].network.depth = len(self.buffer[i].network.hidden)
            #if random.random() < 0.25* random.random():
                #self.mutate(self.buffer[i])

    def mutate(self,member):
        ipos = random.randint(0, member.network.depth)
        delta = random.randint(2, 200)
        member.network.hidden[ipos] = delta

    def swap(self):
        temp = self.population
        self.population = self.buffer
        self.buffer = temp

    def printBest(self, best):
        print()
        print("BEST CITIZEN :")
        print("FITNESS = ", best.fitness)

    def updateBest(self,best):
        if self.population[0].fitness > best.fitness:
            member = Citizen()
            member.network = self.population[0].network
            member.reg = self.population[0].reg
            member.fitness = self.population[0].fitness
            return member
        return best

    def regression(self):
        for i in range(self.popSize):
            cls = MLPClassifier(hidden_layer_sizes=self.population[i].network.hidden, max_iter=69000,
                                activation=self.population[i].network.activateFunction, solver='adam', random_state=1)
            sum=0
            for j in range(len(cls.coefs_)):
                for k in range(len(cls.coefs_[j])):
                    sum+=cls.coefs_[j][k]* cls.coefs_[j][k]

            c= self.cacl_creg(self.population[i])
            self.population[i].reg=sum*(1/c)*(len(self.X_train))




    def print_results(self,best):
        print("Classifier Report: ")
        cls = MLPClassifier(hidden_layer_sizes=best.network.hidden, max_iter=69000,
                            activation=best.network.activateFunction, solver='adam', random_state=1)
        cls.fit(self.X_train, self.Y_train)
        predict = cls.predict(self.X_test)
        print(classification_report(self.Y_test, predict, zero_division=0))

    def run(self):
        graph_arr=[]
        self.calcFitness()
        self.sortByFitness()
        member= Citizen()
        member.network=self.population[0].network
        member.reg=self.population[0].reg
        member.fitness =self.population[0].fitness
        best = member
        for i in range(self.maxIter):
            self.calcFitness()
            self.sortByFitness()
            best=self.updateBest(best)
            self.printBest(best)
            self.mate()
            self.swap()
            graph_arr.append(best.fitness)
        print()
        print("Best Train accuracy:")
        print(best.fitness)
        self.print_results(best)
        #Graph.draw(graph_arr)
