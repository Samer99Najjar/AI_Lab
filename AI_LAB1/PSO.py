from random import randint, random
import time

MIN_ASCII = 32
MAX_ASCII = 122


class GAStruct:
    def __init__(self, string, fitness):
        self.str = string
        self.fitness = fitness
        self.selfBest = ""
        self.velocity = ""
        self.personalbestfittnes=-1
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

    def __init__(self, GA_POPSIZE, GA_TARGET, GA_ELITRATE, GA_MUTATUONRATE, GA_MUTATION, GA_MAXITER):
        self.population = []
        self.buffer = []
        self.GA_POPSIZE = GA_POPSIZE
        self.GA_TARGET = GA_TARGET
        self.GA_ELITRATE = GA_ELITRATE
        self.GA_MUTATIONRATE = GA_MUTATUONRATE
        self.GA_MUTATION = GA_MUTATION
        self.GA_MAXITER = GA_MAXITER
        self.init_population()
        self.W=1
        self.C1=1
        self.C2=3


    def init_population(self):
        tsize = len(self.GA_TARGET)
        for i in range(self.GA_POPSIZE):
            citizen = GAStruct("", 0)
            for j in range(tsize):
                citizen.str += chr(randint(MIN_ASCII, MAX_ASCII))
                citizen.velocity += chr(randint(MIN_ASCII, MAX_ASCII))
            self.population.append(citizen)
            self.buffer.append(citizen)

    def calc_fitness(self):
        target = self.GA_TARGET
        tsize = len(self.GA_TARGET)
        for i in range(self.GA_POPSIZE):
            fitness = 0
            for j in range(tsize):
                fitness = fitness + abs(ord(self.population[i].get_str()[j]) - ord(target[j]))
            if self.population[i].personalbestfittnes==-1 or self.population[i].personalbestfittnes>fitness:
                self.population[i].personalbestfittnes=fitness
                self.population[i].selfBest=self.population[i].str

            self.population[i].set_fitness(fitness)

    def fitness_sort(self, x):
        return x.get_fitness()

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def update_parameters(self, t, N):   # update the PSO parameters

        self.W = 0.5
        self.C1 = -3 * (t / N) + 3.5
        self.C2 = 3 * (t / N) + 0.5

    def print_best(self,global_best,fit):
        print("Best: ",global_best, " (", fit, ")")


    def run(self):
        self.calc_fitness()
        self.sort_by_fitness()
        myfitness=self.population[0].fitness
        global_best=self.population[0].str
        start = time.time()
        for index in range(int(self.GA_MAXITER)):

            self.update_parameters(index, self.GA_MAXITER)
            self.calc_fitness()
            self.sort_by_fitness()
            self.print_best(global_best,myfitness)
            if self.population[0].fitness<myfitness:
               global_best=self.population[0].str
               myfitness=self.population[0].fitness
            if myfitness==0:
                print("Best: ",global_best,"(",myfitness,")")
                end = time.time()
                print("Time elapsed :", end - start)
                break
            for j in range(self.GA_POPSIZE):
                #in this loop we walk over all the parcials
                #we have POPSIZE parcials in our implementation
                string1= ""
                string2= ""
                rand1 = random()
                rand2 = random()
                for k in range(len(self.GA_TARGET)):
                    #here we calculate the new string for each parcial
                    #and update the velocity and position of it
                    #using the formulas we saw in the lecture
                    num1 = rand1 * self.C1 * (ord(self.population[j].selfBest[k]) - ord(self.population[j].str[k]))
                    num2 = rand2 * self.C2 * (ord(global_best[k]) - ord(self.population[j].str[k]))
                    num3= self.W *(ord(self.population[j].velocity[k]))
                    num4=num1+num2+num3
                    string1+= chr(int(num4)%95+32)
                    num5=ord(string1[k])+ord(self.population[j].str[k])
                    string2+=chr(int(num5%95+32))

                self.population[j].velocity=string1
                self.population[j].str=string2



            if myfitness == 0:

                break
