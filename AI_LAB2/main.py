import math
import time
from random import randint, random

import numpy
from numpy.random import choice
import matplotlib.pyplot as plt

from Baldwin import Baldwin
from PSO import PSO
from NQueens import NQueens
from MinimalConflicts import MinimalConflict
from BinPacking import BinPacking

MIN_ASCII = 32
MAX_ASCII = 122

GA_POPSIZE = 2048
GA_TARGET = 'Hello World!'
GA_ELITRATE = 0.10
GA_MUTATIONRATE = 0.25
GA_MUTATION = random() * GA_MUTATIONRATE
GA_MAXITER = 16384
GA_CROSSOVER = 1
bull_eye = 0
sus = 0
species_count = 30


class GAStruct:
    def __init__(self, string, fitness):
        self.str = string
        self.fitness = fitness
        self.selfBest = ""
        self.velocity = ""
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


def init_population():
    population = []
    buffer = []
    tsize = len(GA_TARGET)
    for i in range(GA_POPSIZE):
        citizen = GAStruct("", 0)
        for j in range(tsize):
            citizen.str += chr(randint(MIN_ASCII, MAX_ASCII))
        population.append(citizen)
        buffer.append(citizen)
    return population, buffer


def split_into_species(population):
    return 0


def distance(firstIndex, secondIndex):
    distances = numpy.zeros((len(firstIndex) + 1, len(secondIndex) + 1))

    for t1 in range(len(firstIndex) + 1):
        distances[t1][0] = t1

    for t2 in range(len(secondIndex) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(firstIndex) + 1):
        for t2 in range(1, len(secondIndex) + 1):
            if firstIndex[t1 - 1] == secondIndex[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    print(distances[len(firstIndex), len(secondIndex)])
    return distances[len(firstIndex), len(secondIndex)]


def aging(population, buffer):
    for i in range(GA_POPSIZE):
        if population[i].age <= 15:
            population[i].fitness -= 10
            if population[i].fitness <= 0:
                population[i].fitness = 1
        if population[i].age < 6:
            population[i].fitness += 20
            buffer[i].fitness = population[i].fitness
            buffer[i].age = population[i].age
    for i in range(GA_POPSIZE):
        population[i].age += 1
        buffer[i].age += 1
    return population, buffer


"""
def sigma_scaling(population,index,avg,standrard_dive,c=2):

    fitness= population[index].fitness
    if standrard_dive > 0:
       return 1+(fitness-avg)/(2*standrard_dive)

    return 1
"""


def SUS(population):
    F = 0
    N = 360
    for i in range(GA_POPSIZE):
        F += population[i].fitness
    fitnessBarometer = F / N
    sqrtFit = 0
    start = randint(0, GA_MAXITER) % fitnessBarometer
    ptr = math.sqrt(random() * fitnessBarometer)
    for i in range(int(start), GA_POPSIZE):
        sqrtFit += i * fitnessBarometer
        if sqrtFit >= ptr:
            return i
    return -1


def tournament(population, k):
    # here we choose random number K
    # we choose random K "genes"
    # we return the best one of them
    best = None
    for _ in range(k):
        member = population[randint(0, len(population) - 1)]
        if best == None:
            best = member
        elif best.fitness > member.fitness:
            best = member
    return best


def calc_fitness(population):
    target = GA_TARGET
    tsize = len(GA_TARGET)
    for i in range(GA_POPSIZE):
        fitness = 0
        for j in range(tsize):
            fitness = fitness + abs(ord(population[i].get_str()[j]) - ord(target[j]))
        population[i].set_fitness(fitness)

    return population


def bulls_eye(population):
    # check if the char is in the right place give no penalty
    # else if it's in wrong place give small penalty
    # else if it's not found give bigger penalty
    target = GA_TARGET
    tsize = len(GA_TARGET)

    for i in range(GA_POPSIZE):
        fitness = 0
        str = population[i].str
        for j in range(tsize):
            if str[j] == target[j]:
                fitness += 0
            else:
                if target.find(str[j]):
                    fitness += 5
                else:
                    fitness += 15
        population[i].set_fitness(fitness)
    return population


def fitness_sort(x):
    return x.get_fitness()


def sort_by_fitness(population):
    population.sort(key=fitness_sort)
    return population


def elitism(population, buffer, esize):
    for i in range(esize):
        buffer[i].set_str(population[i].get_str())
        buffer[i].set_fitness(population[i].getfitness())
    return buffer


def linear_scaling(fitness, a=0.5, b=0):
    return a * fitness + b


def roulette_spin(new_fitnesses):
    fitness_sum = sum(new_fitnesses)
    prob = [fitness / fitness_sum for fitness in new_fitnesses]
    return choice(len(new_fitnesses), p=prob)


def RWS(population):
    new_fitnesses = [1 / linear_scaling(member.fitness + 1) for member in population]
    selection = roulette_spin(new_fitnesses)
    return selection


def one_point_crossover(population, buffer):
    # choose one random point
    # until this point put the string of parent 1
    # from that point till the end put the string of parent 2
    esize = GA_POPSIZE * GA_ELITRATE
    tsize = len(GA_TARGET)
    for i in range(int(esize), GA_POPSIZE):
        if sus == 0:
            i1 = randint(0, GA_POPSIZE / 2)
            i2 = randint(0, GA_POPSIZE / 2)
        else:
            if sus == 1:
                i1 = SUS(population)
                i2 = SUS(population)
            else:
                if sus == 2:
                    i1 = RWS(population)
                    i2 = RWS(population)
                else:
                    i1 = tournament(population, 69)
                    i2 = tournament(population, 69)
        spos = randint(0, tsize - 1)
        if 0 <= sus <= 2:
            distance(i1, i2)
            buffer[i].str = population[i1].str[0: spos] + population[i2].str[spos:]

        else:
            distance(i1.str, i2.str)
            buffer[i].str = i1.str[0: spos] + i2.str[spos:]
        if random() < GA_MUTATION:
            buffer[i] = mutate(buffer[i])

    return population, buffer


def two_point_crossover(population, buffer):
    # choose 2 point
    # from the start to the first point put the string of parent 1
    # from the first point to the second put string of parent 2
    # from the second point to the end put string of parent 1
    esize = GA_POPSIZE * GA_ELITRATE
    tsize = len(GA_TARGET)
    for i in range(int(esize), GA_POPSIZE):
        if sus == 0:
            i1 = randint(0, GA_POPSIZE / 2)
            i2 = randint(0, GA_POPSIZE / 2)
        else:
            if sus == 1:
                i1 = SUS(population)
                i2 = SUS(population)
            else:
                if sus == 2:
                    i1 = RWS(population)
                    i2 = RWS(population)
                else:
                    i1 = tournament(population, 69)
                    i2 = tournament(population, 69)

        spos = randint(0, tsize - 2)
        spos2 = randint(spos + 1, tsize - 1)
        if 0 <= sus <= 2:
            distance(population[i1].str, population[i2].str)
            buffer[i].str = population[i1].str[0: spos] + population[i2].str[spos: spos2] + population[i1].str[spos2:]
        else:
            distance(i1.str, i2.str)
            buffer[i].str = i1.str[0: spos] + i2.str[spos: spos2] + i1.str[spos2:]

        if random() < GA_MUTATION:
            buffer[i] = mutate(buffer[i])

    return population, buffer


def uniform_crossover(population, buffer):
    # for each char at string choose who to copy from parent 1 or 2 randomly
    esize = GA_POPSIZE * GA_ELITRATE
    tsize = len(GA_TARGET)

    for i in range(int(esize), GA_POPSIZE):
        if sus == 0:
            i1 = randint(0, GA_POPSIZE / 2)
            i2 = randint(0, GA_POPSIZE / 2)
        else:
            if sus == 1:
                i1 = SUS(population)
                i2 = SUS(population)
            else:
                if sus == 2:
                    i1 = RWS(population)
                    i2 = RWS(population)
                else:
                    i1 = tournament(population, 69)
                    i2 = tournament(population, 69)
        myStr = ''
        for j in range(tsize):
            spos = randint(0, 1)
            if 0 <= sus <= 2:
                if spos == 0:
                    myStr += population[i1].str[j]
                else:
                    myStr += population[i2].str[j]
            else:
                if spos == 0:
                    myStr += i1.str[j]
                else:
                    myStr += i2.str[j]
        buffer[i].str = myStr

        if random() < GA_MUTATION:
            buffer[i] = mutate(buffer[i])

    return population, buffer


def mutate(member):
    tsize = len(GA_TARGET)
    ipos = randint(0, tsize - 1)
    delta = randint(MIN_ASCII, MAX_ASCII)
    member.str = list(member.str)
    member.str[ipos] = chr((ord(member.str[ipos]) + delta) % 122)
    member.str = "".join(member.str)
    return member


def mate(population, buffer):
    if GA_CROSSOVER == 1:
        return one_point_crossover(population, buffer)
    if GA_CROSSOVER == 2:
        return two_point_crossover(population, buffer)
    return uniform_crossover(population, buffer)


def print_best(gav):
    print("Best: ", gav[0].str, " (", gav[0].fitness, ")")


def swap(population, buffer):
    temp = population
    population = buffer
    buffer = temp
    return population, buffer


def calc_Av2(population):
    # calculate average and standard dev  and print them
    fitnessSum = 0
    for member in population:
        fitnessSum += member.fitness
    avg = fitnessSum / GA_POPSIZE
    sum = 0
    for member in population:
        sum += (avg - member.get_fitness()) ** 2
    standardDev = abs(sum / GA_POPSIZE)
    standardDev = math.sqrt(standardDev)

    # print("Fitness Avg: ", avg, " --- Standard deviation: ", standardDev)
    return avg, standardDev


def calc_Av(population):
    # calculate average and standard dev  and print them
    fitnessSum = 0
    for member in population:
        fitnessSum += member.fitness
    avg = fitnessSum / GA_POPSIZE
    sum = 0
    for member in population:
        sum += (avg - member.get_fitness()) ** 2
    standardDev = abs(sum / GA_POPSIZE)
    standardDev = math.sqrt(standardDev)

    print("Fitness Avg: ", avg, " --- Standard deviation: ", standardDev)
    return avg, standardDev


def random_walk(state, walksnumber):
    mystate = state
    for i in range(walksnumber):
        newstate = steepest_ascend(mystate)
        if newstate.fitness < mystate.fitness:
            mystate = newstate
    return mystate


def k_gene_exchange(population):
    child = GAStruct("", 10)
    for i in range(len(GA_TARGET)):
        child.str += population[randint(0, GA_POPSIZE / 2)].str[i]

    # print(child.NQueens)

    fitness = 0
    for j in range(len(GA_TARGET)):
        fitness = fitness + abs(ord(child.get_str()[j]) - ord(GA_TARGET[j]))
    child.fitness = fitness
    return child


def hill_climb(state):
    if state.fitness == 0:
        return state
    nearstates = calc_near_states(state)
    for city in nearstates:
        if city.fitness < state.fitness:
            state = city
            return state
        if state.fitness == 0:
            return state
    return state


def steepest_ascend(state):
    if state.fitness == 0:
        return state
    nearstates = calc_near_states(state)
    for city in nearstates:
        if city.fitness < state.fitness:
            state = city
        if state.fitness == 0:
            return state
    return state


def calc_near_states(state):
    nearstates = []
    for i in range(len(GA_TARGET)):
        newstate = GAStruct("", 10)
        newstate.str = state.str
        newstate.str = state.str[0:i] + chr(randint(MIN_ASCII, MAX_ASCII)) + state.str[i + 1:]
        nearstates.append(newstate)
    calc_fitness_states(nearstates)
    return nearstates


def calc_fitness_states(nearstates):
    for i in range(len(nearstates)):
        fitness = 0
        for j in range(len(GA_TARGET)):
            fitness = fitness + abs(ord(nearstates[i].get_str()[j]) - ord(GA_TARGET[j]))
        nearstates[i].fitness = fitness
        # print(nearstates[i].fitness)


def draw_hist(points):
    range = (0, 100)
    bins = 10
    plt.hist(points, bins, range, color='green',
             histtype='bar', rwidth=0.8)

    # x-axis label
    plt.xlabel('fitness value')
    # frequency label
    plt.ylabel('No. of fit val')
    # plot title
    plt.title('My histogram')

    # function to show the plot
    plt.show()


def calc_points(population):
    maxfit = 1
    points = []
    for member in population:
        points.append(member.fitness)
        if member.fitness > maxfit:
            maxfit = member.fitness

    for i in range(GA_POPSIZE):
        points[i] = points[i] / maxfit * 100
    return points


def calc_his(population):
    draw_hist(calc_points(population))


def start():
    start = time.time()
    population, buffer = init_population()
    for index in range(GA_MAXITER):
        sTime = time.time()
        if bull_eye == 1:
            population = bulls_eye(population)
        else:
            population = calc_fitness(population)

        population = sort_by_fitness(population)
        calc_Av(population)
        print_best(population)
        end = time.time()
        print("time --> ", end - sTime, "\n")
        if population[0].get_fitness() == 0:
            break
        # population, buffer = aging(population, buffer)
        population, buffer = mate(population, buffer)
        population, buffer = swap(population, buffer)
    end = time.time()
    print("Time elapsed :", end - start)


def nqueens():
    minimalConflict = int(input("MINIMAL CONFLICT :\n1 --> YES  || 0 --> NO\n"))
    if minimalConflict == 1:
        numQueens = int(input("NQUEEN :\n"))
        minimalconf = MinimalConflict(numQueens)
        minimalconf.start()
    else:
        mutateType = int(input("MUTATION TYPE :\n1 --> RANDOM  |  0 --> INVERSION\n"))
        pmx = int(input("PMX OR CX :\n1 --> PMX  |  0 --> CX\n"))
        pmx = pmx == 1
        numQueens = int(input("NQUEEN :\n"))
        nq = NQueens(numQueens, GA_POPSIZE, GA_MAXITER, GA_ELITRATE, GA_MUTATION, pmx, mutateType)
        nq.run()


def first_fit(items, N, C):
    array = []
    resArr = []
    for i in range(N):
        array.append(0)
    for i in range(N):
        for j in range(N):
            if array[j] < C and array[j] + items[i] <= C:
                array[j] = array[j] + items[i]
                resArr.append(j)
                break
    for i in range(N):
        if array[i] == 0:
            print(resArr)
            print("number of packs is :", i)
            break


def run_examples(firstFit):
    examples = ['ex1.txt', 'ex2.txt', 'ex3.txt', 'ex4.txt']
    C = [60, 200, 100, 120]
    for example in examples:
        print(f"EXAMPLE {examples.index(example) + 1}:")
        f = open(example, "r")
        l = []
        for i in list(f.readlines()):
            if i != '\n':
                l.append(int(i))
        if firstFit == 1:
            start = time.time()
            first_fit(l, len(l), C[examples.index(example)])
            end = time.time()
            # total time taken
            print(f"TIME TAKEN: {end - start}\n")
        else:
            maxNum = int(input("MAXIMUM NUMBER OF STEPS :\n"))
            binpacking = BinPacking(C[examples.index(example)], GA_POPSIZE, GA_ELITRATE, GA_MUTATION, maxNum, l)
            start = time.time()
            binpacking.start()
            end = time.time()
            # total time taken
            print(f"TIME TAKEN: {end - start}\n")


def binpacking():
    examples = int(input("1 --> USE THE EXAMPLES  |  0 --> INSERT THE DATA MANUALLY \n"))
    firstFit = int(input("FIRST FIT ?\n1 --> YES  |  0 --> NO  \n"))
    if examples == 1:
        run_examples(firstFit)
    else:
        if firstFit == 1:
            itemsStr = input("ENTER ITEMS VALUE IN ONE LINE (SPACE BETWEEN THE VALUES):\n")
            items = []
            for i in itemsStr.split():
                if i != '\n':
                    items.append(int(i))
            c = int(input("ENTER BIN CAPACITY:\n"))
            start = time.time()
            first_fit(items, len(items), c)
            end = time.time()
            # total time taken
            print(f"TIME TAKEN: {end - start}\n")
        else:
            maxNum = int(input("MAXIMUM NUMBER OF STEPS:\n"))
            itemsStr = input("ENTER ITEMS VALUE IN ONE LINE (SPACE BETWEEN THE VALUES):\n")
            items = []
            for i in itemsStr.split():
                if i != '\n':
                    items.append(int(i))
            c = int(input("ENTER BIN CAPACITY\n"))
            binpacking = BinPacking(c, 500, GA_ELITRATE, GA_MUTATION, maxNum, items)
            start = time.time()
            binpacking.start()
            end = time.time()
            # total time taken
            print(f"TIME TAKEN: {end - start}\n")


def run_hill(beType):
    population, buffer = init_population()
    calc_fitness(population)
    for index in range(GA_MAXITER):
        for i in range(GA_POPSIZE):
            if beType == 1:
                population[i] = hill_climb(population[i])
            if beType == 2:
                population[i] = steepest_ascend(population[i])
            if beType == 3:
                population[i] = random_walk(population[i], 20)
        sort_by_fitness(population)
        print_best(population)
        if population[0].get_fitness() == 0:
            break


def lab2():
    lab2 = int(input("\n1 --> BALDWIN \n2 --> NQUEENS \n3 --> BULLSEYE"))

    if lab2 == 1:
        targetSize = int(input("\nTARGET SIZE :"))
        baldwin = Baldwin(targetSize)
        baldwin.run()

    if lab2 == 2:
        nqType = int(input("\n1 --> CLIMBING HILL\n2 --> STEEPEST ASCENT \n3 --> RANDOM WALK \n"))
        if nqType == 1:
            numQueens = int(input("NQUEEN :\n"))
            nqueen = NQueens(numQueens, GA_POPSIZE, GA_MAXITER, GA_ELITRATE, GA_MUTATION, 1, 1)
            nqueen.part2_run(1)

        if nqType == 2:
            numQueens = int(input("NQUEEN :\n"))
            nqueen = NQueens(numQueens, GA_POPSIZE, GA_MAXITER, GA_ELITRATE, GA_MUTATION, 1, 1)
            nqueen.part2_run(2)

        if nqType == 3:
            numQueens = int(input("NQUEEN :\n"))
            nqueen = NQueens(numQueens, GA_POPSIZE, GA_MAXITER, GA_ELITRATE, GA_MUTATION, 0, 1)
            nqueen.part2_run(3)

    if lab2 == 3:
        beType = int(input("\n1 --> CLIMBING HILL\n2 --> STEEPEST ASCENT \n3 --> RANDOM WALK \n"))
        run_hill(beType)


if __name__ == '__main__':
    run = True
    while run:
        problem = int(
            input(
                "Select: \n1 --> GENETIC ALGO\n2 --> NQUEENS PROBLEM\n3 --> BINPACKING PROBLEM \n4 --> LAB 2  \n5 --> EXIT"))
        if problem == 1:
            pso = int(input("PSO ?  \n1 --> YES  |  0 --> NO  \n"))
            if pso == 1:
                pso = PSO(GA_POPSIZE, GA_TARGET, GA_ELITRATE, GA_MUTATIONRATE, GA_MUTATION, GA_MAXITER)
                pso.run()
            else:
                GA_CROSSOVER = int(
                    input("Choose Crossover: \n1 --> ONE POINT  |  2 --> TWO POINT  |  ELSE --> UNIFORM  \n"))
                bull_eye = int(input("Bulls eye ? \n1 --> YES  |  0 --> NO  \n"))
                sus = int(input("0 --> RANDOM  |  1 --> SUS  |  2 --> RWS  |  ELSE --> TOURNAMENT  \n"))
                start()

        if problem == 2:
            nqueens()

        if problem == 3:
            binpacking()

        if problem == 4:
            lab2()
        if problem == 5:
            run = False
