import CVRP
import City
import TabuSearch
from TabuSearch import *
from SimAnnealing import *
from ACO import *
from math import sqrt
from GA import *
import PSO


class args:
    maxIter = 1500
    numNeighbors = 2048
    maxTabu = 20
    localOptStop = 25
    A = 3
    B = 4
    Q = 1000
    P = 0.1
    alpha = 0.5
    temperature = 100


def get_input(input_file):
    file = open(input_file + '.txt', 'r')  # input string
    for _ in range(3):
        file.readline()

    dimensionLine = file.readline()
    arr = [num for num in dimensionLine.split(' ')]
    dimension = int(arr[2])

    file.readline()

    capacityLine = file.readline()
    arr = [num for num in capacityLine.split(' ')]
    capacity = int(arr[2])

    file.readline()

    cityLine = file.readline()
    arr = [num for num in cityLine.split(' ')]
    depot = City.City(int(arr[0]), int(arr[1]), int(arr[2]))

    cities = []
    for _ in range(dimension - 1):
        cityLine = file.readline()
        arr = [num for num in cityLine.split(' ')]
        city = City.City(int(arr[0]) - 1, int(arr[1]), int(arr[2]))
        cities.append(city)

    file.readline()
    file.readline()

    for i in range(dimension - 1):
        demandLine = file.readline()
        arr = [num for num in demandLine.split(' ')]
        cities[i].setDemand(int(arr[1]))

    cities.insert(0, depot)

    distanceMat = calcDistanceMatrix(cities)
    cities.pop(0)

    problem = CVRP.CVRP(distanceMat, depot, cities, capacity, len(cities))

    return problem


def calcDistanceMatrix(cities):
    array = []
    numOfCities = len(cities)
    for i in range(numOfCities):
        arr = []
        for j in range(numOfCities):
            arr.append(distance(cities[i], cities[j]))
        array.append(arr)
    return array


def distance(city1, city2):
    x = city1.x - city2.x
    dx = x * x

    y = city1.y - city2.y
    dy = y * y

    return sqrt(dx + dy)


def genetic(problem):
    ga = GA(problem)
    ga.run()
    problem.printSolution()


def ant(problem):
    ACO(problem, args())
    problem.printSolution()


def simulated(problem):
    simulatedAnnealing(problem, args())
    problem.printSolution()


def tabu(problem):
    tabuSearch(problem, args())
    problem.printSolution()


if __name__ == '__main__':
    running = True
    print('#########################################################################')
    print('LAB 3')
    print('MORSY BIADSY ID:318241221 \t SAMER NAJJAR ID:207477522')
    print('#########################################################################')
    while running:
        print()
        print('CHOOSE INPUT EXAMPLE:')
        input_file = input('1 --> E-n22-k4 \t 2 --> E-n33-k4 \t 3 --> E-n51-k5 \t 4 --> E-n76-k10 \t 5 --> EXIT\n')

        # EXIT
        if int(input_file) == 5:
            break

        # INPUT FILE
        problem = get_input(input_file)

        # CHOOSE PROBLEM
        solution = int(input('1 --> GA \t 2 --> ACO \t 3 --> SIMU \t 4 --> TABU\n'))
        if solution == 1:
            genetic(problem)
        if solution == 2:
            ant(problem)
        if solution == 3:
            simulated(problem)
        if solution == 4:
            tabu(problem)

