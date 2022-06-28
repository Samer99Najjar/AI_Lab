import numpy as np
import pandas
from GA import GA
import random
import os
from IPython.core.display_functions import clear_output
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import visualize
from CSV import *
import neat

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Relu(x):
    return x * (x > 0)

class generic_algorith:
    def execute(self, pop_size, generations, threshold, x, y, network):
        class Agent:
            def __init__(self, network):
                class neural_network:
                    def __init__(self, network):
                        self.weights = []
                        self.activations = []
                        for layer in network:
                            if layer[0] != None:
                                input_size = layer[0]
                            else:
                                input_size = network[network.index(layer) - 1][1]
                            output_size = layer[1]
                            activation = layer[2]
                            self.weights.append(np.random.randn(input_size, output_size))
                            self.activations.append(activation)

                    def propagate(self, data):
                        input_data = data
                        for i in range(len(self.weights)):
                            z = np.dot(input_data, self.weights[i])
                            a = self.activations[i](z)
                            input_data = a
                        yhat = a
                        return yhat

                self.neural_network = neural_network(network)
                self.fitness = 0

            def __str__(self):
                return 'LOSS: ' + str(self.fitness[0])

        def generate_agents(population, network):
            return [Agent(network) for _ in range(population)]

        def fitness(agents, x, y):
            for agentosh in agents:
                yhat = agentosh.neural_network.propagate(x)
                cost = (yhat - y) ** 2
                agentosh.fitness = sum(cost)
            return agents

        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
            print('\n'.join(map(str, agents)))
            agents = agents[:int(0.2 * len(agents))]
            return agents

        def unflatten(flattened, shapes):
            newarray = []
            index = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[index: index + size].reshape(shape))
                index += size
            return newarray

        def crossover(agents, network, pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(network)
                child2 = Agent(network)
                shapes = [a.shape for a in parent1.neural_network.weights]
                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
                split = random.randint(0, len(genes1) - 1)
                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child1.neural_network.weights = unflatten(child1_genes, shapes)
                child2.neural_network.weights = unflatten(child2_genes, shapes)
                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents

        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= 0.1:
                    W = agent.neural_network.weights
                    shapes = [a.shape for a in W]
                    flattened = np.concatenate([a.flatten() for a in W])
                    randint = random.randint(0, len(flattened) - 1)
                    flattened[randint] = np.random.randn()
                    newarray = []
                    indeweights = 0
                    for shape in shapes:
                        size = np.product(shape)
                        newarray.append(flattened[indeweights: indeweights + size].reshape(shape))
                        indeweights += size
                    agent.neural_network.weights = newarray
            return agents

        for i in range(generations):
            print('Generetions', str(i), ':')
            agents = generate_agents(pop_size, network)
            # print("passed generate")
            agents = fitness(agents, x, y)
            # print("passed fitnesss")
            agents = selection(agents)
            # print("i passed selection")
            agents = crossover(agents, network, pop_size)
            # print("i got passed corss over")
            agents = mutation(agents)
            # print("i got here")
            agents = fitness(agents, x, y)
            # print("but not past here")
            if any(agent.fitness < threshold for agent in agents):
                print('Threshold met at generation ' + str(i) + ' !')
            if i % 100:
                # print("hi")
                clear_output()
        return agents[0]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file):
    # Load configuration.

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    print('#########################################################################')
    print('LAB 5-B')
    print('MORSY BIADSY ID:318241221 \t SAMER NAJJAR ID:207477522')
    print('#########################################################################')
    print()
    print("STARTING NOW...")
    print()

    mycsv= CSV("glass.csv")
    data, label = mycsv.read_data()
    normalized = DataFrame(MinMaxScaler().fit_transform(data))

    train, test, train_vec, test_vec = train_test_split(normalized, label, stratify=label, test_size=0.2,
                                                        random_state=1)
    mycsv.calc_acc(train,train_vec,test,test_vec)

    ga = GA(8, 8, train, train_vec, test, test_vec)
    ga.run()

