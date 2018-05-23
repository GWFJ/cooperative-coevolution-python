# !/usr/bin/python

# BP On Feed-Forward Network for Time Series

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk

# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra

# Gary Wong
# University of the South Pacific, Fiji
# gary.wong.fiji@gmail.com


import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import os
import shutil

class Network:
    def __init__(self, Topo, Train, Test):
        self.lrate = 0.1
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def RMSE(self, actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta[0])

        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta[0])

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    def evaluate_solution(self, data):


        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'

        return fx

    def ForwardFitnessPass(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)
        actual = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            actual[pat] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'


        # FX holds calculated output

        return self.RMSE(actual, fx)

    def ForwardFitnessPassBP(self, data):  # BP with SGD (Stocastic BP)

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)
        actual = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            actual[pat] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'


        # FX holds calculated output

        return self.RMSE(actual, fx)
class Backpropagation:
    def __init__(self):
        print 'initialize'
    def Run(self, Data, Network, Topology):
        NetworkSize = (Topology[0] * Topology[1]) + (Topology[1] * Topology[2]) + Topology[1] + Topology[2]
        InitialWeights =  np.random.uniform(-5,5,NetworkSize)

        Network.decode(InitialWeights)

        size = Data.shape[0]

        Input = np.zeros((1, Topology[0]))  # temp hold input
        Desired = np.zeros((1, Topology[2]))
        fx = np.zeros(size)


        Gen = 0
        while(Gen < 5000):
            for sample in Data:
                Gen += 1
                Input[:] = sample[0:Topology[0]]
                Desired[:] = sample[Topology[0]:]

                Network.ForwardPass(Input)
                Network.BackwardPass(Input, Desired, 0)

                print 'Epoch: ' + str(Gen) + ' Fitness: ' + str(Network.ForwardFitnessPassBP(Data))
class Graphing:

    def PlotGraphs(self, net, run, testdata, traindata, topology):

        testsize = testdata.shape[0]
        trainsize = traindata.shape[0]


        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        ytestdata = testdata[:, topology[0]]
        ytraindata = traindata[:, topology[0]]

        trainout = net.evaluate_solution(traindata)
        testout = net.evaluate_solution(testdata)

        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, trainout, label='predicted (train)')


        plt.legend(loc='upper right')
        plt.title("Plot of Train Data vs MCMC Uncertainty ")

        if not os.path.exists('mcmcresults/run' + str(run) + '/'):
            os.makedirs('mcmcresults/run' + str(run) + '/')

        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_train.png')
        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_train.svg', format='svg', dpi=600)
        plt.clf()

        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, testout, label='predicted (test)')

        plt.legend(loc='upper right')
        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_test.png')
        plt.savefig('mcmcresults/run' + str(run) + '/dmcmc_test.svg', format='svg', dpi=600)
        plt.clf()



class Species:
    def __init__(self, populationsize, speciessize):
        self.populations = [np.random.randn(speciessize) for count in xrange(populationsize)]
        self.bestind = 0
        self.bestfit = 99
        self.worstind = 0
        self.worstfit = 0
        self.fitnesslist = np.zeros(populationsize)
        self.popsize = populationsize

class PopulationMutation:
    def __init__(self, species):
        self.numparents = 2
        self.numkids = 2
        self.species = species
        self.parentlist = np.arange(0, species.popsize)
        self.epsilon = 1e-50
        self.sigma_eta = 0.1
        self.sigma_zeta = 0.1
        self.newpops =  np.zeros((self.numkids,species.populations[0].shape[0]))
        self.kidsfitnesslist = np.zeros(self.numkids)

        self.parentspops = np.zeros((self.numparents, species.populations[0].shape[0]))
        self.parentsfitnesslist = np.zeros(self.numparents)

        self.allnewpops = np.zeros((self.numkids + self.numparents,species.populations[0].shape[0]))
        self.allnewpops_fitness = np.zeros(self.numkids+ self.numparents)
        self.parentsindexes = []


    def selectparents(self):
        # swp = self.parentlist[0]
        # self.parentlist[0] = self.parentlist[self.species.bestind]
        # self.parentlist[self.species.bestind] = swp

        for i in range(0, self.numparents):
            index = random.randint(1, self.species.popsize - 1)
            swp = self.parentlist[index]
            self.parentlist[index] = self.parentlist[i]
            self.parentlist[i] = swp

    def sim_binary_xover(self, ind1,ind2):  # Simulated Binary Crossover https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py

        #:param ind1: The first individual participating in the crossover.
        #:param ind2: The second individual participating in the crossover.
        #:param eta: Crowding degree of the crossover. A high eta will produce
        #           children resembling to their parents, while a small eta will
        #          produce solutions much more different.
        x1 = ind1
        x2 = ind2

        eta = 2

        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            rand = random.random()
            if rand <= 0.5:
                beta = 2. * rand
            else:
                beta = 1. / (2. * (1. - rand))

            beta **= 1. / (eta + 1.)

            ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
            ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

        return ind1, ind2

    def blend_xover(self, ind1, ind2):
        # '''Executes a blend crossover that modify in-place the input individuals.
        # :param ind1: The first individual participating in the crossover.
        # :param ind2: The second individual participating in the crossover.
        # :param alpha: Extent of the interval in which the new values can be drawn
        #           for each attribute on both side of the parents' attributes. '''
        alpha = 0.1
        x1 = ind1
        x2 = ind2

        for i in range(ind1.size):  # (x1, x2) in enumerate(zip(ind1, ind2)):
            gamma = (1. + 2. * alpha) * random.random() - alpha
            ind1[i] = (1. - gamma) * x1[i] + gamma * x2[i]
            ind2[i] = gamma * x1[i] + (1. - gamma) * x2[i]

        return ind1, ind2

    def G3PCX(self):
        for l in range(0, self.numkids):
            centroid = (self.species.populations[self.parentlist[0]] + self.species.populations[self.parentlist[1]]) / self.numparents
            centroid_index_diff = centroid - self.species.populations[self.parentlist[0]]
            parent2_index_diff = self.species.populations[self.parentlist[1]] - self.species.populations[self.parentlist[0]]
            modu_centroid_index_diff = np.linalg.norm(centroid_index_diff)
            modu_parent2_index_diff =  np.linalg.norm(parent2_index_diff)

            if(modu_parent2_index_diff < self.epsilon):
                print 'RUN Points are very close to each other. Quitting this run'
                return

            orthogonal_direction = np.sum(parent2_index_diff * centroid_index_diff)

            if((modu_parent2_index_diff * modu_centroid_index_diff) == 0):
                orthogonal_direction /= 1
            else:
                orthogonal_direction /= (modu_parent2_index_diff * modu_centroid_index_diff)

            if(orthogonal_direction == 1.0):
                return 0

            orthogonal_direction = modu_parent2_index_diff * np.sqrt(1 - np.square(orthogonal_direction))


            if (orthogonal_direction == 0 or np.isnan(orthogonal_direction)):
                return 0


            #Uses python normal distribution. If problems arise try use box muellers as in C++ code
            temp = np.random.uniform(0,orthogonal_direction * self.sigma_eta,centroid.shape[0])
            temp2 = temp

            for i in range(0, centroid.shape[0]):
                if(np.square(modu_centroid_index_diff) == 0):
                    temp2[i] = temp[i] - ((np.sum(temp * centroid)* centroid_index_diff[i]) / 1)
                else:
                    temp2[i] = temp[i] - ((np.sum(temp * centroid) * centroid_index_diff[i]) / np.square(modu_centroid_index_diff))

            temp = temp2

            self.newpops[l] = self.species.populations[self.parentlist[0]] + temp
            self.newpops[l] = self.newpops[l] + (np.random.uniform(0,self.sigma_zeta,1)[0] * centroid_index_diff)

        return 1

    def findparents(self):
        for i in range(0, self.numparents):
            self.parentsindexes.append(np.random.randint(self.species.popsize, size=1)[0])

        for i in range(0, self.numparents):
            self.parentspops[i] = self.species.populations[self.parentsindexes[i]]

    def sort_replace_parents(self):

        #Copy over the two kids chromes and kids fitness
        for i in range(0, self.numkids):
            self.allnewpops[i] = self.newpops[i]
            self.allnewpops_fitness[i] = self.kidsfitnesslist[i]

        #Copy over parents chromes and parents fitness
        for i in range(0, self.numparents):
            self.allnewpops[i + self.numkids] = self.species.populations[self.parentsindexes[i]]
            self.allnewpops_fitness[i + self.numkids] = self.parentsfitnesslist[i]

        #Order new pops by fitness
        list = []
        for i in range(self.numparents + self.numkids):
            list.append(i)

        for i in range((self.numparents + self.numkids) - 1):
            dbest =  self.allnewpops_fitness[list[i]]

            for j in range(i + 1, (self.numparents + self.numkids)):

                if (self.allnewpops_fitness[list[j]] < dbest):
                    dbest = self.allnewpops_fitness[list[j]]
                    temp = list[j]
                    list[j] = list[i]
                    list[i] = temp

        debug = 1

        if(debug == 1):
            #Replace both parents with top 2 new pops
            for i in range(0, self.numparents):
                print 'Replacing Parent at ind ' + str(self.parentlist[i]) + ' with fit: ' + str(self.allnewpops_fitness[list[i]])
                self.species.populations[self.parentlist[i]] =  self.allnewpops[list[i]]
                self.species.fitnesslist[self.parentlist[i]] = self.allnewpops_fitness[list[i]]

            #Sort species, update best ind

            print 'Previous Best Ind: ' + str(self.species.bestind) + ' Previous Best Fit: ' + str(self.species.bestfit)
            self.species.bestind = 0
            self.species.bestfit = self.species.fitnesslist[0]

            for i in range(1, self.species.popsize):
                if (self.species.fitnesslist[i] < self.species.bestfit):
                    self.species.bestind = i
                    self.species.bestfit = self.species.fitnesslist[i]

            print 'New Best Ind: ' + str(self.species.bestind)  + ' New Best Fit: ' + str(self.species.bestfit)


class CoEvolution:
    def __init__(self, topology, popsize, traindata, testdata, max_evals, net):
        self.topology = topology
        self.traindata = traindata
        self.testdata = testdata
        self.popsize = popsize
        self.max_evals = max_evals
        # self.xover_rate = xover_rate
        # self.mu_rate = mu_rate
        self.num_eval = 0
        self.allspecies = []
        self.net = net


    def initialize(self):
        for i in xrange(self.topology[1]):
            self.allspecies.append(Species(self.popsize, self.topology[0] + 1))
        for i in xrange(self.topology[2]):
            self.allspecies.append(Species(self.popsize, self.topology[1] + 1))

    def evolve(self,rep, cyc):
        # self.initialize()

        for h in range(cyc):
            print 'Cycle ' + str(h)
            for u in range(len(self.allspecies)):
                for i in range(rep):
                    regenerate = 0

                    while regenerate == 0:

                        evolvePopulation = PopulationMutation(self.allspecies[u])
                        evolvePopulation.selectparents()

                        crossover_method = 0

                        generated = 0

                        if(crossover_method == 0):
                            #     Binary Crossover
                            [child1, child2] = evolvePopulation.sim_binary_xover(self.allspecies[u].populations[evolvePopulation.parentlist[0]], self.allspecies[u].populations[evolvePopulation.parentlist[1]])
                            evolvePopulation.newpops[0] = child1
                            evolvePopulation.newpops[1] = child2
                            generated = 1

                        elif(crossover_method == 1):
                            #     G3PCX Crossover
                            generated = evolvePopulation.G3PCX()
                        elif (crossover_method == 2):
                            #     Blend Crossover
                            [child1, child2] = evolvePopulation.blend_xover(
                                self.allspecies[u].populations[evolvePopulation.parentlist[0]],
                                self.allspecies[u].populations[evolvePopulation.parentlist[1]])
                            evolvePopulation.newpops[0] = child1
                            evolvePopulation.newpops[1] = child2
                            generated = 1

                        if(generated == 1):

                            print u
                            [ind, startind] = self.getind_bestonly(u)

                            # print self.net.ForwardFitnessPass(self.traindata, self.ind_tonetworkready(ind))
                            if (u == 5):
                                self.print_best_indexes()

                            evolvePopulation.kidsfitnesslist = self.evaluate_newsolutions(u, ind, startind, evolvePopulation.newpops)
                            evolvePopulation.findparents()
                            evolvePopulation.parentsfitnesslist = self.evaluate_newsolutions(u,ind, startind, evolvePopulation.parentspops)
                            evolvePopulation.sort_replace_parents()

                            regenerate = 1 #Quit regenerate while loop

                self.evalspec(u)
                self.print_best_indexes()

            # self.print_best_indexes()
            [individual, startind] = self.getind_bestonly(0)
            # print individual
            fitnesss = self.net.ForwardFitnessPass(self.traindata, self.ind_tonetworkready(individual))
            #
            # insds = self.getind_with_best_networkready(5, self.allspecies[5].bestind)
            # finesse = self.net.ForwardFitnessPass(self.traindata, insds)
            print 'Overall Fitness (RMSE): ' + str(fitnesss)
            # print 'Best Fitness: ' + str(fitnesss) + ' Last Species Best Fitness: ' + str(self.allspecies[5].bestfit) + ' --- ' + str(finesse)

    def print_best_indexes(self):
        indstring = ''
        for u in range(len(self.allspecies)):
            indstring += ' ' + str(self.allspecies[u].bestind)

        print indstring

    def ind_tonetworkready(self, ind):
        layer1 = ind[0:(self.topology[0] * self.topology[1]) + self.topology[1]]
        layer2 = ind[(self.topology[0] * self.topology[1]) + self.topology[1]:]
        layer1 = np.reshape(layer1, (-1, self.topology[1]))
        layer2 = np.reshape(layer2, (-1, self.topology[2]))

        W1 = np.zeros(self.topology[0] * self.topology[1])
        W2 = np.zeros(self.topology[1] * self.topology[2])
        B1 = np.zeros(self.topology[1])
        B2 = np.zeros(self.topology[2])

        W1 = layer1[:, :-1]
        W2 = layer2[:, :-1]
        B1 = layer1[:, self.topology[1] - 1]
        B2 = layer2[:, self.topology[2] - 1]

        W1 = W1.ravel()
        W2 = W2.ravel()
        B1 = B1.ravel()
        B2 = B2.ravel()

        ind = np.concatenate([W1, W2, B1, B2])
        return ind

    def evaluate_newsolutions(self, speciesind, bestindividual, startindex, newpops):
        fitnesslist = np.zeros(newpops.shape[0])
        for i in range(0, newpops.shape[0]):
            temp = startindex
            for u in range(0, newpops[i].shape[0]):
                bestindividual[temp] = newpops[i][u]
                temp += 1
            fitnesslist[i] = self.net.ForwardFitnessPass(self.traindata,  self.ind_tonetworkready(bestindividual))

            # if(speciesind == 5):
            #     print fitnesslist[i]
            #     print bestindividual
        return fitnesslist

    def evalallspec(self):

        for i in range(len(self.allspecies)):
            for popindex in range(len(self.allspecies[i].populations)):
                individual = self.getind_with_best_networkready(i, popindex)
                fitness = self.net.ForwardFitnessPass(self.traindata, individual)
                self.allspecies[i].fitnesslist[popindex] = fitness

                if(fitness < self.allspecies[i].bestfit ):
                    self.allspecies[i].bestfit = fitness
                    self.allspecies[i].bestind = popindex

            # print 'Evaluated Species: ' + str(i)

        # self.MarkBestInSubPopulations()
    def evalspec(self, index):

        for popindex in range(len(self.allspecies[index].populations)):
            individual = self.getind_with_best_networkready(index, popindex)
            fitness = self.net.ForwardFitnessPass(self.traindata, individual)
            self.allspecies[index].fitnesslist[popindex] = fitness

            if(fitness < self.allspecies[index].bestfit ):
                self.allspecies[index].bestfit = fitness
                self.allspecies[index].bestind = popindex

        # print 'Evaluated Species: ' + str(i)

    # self.MarkBestInSubPopulations()


    def getind_bestonly(self, speciesind):

        ind = np.zeros((self.topology[0] *  self.topology[1])+ (self.topology[1] *  self.topology[2]) + self.topology[1] +  self.topology[2])
        speciesstartindex = 0
        speciesFound = 0
        index = 0
        for u in range(len(self.allspecies)):
            if u == speciesind:  # Only replace part of the best individual where this species is
                if speciesFound == 0:  # If this is the first chrome for the current species, mark starting index
                    speciesstartindex = index  # Set beginning index
                    speciesFound = 1

            for chrome in (self.allspecies[u].populations[self.allspecies[u].bestind]):
                ind[index] = chrome  # Append best from other species
                index += 1

        return [ind,speciesstartindex]

    def getind_with_best_networkready(self, speciesind, popind):

        ind = np.zeros((self.topology[0] *  self.topology[1])+ (self.topology[1] *  self.topology[2]) + self.topology[1] +  self.topology[2])
        speciesRange = np.zeros((2,), dtype=np.int)  # start and end index in individual
        speciesFound = 0
        index = 0
        for u in range(len(self.allspecies)):
            if u == speciesind:  # Only replace part of the best individual where this species is

                if speciesFound == 0:  # If this is the first chrome for the current species, mark starting index
                    speciesRange[0] = index  # Set beginning index
                    speciesFound = 1

                for chrome in (self.allspecies[u].populations[popind]):
                    ind[index] = chrome
                    index += 1

            else:
                for chrome in (self.allspecies[u].populations[self.allspecies[u].bestind]):
                    ind[index] = chrome  # Append best from other species
                    index += 1

        speciesRange[1] = speciesRange[0] + (len(self.allspecies[speciesind].populations[0]) - 1)  # Set endpoint

        layer1 = ind[0:(self.topology[0] *  self.topology[1]) + self.topology[1]]
        layer2 = ind[(self.topology[0] * self.topology[1]) + self.topology[1]:]
        layer1 = np.reshape(layer1, (-1, self.topology[1]))
        layer2 = np.reshape(layer2, (-1, self.topology[2]))

        W1 = np.zeros(self.topology[0] *  self.topology[1])
        W2 = np.zeros(self.topology[1] * self.topology[2])
        B1 =  np.zeros(self.topology[1])
        B2 = np.zeros(self.topology[2])

        W1 = layer1[:,:-1]
        W2 = layer2[:,:-1]
        B1 = layer1[:,self.topology[1] - 1]
        B2 = layer2[:, self.topology[2] - 1]

        W1 = W1.ravel()
        W2 = W2.ravel()
        B1 = B1.ravel()
        B2 = B2.ravel()

        ind = np.concatenate([W1,W2, B1, B2])
        return ind

def main():

    if os.path.exists('mcmcresults'):
        shutil.rmtree('mcmcresults/', ignore_errors=True)

    else:
        os.makedirs('mcmcresults')

    start = time.time()


    hidden = 5
    input = 4  #
    output = 1
    populationsize = 50

    # traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
    # testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
    #
    # traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
    # testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")


    # if problem == 1:
    #     traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
    # if problem == 2:

    # if problem == 3:
    traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
    testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #

    print(traindata)

    topology = [input, hidden, output]

    random.seed(time.time())

    runs = 1

    RUNRMSE = []

    net = Network(topology, traindata, testdata)

    graph = Graphing()

    for i in range(0, runs):
        coevolve = CoEvolution(topology,populationsize, traindata, testdata, 5000, net)
        coevolve.initialize()
        coevolve.evalallspec()
        coevolve.evolve(1, 1000)
        graph.PlotGraphs(net, i, testdata, traindata, topology)


    #
    # for i in range(0, runs):
    #     Procedure = Backpropagation()
    #     Procedure.Run(traindata, net, topology)
    #     graph.PlotGraphs(net, i,testdata,traindata,topology)


    print 'End simulation'
    end = time.time()
    print str(end - start) + ' Seconds'


if __name__ == "__main__": main()
