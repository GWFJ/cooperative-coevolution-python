# !/usr/bin/python


# Cooperative Coevolution

# Data (Sunspot and Lazer). Taken' Theorem used for Data Reconstruction (Dimension = 4, Timelag = 2).
# Data procesing file is included.

# RMSE (Root Mean Squared Error)

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk

# Gary Wong, University of the South PAcific


# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra


import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import os
import shutil






# An example of a class
class Network:
    def __init__(self, Topo, Train, Test):
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

    def RMSE_Er(self, targets):
        return np.sqrt((np.square( np.subtract(np.absolute(self.out),np.absolute(targets)))).mean())

    def RMSE(self, actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

        # error = np.subtract(abs(self.out), abs(actualout))
        # # sqerror = np.sum(np.square(error)) / self.Top[2]
        # rootsqerror = np.sqrt(np.sum(np.square(error)) / (len(error) * self.Top[2]))
        # return rootsqerror

    # def RMSE_Er(self, actualout):
    #     error = np.subtract(abs(self.out), abs(actualout))
    #     #sqerror = np.sum(np.square(error)) / self.Top[2]
    #     rootsqerror = np.sqrt(np.sum(np.square(error)) / (len(error) * self.Top[2]))
    #     return rootsqerror

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        #sqerror = np.sum(np.square(error)) / len(error)
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


    def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

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




class Species:
    def __init__(self, PopulationSize, SpeciesSize):
        self.Populations = [SpeciesPopulation(SpeciesSize) for count in xrange(PopulationSize)]
        self.NewPop = [SpeciesPopulation(SpeciesSize) for count in xrange(NPSize)]
        self.BestIndex = 0
        self.BestFitness = 0
        self.WorstIndex = 0
        self.WorstFitness = 0

        self.RandParents = np.zeros(2)

        self.TempIndex =  np.arange(0, PopulationSize)
        self.List = np.arange(0, PopulationSize)
        self.Mom = np.arange(0, PopulationSize)
        self.PopulationSize = PopulationSize
        self.SpeciesSize = SpeciesSize

        #Assign Initial Rank
        for i in xrange(len(self.Populations)):
            self.Populations[i].Rank = i

    def RandomParents(self):
        swp = self.TempIndex[0]
        self.TempIndex[0] = self.TempIndex[self.BestIndex]
        self.TempIndex[self.BestIndex] = swp

        for i in range(1, NumParents):
            index = random.randint(1, self.PopulationSize - 1)
            swp = self.TempIndex[index]
            self.TempIndex[index] = self.TempIndex[i]
            self.TempIndex[i] = swp
    def Mod(self, List):
        sum = 0
        for i in range(self.SpeciesSize):
            sum += (List[i] * List[i] )
        return np.sqrt(sum)

    def RandomAddition(self):
        chance = np.random.randint(0, 1)
        randomWeight, NegativeWeight = (0,0)
        if (chance == 0):
            randomWeight = np.random.randint(0, 100)
            return randomWeight * 0.009
        else:
            NegativeWeight = np.random.randint(0, 100)
            return NegativeWeight * -0.009

    def InnerProd(self, Ind1, Ind2):

        sum = 0.0

        for i in range(self.SpeciesSize):
            sum += (Ind1[i] * Ind2[i])

        return sum

    def GenerateNewPCX(self, Pass):
        Centroid = np.zeros(self.SpeciesSize)
        tempar1 = np.zeros(self.SpeciesSize)
        tempar2 = np.zeros(self.SpeciesSize)
        d = np.zeros(self.SpeciesSize)
        D = np.zeros(NumParents)

        temp1, temp2, temp3 = (0,0,0)

        diff = np.zeros((NumParents, self.SpeciesSize))

        for i in range(self.SpeciesSize):
            for u in range(NumParents):
                Centroid[i] += self.Populations[self.TempIndex[u]].Chromes[i]
            Centroid[i] /= NumParents


        for j in range(1, NumParents):
            for i in range(self.SpeciesSize):
                if j == 1:
                    d[i] = Centroid[i] - self.Populations[self.TempIndex[0]].Chromes[i]

                if(np.isnan(self.Populations[self.TempIndex[j]].Chromes[i] - self.Populations[self.TempIndex[0]].Chromes[i])):
                    print 'diff nan'
                    diff[j][i] = 1
                    return 0
                else:
                    diff[j][i] = self.Populations[self.TempIndex[j]].Chromes[i] - self.Populations[self.TempIndex[0]].Chromes[i]

            if (self.Mod(diff[j])  < EPSILON):
                # print "RUN Points are very close to each other. Quitting this run   "
                return 0


        dist = self.Mod(d)

        if (dist < EPSILON):
            # print "RUN Points are very close to each other. Quitting this run   "
            return 0

        # orthogonal directions are computed
        for j in range(1, NumParents):
            temp1 = self.InnerProd(diff[j].tolist(), d.tolist())

            if ((self.Mod(diff[j]) * dist) == 0):
                print "Division by zero"
                temp2 = temp1 / (1)
            else:
                temp2 = temp1 / (self.Mod(diff[j]) * dist)

            # Gary
            # if (temp2 == 1):
            #     temp2 = 0.99

            temp3 = 1.0 - np.power(temp2, 2)

            D[j] = self.Mod(diff[j]) * np.sqrt(np.abs(temp3))

        D_not = 0
        for i in range(1, NumParents):
            D_not += D[i]
        D_not /= (NumParents - 1) # this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector

        for j in range(self.SpeciesSize):
            tempar1[j] = np.random.normal(0, sigma_eta, 1) #rand_normal(0, D_not * sigma_eta);
            tempar2[j] = tempar1[j]

        for j in range(self.SpeciesSize):
            if(np.power(dist, 2) == 0):
                print " division by zero: part 2"
                tempar2[j] = tempar1[j] - ((self.InnerProd(tempar1, d) * d[j]) / 1)
            else:
                tempar2[j] = tempar1[j] - ((self.InnerProd(tempar1, d) * d[j]) / np.power(dist, 2.0))

        for j in range(self.SpeciesSize):
            tempar1[j] = tempar2[j]

        for k in range(self.SpeciesSize):
            self.NewPop[Pass].Chromes[k] = self.Populations[self.TempIndex[0]].Chromes[k] + tempar1[k]

        tempvar = np.random.normal(0, sigma_zeta, 1)

        for k in range(self.SpeciesSize):
            self.NewPop[Pass].Chromes[k] += (tempvar[0] * d[k])


        Chroomes = np.zeros(self.SpeciesSize)

        for k in range(self.SpeciesSize):
            if(np.isnan(self.NewPop[Pass].Chromes[k])):
                self.NewPop[Pass].Chromes[k] = self.RandomAddition()

            else:
                Chroomes = self.NewPop[Pass].Chromes[k]


        return 1

    def Family(self):

        swp = 0

        for i in range(family):
            randomIndex = random.randint(0, self.PopulationSize - 1)  # Get random index in population
            swp = self.Mom[randomIndex]
            self.Mom[randomIndex] = self.Mom[i]
            self.Mom[i] = swp



class SpeciesPopulation:
    def __init__(self, SpeciesSize):
        self.Likelihood = 0
        self.Fitness = 0
        # self.Chromes = np.random.randn(SpeciesSize)
        self.Chromes = np.random.uniform(-5,5,SpeciesSize)
        self.Rank = 99
class CoEvolution:
    def __init__(self, Topology, PopulationSize, samples, traindata, testdata):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = Topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.populationsize = PopulationSize

        #SpeciesSize keeps track of the size of each species
        #  self.SpeciesSize = np.ones(Hidden + Output) # No species determined by no hidden and output neurons
        self.IndividualSize =  (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]
        self.BestIndividual = np.zeros(self.IndividualSize)

        self.AllSpecies = []

        self.Kids = 0
        self.Parents = []

        for i in xrange(self.topology[1]):
            x = Species(self.populationsize, self.topology[0] + 1)
            self.AllSpecies.append(x)
        for i in xrange(self.topology[2]):
            x = Species(self.populationsize, self.topology[1] + 1)
            self.AllSpecies.append(x)

    def Sort(self, Pass):

        dbest = 99

        # Reset List
        for i in range(KIDS + family):
            self.AllSpecies[Pass].List[i] = i

        for i in range(KIDS + family - 1):
            dbest = self.AllSpecies[Pass].NewPop[self.AllSpecies[Pass].List[i]].Fitness

            for j in range(i + 1, KIDS + family):

                if(self.AllSpecies[Pass].NewPop[self.AllSpecies[Pass].List[j]].Fitness < dbest):
                    dbest = self.AllSpecies[Pass].NewPop[self.AllSpecies[Pass].List[j]].Fitness
                    temp = self.AllSpecies[Pass].List[j]
                    self.AllSpecies[Pass].List[j] = self.AllSpecies[Pass].List[i]
                    self.AllSpecies[Pass].List[i] = temp


    def ReplaceParents(self, Pass, Network):

        for j in range(family):
            self.AllSpecies[Pass].Populations[ self.AllSpecies[Pass].Mom[j]].Chromes =  self.AllSpecies[Pass].NewPop[ self.AllSpecies[Pass].List[j]].Chromes # Update population with new species

            [ind, speciesRange] = self.GetIndividualCombinedWithBest(Pass, self.AllSpecies[Pass].Mom[j])

            # pred_train = Network.evaluate_proposal(self.traindata, self.ToMCMCIndividual(ind))
            # fitness = neuralnet.sampleEr(pred_train)
            # fitness = Network.RMSE_Er(pred_train)
            fitness = Network.ForwardFitnessPass(self.traindata, self.ToMCMCIndividual(ind))
            self.AllSpecies[Pass].Populations[self.AllSpecies[Pass].Mom[j]].Fitness = fitness

    def FindParents(self, Pass, Network):
        self.AllSpecies[Pass].Family()

        for j in range(family):
            self.AllSpecies[Pass].NewPop[KIDS + j].Chromes = self.AllSpecies[Pass].Populations[self.AllSpecies[Pass].Mom[j]].Chromes
            [ind, speciesRange] = self.GetNewPopulationCombinedWithBest(Pass, KIDS + j)

            # pred_train = Network.evaluate_proposal(self.traindata, self.ToMCMCIndividual(ind))
            # fitness = neuralnet.sampleEr(pred_train)
            # fitness = Network.RMSE_Er(pred_train)
            fitness = Network.ForwardFitnessPass(self.traindata, self.ToMCMCIndividual(ind))
            self.AllSpecies[Pass].NewPop[KIDS + j].Fitness = fitness

    def PrintBestIndexes(self):
        tempStr = '['
        for i in range(len(self.AllSpecies)):
            tempStr += 'Species ' + str(i) + ': ' + str(self.AllSpecies[i].BestIndex) + ' '
        tempStr += ']'
        print tempStr

    def MarkBestInSubPopulations(self):
        for i in range(len(self.AllSpecies)):
            bestIndex = 0  # Set initial lowest
            bestFitness = self.AllSpecies[i].Populations[0].Fitness
            for y in range(len(self.AllSpecies[i].Populations)):
                #if   self.AllSpecies[i].Populations[y].Fitness < bestFitness:
                if  self.AllSpecies[i].Populations[y].Fitness < bestFitness:
                    bestIndex = y
                    bestFitness = self.AllSpecies[i].Populations[y].Fitness

            self.AllSpecies[i].BestIndex = bestIndex
            self.AllSpecies[i].BestFitness = bestFitness

    def MarkWorstInSubPopulations(self):
        for i in range(len(self.AllSpecies)):
            worstIndex = 0  # Set initial lowest
            worstFitness = 0
            for y in range(len(self.AllSpecies[i].Populations)):
                #if   self.AllSpecies[i].Populations[y].Fitness < bestFitness:
                if  self.AllSpecies[i].Populations[y].Fitness > worstFitness:
                    worstIndex = y
                    worstFitness = self.AllSpecies[i].Populations[y].Fitness

            self.AllSpecies[i].WorstIndex = worstIndex
            self.AllSpecies[i].WorstFitness = worstFitness
    def OrderSubPopulations(self):
        #This function assumes that populations are newly generated with Consecutive ranks. Population[0] = Rank 0, Population[1] = Rank 1 and so on
        for i in range(len(self.AllSpecies)):
            #bestIndex = 0  # Set initial lowest
            #bestFitness = self.AllSpecies[i].Populations[0].Fitness  # Set initial lowest
            for y in range(len(self.AllSpecies[i].Populations) - 1):
                min_idx = y
                for x in range(y + 1, len(self.AllSpecies[i].Populations)):
                    if self.AllSpecies[i].Populations[x].Fitness < self.AllSpecies[i].Populations[min_idx].Fitness:
                       min_idx = x
                if (self.AllSpecies[i].Populations[y].Rank < self.AllSpecies[i].Populations[min_idx].Rank ):  # Only swap ranks if next rank is less else leave as is
                    tempRank = self.AllSpecies[i].Populations[min_idx].Rank  # Take previous best index rank
                    self.AllSpecies[i].Populations[min_idx].Rank = self.AllSpecies[i].Populations[y].Rank  # Swap ranks
                    self.AllSpecies[i].Populations[y].Rank = tempRank



            #for x in range(len(self.AllSpecies[i].Populations)):
                    #  for y in range(len(self.AllSpecies[i].Populations) - 1 ):
                    #Check if next fitness is less than current. If it is, check rank and swap
                    #    if self.AllSpecies[i].Populations[y].Fitness < self.AllSpecies[i].Populations[y + 1].Fitness:
                        # Swap ranks
                    #     if(self.AllSpecies[i].Populations[y + 1].Rank < self.AllSpecies[i].Populations[y].Rank): #Only swap ranks if next rank is less else leave as is
                    #     tempRank = self.AllSpecies[i].Populations[y].Rank # Take previous best index rank
                    #    self.AllSpecies[i].Populations[y].Rank = self.AllSpecies[i].Populations[y + 1].Rank  # Swap ranks
                    #    self.AllSpecies[i].Populations[y + 1].Rank = tempRank
                    # else:
                        #Next fitness is less than current swap
                    #  if (self.AllSpecies[i].Populations[y + 1].Rank > self.AllSpecies[i].Populations[y].Rank):  # Only swap ranks if next rank is less else leave as is
                    #  tempRank = self.AllSpecies[i].Populations[y + 1].Rank  # Take previous best index rank
                    #  self.AllSpecies[i].Populations[y + 1].Rank = self.AllSpecies[i].Populations[ y].Rank  # Swap ranks
            #   self.AllSpecies[i].Populations[y].Rank = tempRank

            #Set best index and fitness
            for x in range(len(self.AllSpecies[i].Populations)):
                if self.AllSpecies[i].Populations[x].Rank == 0:
                    self.AllSpecies[i].BestIndex = x
                    self.AllSpecies[i].BestFitness = self.AllSpecies[i].Populations[x].Fitness


        print('Completed Population Sort by Fitness')
        #self.PrintPopulationFitness()
    def PrintPopulationFitness(self):

        #temp = ''
        #for i in range(len(self.AllSpecies)):  # Go through each species
          #  temp += 'Species ' + str(i) + ' | '

        temp2 = ''

        for y in range(self.populationsize):  #Go through each population
            temp = ''
            for i in range(len(self.AllSpecies)): #Go through each species
                if y == self.AllSpecies[i].BestIndex:
                    temp += 'F:' + str(self.AllSpecies[i].Populations[y].Fitness) + ' B |   '



                    temp2 += "["

                    temp2 += 'F:' + str(self.AllSpecies[i].Populations[y].Fitness) + ' B |   '

                    for t in range(len(self.AllSpecies[i].Populations[y].Chromes)):
                        temp2 += str(self.AllSpecies[i].Populations[y].Chromes[t]) + ' '
                    temp2 += "]"

                else:
                    temp += 'F:' + str(self.AllSpecies[i].Populations[y].Fitness) + '   |   '


            print temp
        print temp2

    def EvaluateSpeciesByFitness(self, neuralnet):

        # Go through all SpeciesPopulation
        #   Conjoin with best from other Species
        #   Perform a forward pass and assign fitness
        individual = []

        for i in range(len(self.AllSpecies)):
            for popindex in range(len(self.AllSpecies[i].Populations)):
                [individual, speciesRange] = self.GetIndividualCombinedWithBest(i,popindex)
                # pred_train = neuralnet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(individual))
                #fitness = neuralnet.sampleEr(pred_train)
                # fitness = neuralnet.RMSE_Er(pred_train)
                fitness = neuralnet.ForwardFitnessPass(self.traindata, self.ToMCMCIndividual(individual))
                self.AllSpecies[i].Populations[popindex].Fitness = fitness
                individual = [] # reset individual
            print 'Evaluated Species: ' + str(i)

        self.MarkBestInSubPopulations()
        #self.PrintPopulationFitness()
        #self.OrderSubPopulations()

       # w = self.GetBestIndividual()
       # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
       # fitness = neuralnet.sampleEr(pred_train)

    def GetIndividualCombinedWithBest(self, SpeciesIndex, PopulationIndex):

        combinedWithBestIndividual = []
        speciesRange = np.zeros((2,), dtype=np.int)  # start and end index in individual
        speciesFound = 0
        index = 0
        for u in range(len(self.AllSpecies)):
            if u == SpeciesIndex: #Only replace part of the best individual where this species is

                if speciesFound == 0: #If this is the first chrome for the current species, mark starting index
                    speciesRange[0] = index #Set beginning index
                    speciesFound = 1

                for chrome in (self.AllSpecies[u].Populations[PopulationIndex].Chromes):
                    combinedWithBestIndividual.append(chrome)
                    index += 1

            else:
                for chrome in (self.AllSpecies[u].Populations[self.AllSpecies[u].BestIndex].Chromes):
                    combinedWithBestIndividual.append(chrome) #Append best from other species
                    index += 1

        speciesRange[1] = speciesRange[0] + (len(self.AllSpecies[SpeciesIndex].Populations[0].Chromes) - 1) # Set endpoint

        return [combinedWithBestIndividual, speciesRange]

    def GetNewPopulationCombinedWithBest(self, SpeciesIndex, PopulationIndex):
        combinedWithBestIndividual = []
        speciesRange = np.zeros((2,), dtype=np.int)  # start and end index in individual
        speciesFound = 0
        index = 0
        for u in range(len(self.AllSpecies)):
            if u == SpeciesIndex:  # Only replace part of the best individual where this species is

                if speciesFound == 0:  # If this is the first chrome for the current species, mark starting index
                    speciesRange[0] = index  # Set beginning index
                    speciesFound = 1

                for chrome in (self.AllSpecies[u].NewPop[PopulationIndex].Chromes):
                    combinedWithBestIndividual.append(chrome)
                    index += 1

            else:
                for chrome in (self.AllSpecies[u].Populations[self.AllSpecies[u].BestIndex].Chromes):
                    combinedWithBestIndividual.append(chrome)  # Append best from other species
                    index += 1

        speciesRange[1] = speciesRange[0] + (
                    len(self.AllSpecies[SpeciesIndex].Populations[0].Chromes) - 1)  # Set endpoint

        return [combinedWithBestIndividual, speciesRange]
    def ExtractPopulationFromIndividual(self, SpeciesIndex, Individual):
        population = []
        index = 0
        for u in range(len(self.AllSpecies)):
            if u == SpeciesIndex:
                for chrome in (self.AllSpecies[u].Populations[0].Chromes):
                    population.append(Individual[index])
                    index += 1
                break

            else:
                index += len(self.AllSpecies[u].Populations[0].Chromes)
        return population

    def ReplaceRandomIndexesInSpecies(self, Individual):
        index = 0
        for u in range(len(self.AllSpecies)):
            randomIndex = random.randint(0, self.populationsize - 1)  # Get random index in population
            for chrome in (self.AllSpecies[u].Populations[randomIndex].Chromes):
                chrome = Individual[index]
                index += 1
            self.AllSpecies[u].BestIndex = randomIndex

    def ReplaceWorstIndexesInSpecies(self, Individual):

        self.MarkWorstInSubPopulations()

        index = 0
        for u in range(len(self.AllSpecies)):

            worstIndex = self.AllSpecies[u].WorstIndex
            for chrome in (self.AllSpecies[u].Populations[worstIndex].Chromes):
                chrome = Individual[index]
                index += 1

    def GetBestIndividual(self):
        # Make first individual the best
        bestIndividual = []
        for i in range(len(self.AllSpecies)):
            for y in range(len(self.AllSpecies[i].Populations[self.AllSpecies[i].BestIndex].Chromes)):
                bestIndividual.append(self.AllSpecies[i].Populations[self.AllSpecies[i].BestIndex].Chromes[y])
        self.BestIndividual = bestIndividual
        return bestIndividual

    # ------------------------------------------------------------------------------------------------------------------
    #   This method updates n number of populations in a species
    #   It combines each population with the best from other populations
    #   Performs MCMC for a number of samples
    #   Breaks the new weights back into species size
    #   Replaces currently selected population with updated weights from MCMC
    #   Evaluate fitness, assign fitness and recheck best fitness in population
    # ------------------------------------------------------------------------------------------------------------------

    def PopulationsRandomWalk_MCMC_Populations(self, NumIndexes, Cycles, NeuralNet):
        dmcmc = DecomposedMCMC(self.samples, self.traindata, self.testdata, self.topology)

        weights = []
        bestWeightsOutputInCycles_Train = []
        bestWeightsOutputInCycles_Test = []
        weightsAtEndOfCycles = []
        fitnessAtEndOfCycles = []
        populationDataAtEndOfCycles = []

        for h in range(Cycles):
            print 'Cycle ' + str(h)
            for u in range(len(self.AllSpecies)):
                #print 'Species ' + str(u) + ' ----------------------------------'
                for i in range(NumIndexes):
                    #print 'Index ' + str(i)

                    randomIndex = random.randint(0, self.populationsize - 1) #Get random index in population
                    #1. Update random index chromes
                    #2. Get fitness - combine with unchanged best

                    [individual, speciesRange] = self.GetIndividualCombinedWithBest(u,randomIndex)
                    modifiedIndividual = dmcmc.sampler(self.ToMCMCIndividual(individual), NeuralNet, 0)
                    #modifiedIndividual = dmcmc.samplerPopulationProposals(individual, NeuralNet, 0, speciesRange)

                    # ytraindata = self.traindata[:, self.topology[0]]
                    extractedChromes = self.ExtractPopulationFromIndividual(u, self.ToDecomposedIndividual(modifiedIndividual))
                    # Replace random index population
                    self.AllSpecies[u].Populations[randomIndex].Chromes = extractedChromes
                    # Update fitness by performing a forward pass
                    [ind, speciesRange] = self.GetIndividualCombinedWithBest(u, randomIndex)
                    # pred_train = NeuralNet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(ind))
                    #fitness = NeuralNet.sampleEr(pred_train)
                    # fitness = NeuralNet.RMSE_Er(pred_train)
                    fitness = NeuralNet.ForwardFitnessPass(self.traindata, self.ToMCMCIndividual(ind))
                    self.AllSpecies[u].Populations[randomIndex].Fitness = fitness



                #After every species, evaluate best in species


                self.AllSpecies[u].BestIndex = 0
                self.AllSpecies[u].BestFitness = self.AllSpecies[u].Populations[0].Fitness


                for t in range(1, self.populationsize):
                    if(self.AllSpecies[u].Populations[t].Fitness < self.AllSpecies[u].BestFitness):
                        tempfit = self.AllSpecies[u].Populations[t].Fitness
                        self.AllSpecies[u].BestIndex = t
                        self.AllSpecies[u].BestFitness = tempfit


                    #Assign fitness?

            #Evaluate all species
            #self.EvaluateSpeciesByFitness(NeuralNet) #Assign new fitness
            #self.MarkBestInSubPopulations() #Mark fittest populations

            #Show decreasing fitness
            individual = self.GetBestIndividual()
            pred_train = NeuralNet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(individual))
            pred_test = NeuralNet.evaluate_proposal(self.testdata, self.ToMCMCIndividual(individual))
            #fitness = NeuralNet.sampleEr(pred_train)
            fitness = NeuralNet.RMSE_Er(pred_train)
            print 'Fitness: ' + str(fitness)
            self.PrintBestIndexes()

            #These are for graph data at the end of the simulation
            weightsAtEndOfCycles.append(self.ToMCMCIndividual(individual))
            fitnessAtEndOfCycles.append(fitness)
            populationDataAtEndOfCycles.append(self.AllSpecies)
            bestWeightsOutputInCycles_Train.append(pred_train)
            bestWeightsOutputInCycles_Test.append(pred_test)

        weights.append(bestWeightsOutputInCycles_Train) #Output[0]
        weights.append(bestWeightsOutputInCycles_Test)  #Output[1]
        weights.append(weightsAtEndOfCycles)            #Output[2]
        weights.append(fitnessAtEndOfCycles)            #Output[3]
        weights.append(populationDataAtEndOfCycles)     #Output[4]

        return weights

    def EvaluateNewPopulation(self, Pass, NeuralNet, SpeciesIndex):
        [ind, speciesRange] = self.GetNewPopulationCombinedWithBest(SpeciesIndex, Pass)
        # pred_train = NeuralNet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(ind))
        #
        # fitness = NeuralNet.RMSE_Er(pred_train)
        fitness = NeuralNet.ForwardFitnessPass(self.traindata, self.ToMCMCIndividual(ind))
        self.AllSpecies[SpeciesIndex].NewPop[Pass].Fitness = fitness


    def EvolveSubpopulations(self, Repetitions, Cycles, NeuralNet):

        self.Kids = KIDS
        tempfit = 99

        prevfitness = 99


        dmcmc = DecomposedMCMC(self.samples, self.traindata, self.testdata, self.topology)

        weights = []
        bestWeightsOutputInCycles_Train = []
        bestWeightsOutputInCycles_Test = []
        weightsAtEndOfCycles = []
        fitnessAtEndOfCycles = []
        populationDataAtEndOfCycles = []

        for h in range(Cycles):
            print 'Cycle ' + str(h)
            for u in range(len(self.AllSpecies)):

                for i in range(Repetitions):
                    tempfit = self.AllSpecies[u].BestFitness
                    self.AllSpecies[u].RandomParents()

                    for t in range(KIDS):
                        tag = self.AllSpecies[u].GenerateNewPCX(t)

                        if (tag == 0):
                            break

                    #Evaluate new population
                    for t in range(KIDS):
                        self.EvaluateNewPopulation(t,NeuralNet, u)


                    self.FindParents(u, NeuralNet)
                    self.Sort(u)
                    self.ReplaceParents(u, NeuralNet)





                    #After every species, evaluate best in species

                    self.AllSpecies[u].BestIndex = 0
                    self.AllSpecies[u].BestFitness = self.AllSpecies[u].Populations[0].Fitness


                    for t in range(1, self.populationsize):
                        if(self.AllSpecies[u].Populations[t].Fitness < self.AllSpecies[u].BestFitness):

                            self.AllSpecies[u].BestIndex = t
                            self.AllSpecies[u].BestFitness = tempfit

                    #print 'temp'


            #Show decreasing fitness
            individual = self.GetBestIndividual()
            pred_train = NeuralNet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(individual))
            pred_test = NeuralNet.evaluate_proposal(self.testdata, self.ToMCMCIndividual(individual))

            train_fitness = NeuralNet.ForwardFitnessPass(self.traindata, self.ToMCMCIndividual(individual))
            #fitness = NeuralNet.sampleEr(pred_train)
            # fitness = NeuralNet.RMSE_Er(pred_train)

            if (prevfitness < train_fitness):
                print   'stop'
            else:
                prevfitness = train_fitness


            print 'Fitness: ' + str(train_fitness)
            self.PrintBestIndexes()

            #These are for graph data at the end of the simulation
            weightsAtEndOfCycles.append(self.ToMCMCIndividual(individual))
            fitnessAtEndOfCycles.append(train_fitness)
            populationDataAtEndOfCycles.append(self.AllSpecies)
            bestWeightsOutputInCycles_Train.append(pred_train)
            bestWeightsOutputInCycles_Test.append(pred_test)

        weights.append(bestWeightsOutputInCycles_Train) #Output[0]
        weights.append(bestWeightsOutputInCycles_Test)  #Output[1]
        weights.append(weightsAtEndOfCycles)            #Output[2]
        weights.append(fitnessAtEndOfCycles)            #Output[3]
        weights.append(populationDataAtEndOfCycles)     #Output[4]

        return weights

    def ToDecomposedIndividual(self, ind):

        individual = None
        if type(ind) is np.ndarray:
            individual = ind.tolist()
        else:
            individual = ind

        WeightsSize = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2])
        BiasSize = self.topology[1] + self.topology[2]
        Decomposed_Individual = []
        current_index = 0

        Species_Weights = []
        Weights = individual[0:WeightsSize]
        Biases = individual[WeightsSize:WeightsSize + BiasSize]

        for i in range(len(self.AllSpecies)):
            fromIndex = current_index
            endIndex = fromIndex + len(self.AllSpecies[i].Populations[0].Chromes) - 1
            Species_Weights.append(individual[fromIndex:endIndex])
            current_index = endIndex

        for i in range(len(Biases)):
            Species_Weights[i].append(Biases[i])

        # Create final individual
        for i in range(len(Species_Weights)):
            for t in range(len(Species_Weights[i])):
                Decomposed_Individual.append(Species_Weights[i][t])

        return Decomposed_Individual

    def ToMCMCIndividual(self, individual):

        W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0])
        B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer

        w_layer1size = self.topology[0] * self.topology[1]
        w_layer2size = self.topology[1] * self.topology[2]

        W1_CCSize = (self.topology[0] * self.topology[1]) + self.topology[1]
        W2_CCSize = (self.topology[1] * self.topology[2]) + self.topology[2]

        Layer1 = np.reshape(individual[0:W1_CCSize],(self.topology[1], self.topology[1]))
        Layer2 = np.reshape(individual[W1_CCSize:len(individual)],(self.topology[2], self.topology[1] + 1))

        MCMC_Individual = []

        # Copy Layer 1 Weights
        for u in range(len(W1)):
            for t in range(len(Layer1)):
                W1[u][t] = Layer1[t][u]
                MCMC_Individual.append(W1[u][t])

        # Copy Layer 2 Weights
        for u in range(len(W2)):
            for t in range(len(Layer2)):
                W2[u][t] = Layer2[t][u]
                MCMC_Individual.append(W2[u][t])


        # Copy Layer 1 Bias

        for u in range(len(B1)):
            for t in range(len(Layer1)):
                B1[u][t] = Layer1[t][len(Layer1[t]) - 1]
                MCMC_Individual.append(B1[u][t])

        # Copy Layer 2 Bias

        for u in range(len(B2)):
            for t in range(len(Layer2)):
                B2[u][t] = Layer2[t][len(Layer2[t]) - 1]
                MCMC_Individual.append(B2[u][t])




        # W1 = np.reshape((self.topology[0], self.topology[1]))
        # W2 = np.reshape((self.topology[1], self.topology[2]))
        #
        # Weights = []
        # Biases = []
        # current_index = 0
        #
        #
        # for i in range(len(self.AllSpecies)):
        #     fromIndex = current_index
        #     endIndex = fromIndex + len(self.AllSpecies[i].Populations[0].Chromes) - 1
        #     Weights.append(individual[fromIndex:endIndex])
        #     Biases.append(individual[endIndex])
        #     current_index = endIndex + 1
        #
        # for i in range(len(Weights)):
        #     for t in range(len(Weights[i])):
        #         MCMC_Individual.append(Weights[i][t])
        #
        # for i in range(len(Biases)):
        #     MCMC_Individual.append(Biases[i])

        return MCMC_Individual

class DecomposeProcedure:
    def __init__(self, samples, traindata, testdata, topology, populationSize):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.populationSize = populationSize

    #-----------------------------------------------------------#
    #            Main Decomposed MCMC Procedure                 #
    # ----------------------------------------------------------#

    def Procedure(self, run):
        neuronLevel = CoEvolution(self.topology, self.populationSize, self.samples, self.traindata, self.testdata )
        #neuronLevel.EvaluateSpecies()

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]

        neuralnet = Network(self.topology, self.traindata, self.testdata)

        neuronLevel.EvaluateSpeciesByFitness(neuralnet)

        initial_w = neuronLevel.GetBestIndividual()

        output = neuronLevel.EvolveSubpopulations(1, 100, neuralnet)  # (NumIndexes, Cycles, NeuralNet):


        #
        # # -------------------------------------
        #
        # #   Initially, best index are all set to 0. During initial fitness evaluation lets run MCMC for a number of samples
        # #   where we combine all index 0 into an individual and run to initialize direction
        # dmcmc = DecomposedMCMC(self.samples, self.traindata, self.testdata, self.topology)
        # ini = neuronLevel.GetBestIndividual()
        #
        # # temp1 = neuronLevel.ToMCMCIndividual(ini)
        # # temp2 = neuronLevel.ToDecomposedIndividual(temp1)
        #
        #
        # # initial_w = dmcmc.sampler(ini, neuralnet, 1000) #Update initial best
        # neuronLevel.EvaluateSpeciesByFitness(neuralnet)
        #
        #
        #
        # output = neuronLevel.EvolveSubpopulations(20,20,neuralnet) #(NumIndexes, Cycles, NeuralNet):
        # #neuronLevel.PopulationsRandomWalk_MCMC_BestOnly(10, 20, neuralnet)  # (NumIndexes, Cycles, NeuralNet):


        self.PlotGraphs(neuronLevel.ToMCMCIndividual(neuronLevel.GetBestIndividual()), neuralnet,neuronLevel.ToMCMCIndividual(initial_w),run, output)

        neuronLevel.PrintPopulationFitness()
        # self.PlotGraphs(initial_w, neuralnet, ini)

        return output[3][len(output[3]) - 1] # return final population best fitness

    def PlotGraphs(self, w, net, initial_w, run, output):
        print 'New and Old Weights '
        print w
        print initial_w
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        ytestdata = self.testdata[:, self.topology[0]]
        ytraindata = self.traindata[:, self.topology[0]]

        initial_trainout = net.evaluate_proposal(self.traindata,initial_w)
        trainout = net.evaluate_proposal(self.traindata,w)

        # print 'Initial Trainout'
        # print initial_trainout
        # print 'Trainout'
        # print trainout


        initial_testout = net.evaluate_proposal(self.testdata, initial_w)
        testout = net.evaluate_proposal(self.testdata,w)

        #Print train

        for i in range(len(output[0])):
            plt.plot(x_train, output[0][i], label='', color="black", lw=0.5, ls = '-')

        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, trainout, label='predicted (train)')
        plt.plot(x_train, initial_trainout, label='initial (train)')




        plt.legend(loc='upper right')
        plt.title("Plot of Train Data vs MCMC Uncertainty ")

        if not os.path.exists('mcmcresults/run' + str(run) + '/'):
            os.makedirs('mcmcresults/run' + str(run)+ '/')

        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_train.png')
        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_train.svg', format='svg', dpi=600)
        plt.clf()

        # Print test

        for i in range(len(output[1])):
            plt.plot(x_test, output[1][i], label='', color="black", lw=0.5, ls = '-')

        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, testout, label='predicted (test)')
        plt.plot(x_test, initial_testout, label='initial (test)')
        plt.legend(loc='upper right')
        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_test.png')
        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_test.svg', format='svg', dpi=600)
        plt.clf()


        colors = (0, 0, 0)
        area = np.pi * 3

        # Plot Weights


        for i in range(len(output[2])):

            # Plot Best Individual at the end of cycle
            x_weights = np.linspace(0, 1, num=len(output[2][0]))
            plt.plot(x_weights, output[2][i],'bo', x_weights, output[2][i],'k',label='')
            plt.legend(loc='upper right')
            plt.title('Best Solution at the end of [Cycle: ' + str(i) + ' with Fitness: ' + format(output[3][i], '.5f') + ']')
            plt.savefig('mcmcresults/run' + str(run) + '/weightsCycle'+str(i)+'.png')
            plt.savefig('mcmcresults/run' + str(run) + '/weightsCycle'+str(i)+'.svg', format='svg', dpi=600)
            plt.clf()

            # Plot population data at the end of cycle

            #----------------------------------------------------------------
            # outputCyclePopulations = 1
            #
            # if outputCyclePopulations == 1:
            #     clust_data = []
            #
            #     for y in range(self.populationSize):  # Go through each population
            #         temp = []
            #         for l in range(len(output[4][i])):  # Go through each species
            #             if y == output[4][i][l].BestIndex:
            #                 temp.append('F:' + str(output[4][i][l].Populations[y].Fitness) + ' B ')
            #             else:
            #                 temp.append('F:' + str(output[4][i][l].Populations[y].Fitness))
            #
            #         clust_data.append(temp)
            #
            #     colLabels = []
            #     for p in range(len(output[4][i])):
            #         colLabels.append('Species ' + str(p))
            #
            #     nrows, ncols = len(clust_data) + 1, len(colLabels)
            #     hcell, wcell = 0.3, 1.
            #     hpad, wpad = 0, 0
            #     fig = plt.figure(figsize=(ncols * wcell + wpad, nrows * hcell + hpad))
            #     ax = fig.add_subplot(111)
            #     ax.axis('off')
            #     # do the table
            #     the_table = ax.table(cellText=clust_data,
            #                          colLabels=colLabels,
            #                          loc='center')
            #     plt.savefig('mcmcresults/run' + str(run) + '/populationsData' + str(i) + '.png')
            #     plt.savefig('mcmcresults/run' + str(run) + '/populationsData' + str(i) + '.svg', format='svg', dpi=600)
            #     plt.close(fig)
            #     plt.clf()

            # ----------------------------------------------------------------



class DecomposedMCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self, Individual, neuralnet, initialsamples):
        #print 'Begin sampling '
        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        #print y_train.size
        #print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        #w = np.random.randn(w_size)
        w = Individual
        w_proposal = np.random.randn(w_size)

        #step_w = 0.02;  # defines how much variation you need in changes to w
        step_w = 0.1  # defines how much variation you need in changes to w
        step_eta = 0.01
        # --------------------- Declare FNN and initialize


        # print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w,
                                                 tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        #print likelihood

        #naccept = 0
        #print 'begin sampling using mcmc random walk'
        #plt.plot(x_train, y_train)
        #plt.plot(x_train, pred_train)
        #plt.title("Plot of Data vs Initial Fx")
        #plt.savefig('mcmcresults/begin.png')
        # plt.clf()

        # plt.plot(x_train, y_train)

        samp = samples

        if initialsamples != 0:
            samp = initialsamples

        for i in range(samp - 1):
            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata,
                                                                                w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata,
                                                                            w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

                u = random.uniform(0, 1)

                if u < mh_prob:
                    # Update position
                    #print    i, ' is accepted sample'
                    #naccept += 1
                    likelihood = likelihood_proposal
                    prior_likelihood = prior_prop
                    w = w_proposal
                    eta = eta_pro


                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # # fitness = NeuralNet.sampleEr(pred_train)
                    # fitness = neuralnet.RMSE_Er(pred_train)
                    # print 'Fitness: ' + str(fitness)

                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # fitness = neuralnet.RMSE_Er(pred_train)
                    print 'Fitness: ' + str(rmsetrain)

                    # fxtrain_samples[i + 1,] = pred_train
                    # fxtest_samples[i + 1,] = pred_test
            except Exception:
                print '######################### OverflowError: math range error: Skipping current sample'
                pass  # or you could use 'continue'


        return w  # return as soon as we get an accepted sample

    def samplerPopulationProposals(self, Individual, neuralnet, initialsamples, speciesRange):

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        #print y_train.size
        #print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        #w = np.random.randn(w_size)
        w = Individual
        w_proposal = self.replaceChromesIndividual(w, np.random.randn(w_size), speciesRange)

        #step_w = 0.02;  # defines how much variation you need in changes to w
        step_w = 0.002  # defines how much variation you need in changes to w
        step_eta = 0.01
        # --------------------- Declare FNN and initialize


        # print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w,
                                                 tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        #print likelihood

        #naccept = 0
        #print 'begin sampling using mcmc random walk'
        #plt.plot(x_train, y_train)
        #plt.plot(x_train, pred_train)
        #plt.title("Plot of Data vs Initial Fx")
        #plt.savefig('mcmcresults/begin.png')
        # plt.clf()

        # plt.plot(x_train, y_train)

        samp = samples

        if initialsamples != 0:
            samp = initialsamples

        for i in range(samp - 1):
            #w_proposal = w +
            w_proposal = self.replaceChromesIndividual(w, np.random.normal(0, step_w, w_size), speciesRange) #(old,new,range)


            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata,
                                                                                w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata,
                                                                            w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood
            # print 'diff_likelihood: ' + str(diff_likelihood)
            # print 'diff_priorliklihood: ' + str(diff_priorliklihood)

            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

                u = random.uniform(0, 1)

                if u < mh_prob:
                    # Update position
                    # print    i, ' is accepted sample'
                    # naccept += 1
                    likelihood = likelihood_proposal
                    prior_likelihood = prior_prop
                    w = w_proposal
                    eta = eta_pro

                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # # fitness = NeuralNet.sampleEr(pred_train)
                    # fitness = neuralnet.RMSE_Er(pred_train)
                    # print 'Fitness: ' + str(fitness)

                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # fitness = neuralnet.sampleEr(pred_train)
                    # print 'Fitness: ' + str(fitness)
            except Exception:
                print '######################### OverflowError: math range error: Skipping current sample'
                pass  # or you could use 'continue'

        return w  # return as soon as we get an accepted sample

    #This function takes in two individuals, a previous and a new. It replaces all chromes from starting index (speciesRange[0]) to end index (speciesRange[1])
    def replaceChromesIndividual(self, originalIndividual, newIndividual, speciesRange):
        for i in range(speciesRange[0], speciesRange[1]):
            originalIndividual[i] = newIndividual[i]

        return originalIndividual


def main():
    global NumParents, EPSILON, sigma_eta, NPSize, KIDS, sigma_zeta, family
    KIDS = 2
    NumParents = 3
    EPSILON = 1e-50
    sigma_eta = 0.1
    sigma_zeta = 0.1
    NPSize = KIDS + 2
    family = 2


    if os.path.exists('mcmcresults'):
        shutil.rmtree('mcmcresults/', ignore_errors=True)
        os.makedirs('mcmcresults')

    else:
        os.makedirs('mcmcresults')

        # shutil.rmtree('mcmcresults/', ignore_errors=True)

    # os.makedirs('mcmcresults')

    start = time.time()


    outres = open('mcmcresults/resultspriors.txt', 'w')
    hidden = 5
    input = 4  #
    output = 1

    populationSize = 10
    traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
    testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
    # if problem == 1:
    #     traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
    # if problem == 2:

    # if problem == 3:
    #     traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #

    print(traindata)

    topology = [input, hidden, output]

    MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)

    random.seed(time.time())

    numSamples = 20  # need to decide yourself 80000

    # 2 for population
    #1000 for bestonly
    # for best only increase poulation size

    runs = 5

    RUNRMSE = []

    for i in range(0, runs):
        strategy = DecomposeProcedure(numSamples, traindata, testdata, topology, populationSize)
        RUNRMSE.append(strategy.Procedure(i))

    Errors = np.zeros(runs).tolist()
    # Errors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x_train = range(0, (runs + 1))
    y_pos = np.arange((runs + 1))

    MeanRMSE = np.mean(RUNRMSE)
    MeanRMSESTD = np.std(RUNRMSE)

    RUNRMSE.append(MeanRMSE)
    Errors.append(MeanRMSESTD)

    plt.bar(y_pos, RUNRMSE, align='center', alpha=0.5)
    plt.errorbar(
        x_train,  # X
        RUNRMSE,  # Y
        yerr=Errors,  # Y-errors
        label="Error bars plot",
        fmt="gs",  # format line like for plot()
        linewidth=1  # width of plot line
    )
    plt.xticks(y_pos, x_train)
    plt.ylabel('Train RMSE')
    plt.ylabel('Run')
    plt.title('Programming language usage')
    plt.savefig('mcmcresults/AverageResults.png')
    plt.savefig('mcmcresults/AverageResults.svg', format='svg', dpi=600)
    plt.clf()

    # Plot Average RUN RMSE


    # mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class
    #
    # [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
    # print 'sucessfully sampled'
    #
    # burnin = 0.1 * numSamples  # use post burn in samples
    #
    # pos_w = pos_w[int(burnin):, ]
    # pos_tau = pos_tau[int(burnin):, ]
    #
    # fx_mu = fx_test.mean(axis=0)
    # fx_high = np.percentile(fx_test, 95, axis=0)
    # fx_low = np.percentile(fx_test, 5, axis=0)
    #
    # fx_mu_tr = fx_train.mean(axis=0)
    # fx_high_tr = np.percentile(fx_train, 95, axis=0)
    # fx_low_tr = np.percentile(fx_train, 5, axis=0)
    #
    # rmse_tr = np.mean(rmse_train[int(burnin):])
    # rmsetr_std = np.std(rmse_train[int(burnin):])
    # rmse_tes = np.mean(rmse_test[int(burnin):])
    # rmsetest_std = np.std(rmse_test[int(burnin):])
    # print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
    # np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')
    #
    # ytestdata = testdata[:, input]
    # ytraindata = traindata[:, input]
    #
    # plt.plot(x_test, ytestdata, label='actual')
    # plt.plot(x_test, fx_mu, label='pred. (mean)')
    # plt.plot(x_test, fx_low, label='pred.(5th percen.)')
    # plt.plot(x_test, fx_high, label='pred.(95th percen.)')
    # plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
    # plt.legend(loc='upper right')
    #
    # plt.title("Plot of Test Data vs MCMC Uncertainty ")
    # plt.savefig('mcmcresults/mcmcrestest.png')
    # plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
    # plt.clf()
    # # -----------------------------------------
    # plt.plot(x_train, ytraindata, label='actual')
    # plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
    # plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
    # plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
    # plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
    # plt.legend(loc='upper right')
    #
    # plt.title("Plot of Train Data vs MCMC Uncertainty ")
    # plt.savefig('mcmcresults/mcmcrestrain.png')
    # plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
    # plt.clf()
    #
    # mpl_fig = plt.figure()
    # ax = mpl_fig.add_subplot(111)
    #
    # ax.boxplot(pos_w)
    #
    # ax.set_xlabel('[W1] [B1] [W2] [B2]')
    # ax.set_ylabel('Posterior')
    #
    # plt.legend(loc='upper right')
    #
    # plt.title("Boxplot of Posterior W (weights and biases)")
    # plt.savefig('mcmcresults/w_pos.png')
    # plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)
    #
    # plt.clf()

    print 'End simulation'
    end = time.time()
    print str(end - start) + ' Seconds'


if __name__ == "__main__": main()
