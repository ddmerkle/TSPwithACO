# Authors: Dawson Merkle : Thomas Levitt
# Date: 6/22/23
# Assignment 3
# CAP 4630


# Steps to the Algorithm
# 1. Set up Pheromones
# 2. Set up the Population of Ants
# 3. Choose the Next Visit for each Ant
# 4. Are there more Destinations?
    # If yes, go back to 3
    # If no, move to 5
# 5. Update the Pheromone Trails
# 6. Update the Best Solution
# 7. Reached Stopping Condition?
    # If no, go back to 2
    # If yes, move to 8
# 8. Return the Best Solution



# Hyperparameters

# Number of ants (usually the number of cities)
# Number of iterations
# Alpha
# Beta
# Evaporation rate


# Necessary things

# Going to need an adjacency matrix for the distances from each city
# Going to need an adjacency matrix for the pheromones
# Need a best solution






import random
import numpy as np
import matplotlib.pyplot as plt


class City():
    '''city class'''

    def __init__(self, name, x, y):
        '''constructor'''
        self.name = name
        self.x = x
        self.y = y
    
    def get_distance(self, other):
        '''gets distance from one city from another'''
        xDis = abs(self.x - other.x)
        yDis = abs(self.y - other.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def get_coords(self):
        '''returns tuple of x and y'''
        return self.x, self.y
    
    def __repr__(self):
        '''returns string representation of city'''

        return "({name}: x:{x}, y:{y})".format(name = self.name, 
                                         x = self.x, y = self.y)
    

class Ant():
    '''ant class'''

    def __init__(self, name, startPoint):
        '''constructor'''
    
        self.name = name
        self.startPoint = startPoint
        self.visitLis = []
        
        #defines starting point in list
        self.visitLis.append(self.startPoint)
    
    # def visited_cities(self, cityObject)
    
    # def get_distance_traveled(self):
    
    def __repr__(self):
        '''returns string representation of ant'''
    
        return '({name}: list: {visitLis})'.format(name = self.name, 
                                                   visitLis = self.visitLis)

#################################################


# def visit_city(self, pheromoneTrails):
# def visit_random_city(self):
# def visit_probabilistic_city(self, pheromonTrails):
# def roulette_wheel_selection(self, probabilities):







def proportional_roulette_selection(population, normProbabilities):
    '''Selecting parents using proportional roulette selection'''

    slices = []
    for slice in range(len(normProbabilities)):
        sliceCounter = 0
        if slice == 0:
            slices.append(normProbabilities[0])
        else:

            slices.append(slices[sliceCounter-1] + normProbabilities[slice])

    randomNumber = random.uniform(0,1)

    for index, slice in enumerate(slices):
        if randomNumber < slice:
            return population[index]
        

def create_colony(cityList, NUM_ANT_FACTOR, NUM_CITY):

    antColony = []
    for antCount in range(0, int((NUM_ANT_FACTOR * NUM_CITY))):
        random.shuffle(cityList)
        antName = 'Ant_' + str(antCount)
        antColony.append(Ant(name=antName, startPoint=cityList[0]))

    return antColony


def create_pheromone_matrix(cityList):
    '''pheromone matrix'''
    pheratrix = [[1]*len(cityList) for i in range(len(cityList))]
    return pheratrix


def create_distance_matrix(cityList):
    '''distance matrix'''

    disMatrix = [[0]*len(cityList) for i in range(len(cityList))]
    for x in range(len(cityList)):
        for y in range(len(cityList)):
            disMatrix[x][y] = cityList[x].get_distance(cityList[y])

    # print('\n')
    # print(disMatrix[0][9])
    # print('\n')
    # print(disMatrix[9])
    
    return disMatrix


def create_cities(NUM_ANT_FACTOR, NUM_CITY):
    '''creates the cities and population'''

    # 58 cities
    cityName = ['Apalachicola', 'Bartow','Belle Glade', 'Boca Raton', 
                'Bradenton', 'Cape Coral', 'Clearwater', 'Cocoa Beach', 
                'Cocoa-Rockledge', 'Coral Gables', 'Daytona Beach', 'De Land',
                'Deerfield Beach', 'Delray Beach', 'Fernandina Beach',
                'Fort Lauderdale', 'Fort Myers', 'Fort Pierce', 
                'Fort Walton Beach', 'Gainesville', 'Hallandale Beach', 
                'Hialeah', 'Hollywood', 'Homestead', 'Jacksonville', 
                'Key West', 'Lake City', 'Lake Wales', 'Lakeland', 'Largo', 
                'Melbourne', 'Miami', 'Miami Beach', 'Naples', 
                'New Smyrna Beach', 'Ocala', 'Orlando', 'Ormond Beach', 
                'Palatka', 'Palm Bay', 'Palm Beach', 'Panama City', 
                'Pensacola', 'Pompano Beach', 'Saint Augustine', 
                'Saint Petersburg', 'Sanford', 'Sarasota', 'Sebring', 
                'Tallahassee', 'Tampa', 'Tarpon Springs', 'Titusville', 
                'Venice', 'West Palm Beach', 'White Springs', 'Winter Haven', 
                'Winter Park']
    
    random.shuffle(cityName)
    
    cityList = []
    for city in range(0, int((NUM_ANT_FACTOR * NUM_CITY))):
        cityList.append(City(name=cityName[city], 
                            x=int(random.random() * 200), 
                             y=int(random.random() * 200)))
        
    return cityList
        



def main():
    '''main function'''



    # Hyperparameters
    NUM_ANT_FACTOR = 1
    NUM_CITY = 10


    random.seed(1)

    cityList = create_cities(NUM_ANT_FACTOR, NUM_CITY)
    # print('\n')
    # print('This is the city list: ')
    # print(cityList)
    # print('\n')


    pheratrix = create_pheromone_matrix(cityList)
    print('\n')
    for row in pheratrix:
        print(row)
        
    

    # create_distance_matrix(cityList)
    disMatrix = create_distance_matrix(cityList)
    print('\n')
    for row in disMatrix:
        print(row)

    # print(disMatrix[3][4])

    # print(len(disMatrix))

    # antColony = create_colony(cityList, NUM_ANT_FACTOR, NUM_CITY)
    # print('\n')
    # print('This is the colony: ')
    # print(antColony)
    # print('\n')


    


    # poop = [1,1,4,23,43,4,2,123,4,1]

    
    # adjMatrix = [[0] * len(poop)] * len(poop)

    # for x in range(len(adjMatrix)):
    #     print(adjMatrix[x])





if __name__ == '__main__':
    main()