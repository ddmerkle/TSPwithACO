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





# WHAT WERE GOING TO DO

# def create_probabilities(colony(unvisited cities), cityList, pheratrix) 
    #-> ant, probabilities

    # for city in cityList:
    #     if city not in ant.visited:


# def roulette_wheel(colony, probabilities)
    # returns the city that the ant should move to 
    # unvisitedCities(index of city chosen)


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
    
    def get_name(self):
        '''returns string representation as function.
            Usable as object call'''

        return '({name})'.format(name = self.name)
    
    def __repr__(self):
        '''returns string representation of city'''

        return '({name}: x:{x}, y:{y})'.format(name = self.name, 
                                         x = self.x, y = self.y)
    

class Ant():
    '''ant class'''

    def __init__(self, name, startPoint):
        '''constructor'''
    
        self.name = name
        self.tourList = []
        
        #defines starting point in list
        self.tourList.append(startPoint)
    
    def visit(self, cityObject):
        '''Ant has visited this city'''
        self.tourList.append(cityObject)

    def get_tour(self):

        return self.tourList
    
    # def get_distance_traveled(self):
    
    def __repr__(self):
        '''returns string representation of ant'''
    
        return '({name}: list: {tourList})'.format(name = self.name, 
                                                   tourList = self.tourList)


#################################################

# for ant in colony:
#     for city in cityList:
#         finishtour(city)

# def visit_city(self, pheromoneTrails):
# def visit_random_city(self):
# def visit_probabilistic_city(self, pheromonTrails):
# def roulette_wheel_selection(self, probabilities):





def do_tour(ant, cityList, strCityList, disMatrix, pheratrix, ALPHA, BETA):
    '''does tour for all ants'''
    probList = create_probabilities(ant, cityList, strCityList, disMatrix, 
                                    pheratrix, ALPHA, BETA)

def create_probabilities(ant, cityList, strCityList, disMatrix, pheratrix, 
                         ALPHA, BETA):
    '''create probabilities for cities to be used in roulette for an ant'''

    print(str(ant))
    print('\n')

    denominator = 0
    currentCity = cityList.index(ant.get_tour()[-1])
    unVisitedList = []
    probList = []
    # print(currentCity)

    for city in cityList:
        if city not in ant.get_tour():
            i = strCityList.index(city.__repr__())
            unVisitedList.append(i)
            #print(unVisitedList)

    for city in unVisitedList:
        # print(city)
        if currentCity == city:
            continue

        denominator += ((pheratrix[currentCity][city]**ALPHA) * 
                        ((1/disMatrix[currentCity][city])**BETA))

    for city in unVisitedList:
        if currentCity == city:
            continue

        numarator = ((pheratrix[currentCity][city]**ALPHA) * 
                        ((1/disMatrix[currentCity][city])**BETA))

        cityProb = numarator / denominator

        probList.append(cityProb)

    print('These are the probabilities')
    print(probList)
    print('\n')
    return probList


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

    # disMatDict = dict(zip(strCityList, disMatrix))
    # print('!!!!!!!!!!!!!!')
    
    # i = list(disMatDict).index('(Palm Bay: x:176, y:155)')
    # j = list(disMatDict).index('(Key West: x:179, y:194)')
    # print(i)
    # print(j)
    # print(disMatrix[i][j])
    # print('!!!!!!!!!!!!!!')

    '''
    NEED
    
    pharatrix
    dismatrix
    
    ant starts at city[0] and goes to city[1] (for iteration 0 next city
    is determined by roulette wheel with lowest distance being the only factor),
    we need to find from the distance matrix where city[0] and city[1] is. 
    Could be Boca to Key West which could return disMatrix[0][2]. 
    We need to save this value in order to update pheratrix[10][5]. 
    ONE ant will return pairs of [x][y] to update all routes 
    it took during it's tour
    
    The ants city[0] and city[1] return a string (Boca)(Key West) which 
    is then searched in the disMatrix. This is done with a dictionary 
    (disMatDict) where {Boca : [0, 2, 4, 5...]} and 
    {Key West : [4, 6, 0, 4...]}. The value of 0 indicates which city is 
    at index 0 of disMatrix so disMatrix[0] is Boca and disMatrix[2] 
    is key west. disMatrix[0][2] will give us the distance between those two 
    cities'''
    
    # return disMatrix, disMatDict

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
        
    strCityList = []
    for city in cityList:
        strCityList.append(str(city.__repr__()))
        
    return cityList, strCityList
        



def main():
    '''main function'''

    # Hyperparameters
    NUM_ANT_FACTOR = 1
    NUM_CITY = 10
    ALPHA = 1
    BETA  = 2
    RANDOM_ATTRACTION_FACTOR = 0.5
    NUM_ITERATIONS = 10
    EVAPORATION_RATE = 0.5


    random.seed(1)

    cityList, strCityList = create_cities(NUM_ANT_FACTOR, NUM_CITY)
    print('\n')
    print('This is the city list in object format: ')
    print(cityList)
    print('!!!!!!!!!!!!!!')
    print('This is the city list in str format: ')
    print(strCityList)
    print('\n')

    disMatrix = create_distance_matrix(cityList)
    print('\n')
    for row in disMatrix:
        print(row)

    i = strCityList.index('(Palm Bay: x:176, y:155)')
    j = strCityList.index('(Key West: x:179, y:194)')

    ant = disMatrix[i][j]
    print('!!!!!!!!!!!!!!')
    print(ant)
    print('!!!!!!!!!!!!!!')

    pheratrix = create_pheromone_matrix(cityList)
    print('\n')
    for row in pheratrix:
        print(row)

    antColony = create_colony(cityList, NUM_ANT_FACTOR, NUM_CITY)
    print('\n')
    print('This is the colony: ')
    print(antColony)
    print('\n')

    for ant in antColony:
        do_tour(ant, cityList, strCityList, disMatrix, pheratrix, ALPHA, BETA)
        break

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('done')




    # disMatrix, disMatDict = create_distance_matrix(cityList)
    # print('\n')
    # for row in disMatrix:
    #     print('\n')
    #     print(row)

    # print('\n')
    # print(disMatDict)
    # print('\n')

    # print(disMatrix[3][4])

    # print(len(disMatrix))

    

if __name__ == '__main__':
    main()