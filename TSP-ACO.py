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
import math
import copy


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

    def __init__(self, name):
        '''constructor'''
    
        self.name = name
        self.tourList = []
        self.distance = 0
    
    def set_start(self, startPoint):
        '''#defines starting point in list'''
        self.tourList.append(startPoint)
    
    def visit(self, cityObject):
        '''Ant has visited this city'''
        self.tourList.append(cityObject)

    def get_tour(self, reset=False):

        if reset:
            self.tourList = []

        return self.tourList

    def set_distance(self, antDistance):
        '''sets distance of ant'''

        self.distance = antDistance

    def get_distance(self):
        '''gets distance of ant'''

        return self.distance
    
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





def do_tour(ant, cityList, strCityList, disMatrix, pheratrix, ALPHA, BETA, BEST_GLOBAL_DISTANCE, BEST_GLOBAL_ROUTE, tempPheramones):
    '''does tour for all ants'''
    

    # HERES WHERE WE WILL HAVE A LOOP FOR THE ANT TO COMPLETE ITS TOUR

    for _ in range(len(cityList) - 1):

        

        probList, unVisitedList = create_probabilities(ant, cityList, strCityList, 
                                                    disMatrix, pheratrix, 
                                                    ALPHA, BETA)
    
        indexofNextCity = proportional_roulette_selection(probList)

        tempindex = unVisitedList[indexofNextCity]
        ant.visit(cityList[tempindex])


        
    # finding the total distance for the ants route
    totalDis = 0
    for index in range(len(cityList)):
        if index == (len(cityList)-1):
            totalDis += ant.get_tour()[index].get_distance(ant.get_tour()[0])
        else:
            totalDis += ant.get_tour()[index].get_distance(
                ant.get_tour()[index+1])
        # print(totalDis)



    # UPDATE PHERAMONE MATRIX HERE
    pheramone = 1 / totalDis
    for index in range(len(cityList) - 1):
        # get city index and second city index and update matrix

        cityindex1 = cityList.index(ant.get_tour()[index])
        cityindex2 = cityList.index(ant.get_tour()[index + 1])

        tempPheramones[cityindex1][cityindex2] += pheramone



    ant.set_distance(totalDis)
        
    #adding start point to end
    ant.visit(ant.get_tour()[0])
    

    # seeing if the route is new best route
    if ant.get_distance() < BEST_GLOBAL_DISTANCE:
        BEST_GLOBAL_DISTANCE = ant.get_distance()
        BEST_GLOBAL_ROUTE = copy.deepcopy(ant.get_tour())



    # print('\n')
    # print('This is the distance of the ants route:')
    # print(ant.get_distance())
    # print('\n')

    # print('\n')
    # print('ANT AFTER PROBS/SELECTION: ')
    # print(str(ant))
    # print('\n')
    
    


    # UPDATE TEMP PHERAMONE MATRIX



    return BEST_GLOBAL_DISTANCE, BEST_GLOBAL_ROUTE, ant.get_distance(), tempPheramones
    



def create_probabilities(ant, cityList, strCityList, disMatrix, pheratrix, 
                         ALPHA, BETA):
    '''create probabilities for cities to be used in roulette for an ant'''

    # print('\n')
    # print('ANT: ')
    # print(str(ant))
    # print('\n')

    # print('\n')
    # print('This is the city list: ')
    # print(cityList)
    # print('\n')


    denominator = 0

    currentCity = cityList.index(ant.get_tour()[-1])


    # print('\n')
    # print('This is the city list: ')
    # print(cityList)
    # print('\n')

    # print('\n')
    # print('This is the current city:')
    # print(currentCity)
    # print('\n')

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

        numerator = ((pheratrix[currentCity][city]**ALPHA) * 
                        ((1/disMatrix[currentCity][city])**BETA))

        cityProb = numerator / denominator

        probList.append(cityProb)

    # print('These are the probabilities')
    # print(probList)
    # print('\n')

    # print('\n')
    # print('This is the unvisited list: ')
    # print(unVisitedList)
    # print('\n')
    return probList, unVisitedList


def proportional_roulette_selection(probList):
    '''Selecting parents using proportional roulette selection'''

    slices = []
    for slice in range(len(probList)):
        sliceCounter = 0
        if slice == 0:
            slices.append(probList[0])
        else:
            slices.append(slices[sliceCounter-1] + probList[slice])

    # print('\n')
    # print('These are the slices: ')
    # print(slices)
    # print('\n')

    randomNumber = random.uniform(0,1)

    for index, slice in enumerate(slices):
        if randomNumber < slice:
            # print('\n')
            # print('This is the slice chosen:')
            # print(slices[index])
            # print('\n')

            # print('\n')
            # print('This is the index of the chosen city: ')
            # print(index)
            # print('\n')
            return index
            # return slices[index]
        


def create_colony(cityList, NUM_ANT_FACTOR, NUM_CITY):

    # print('\n')
    # print('CITY LIST CHECK:')
    # print(cityList)
    # print('\n')


    antColony = []
    for antCount in range(0, int((NUM_ANT_FACTOR * NUM_CITY))):
        randomIndex = random.randint(0, len(cityList) - 1)
        antName = 'Ant_' + str(antCount)
        antColony.append(Ant(name=antName))
        antColony[antCount].set_start(cityList[randomIndex])

    return antColony


# ORIGINAL FUNCTION 

# def create_colony(cityList, NUM_ANT_FACTOR, NUM_CITY):

#     print('\n')
#     print('CITY LIST CHECK:')
#     print(cityList)
#     print('\n')

#     tempCityList = cityList

#     antColony = []
#     for antCount in range(0, int((NUM_ANT_FACTOR * NUM_CITY))):
#         random.shuffle(tempCityList)
#         antName = 'Ant_' + str(antCount)
#         antColony.append(Ant(name=antName, startPoint=tempCityList[0]))

#     cityList = tempCityList

#     return antColony



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
    NUM_ITERATIONS = 1000
    EVAPORATION_RATE = 0.5

    BASELINE_DISTANCE = 0
    
    BEST_ITERATION_DISTANCE = []
    CURRENT_TOUR = 0
    

    BEST_GLOBAL_DISTANCE = math.inf
    BEST_GLOBAL_ROUTE = []


    # random.seed(1)

    cityList, strCityList = create_cities(NUM_ANT_FACTOR, NUM_CITY)
    # print('\n')
    # print('This is the city list in object format: ')
    # print(cityList)
    # print('!!!!!!!!!!!!!!')
    # print('This is the city list in str format: ')
    # print(strCityList)
    # print('\n')


    # Looping through cityList to get baseline distance
    for index, city in enumerate(cityList):

        if index == len(cityList)-1:
            BASELINE_DISTANCE += city.get_distance(cityList[0])
        else:
            BASELINE_DISTANCE += city.get_distance(cityList[index+1])

    
        
    



    disMatrix = create_distance_matrix(cityList)
    # print('\n')
    # for row in disMatrix:
    #     print(row)


    # i = strCityList.index('(Palm Bay: x:176, y:155)')
    # j = strCityList.index('(Key West: x:179, y:194)')

    # ant = disMatrix[i][j]
    # print('!!!!!!!!!!!!!!')
    # print(ant)
    # print('!!!!!!!!!!!!!!')

    pheratrix = create_pheromone_matrix(cityList)
    # print('\n')
    # for row in pheratrix:
    #     print(row)


    antColony = create_colony(cityList, NUM_ANT_FACTOR, NUM_CITY)
    # print('\n')
    # print('This is the colony: ')
    # print(antColony)
    # print('\n')

    # print('\n')
    # print('This is the city list CHECK:')
    # print(cityList)
    # print('\n')


    # creating a temporary pheratrix
    tempPheramones = pheratrix

    for _ in range(NUM_ITERATIONS):

        TEMP_TOUR_LIST = []

        for ant in antColony:



            BEST_GLOBAL_DISTANCE, BEST_GLOBAL_ROUTE, CURRENT_TOUR, tempPheramones = do_tour(ant, cityList, strCityList, disMatrix, 
                                        pheratrix, ALPHA, BETA, 
                                        BEST_GLOBAL_DISTANCE, BEST_GLOBAL_ROUTE, tempPheramones)
            




            TEMP_TOUR_LIST.append(CURRENT_TOUR)


            # break

        pheratrix = tempPheramones

        TEMP_TOUR_LIST.sort()
        BEST_ITERATION_DISTANCE.append(TEMP_TOUR_LIST[0])




        # get rid of stuff
        for ant in antColony:
            ant.get_tour(True)
            ant.set_distance(0)
            randomindex = random.randint(0, len(cityList) - 1)
            ant.set_start(cityList[randomindex])




    print('!!!!!!!!!!!!!!')
    print('\n')
    print('This is baseline length: ')
    print(BASELINE_DISTANCE)
    print('\n')

    print('\n')
    print('This is iteration best distance: ')
    print(BEST_ITERATION_DISTANCE)
    print('\n')

    print('\n')
    print('This is global best distance: ')
    print(BEST_GLOBAL_DISTANCE)
    print('\n')

    print('\n')
    print('This is global best route: ')
    print(BEST_GLOBAL_ROUTE)
    print('\n')


    print('\n')
    for row in pheratrix:
        print(row)
    print('\n')

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