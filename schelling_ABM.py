# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:41:04 2016
@author: ahowe42

Implement the Schelling Segregation Model with agents
I've created Person as a class because that's the most logical and flexible
structure.  I've created Race as a class because I can then set various
variables to be race-specific. This structure also allows me to extend the
model to consider other demographics, such as age band, etc...
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class Race:
    def __init__(self,name,plot_color,simil_pref):
        self.name = name
        self.plot_color = plot_color
        self.simil_pref = simil_pref
    def __repr__(self):
        return self.name

class Person:
    def __init__(self,race):
        self.race = race
        self.happy = None
    def is_happy(self,loc,houses):
        race_diff_simil = [0,0] # 1st is diff, 2nd is sim
        neighbors = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
        # check each neighbor
        for coord in neighbors:
            neighbor = houses[loc[0]+coord[0], loc[1]+coord[1]]
            if neighbor is not None:
                race_diff_simil[self.race == neighbor.race]+=1
        # check if similarity count is pleasing
        if sum(race_diff_simil) == 0:
            self.happy = False
        else:
            self.happy = race_diff_simil[1]/sum(race_diff_simil)>=self.race.simil_pref
        return self.happy
    def __repr__(self):
        return '%s(%s)'%(self.race.name,self.happy)

class Map:
    def __init__(self,width,height,p_empty,p_races,races):
        '''
        note that it is assumed all proportions sum to 1!!!
        '''
        # save stuff
        self.width = width
        self.height = height
        self.races = races
        self.people_cnt = width*height*(1-p_empty)
        # prepare the houses
        houses = []
        houses_count = width*height
        # assign each race to houses - first we do in a vector
        # later, we'll randomize, reshape and assign coords
        for i,r in enumerate(races):
            # create each agent and house him
            rcnt = int(p_races[i]*houses_count)
            rh = [None]*rcnt
            for h in range(rcnt):
                rh[h] = Person(r)
            houses.extend(rh)
        # now houses have been defined for all races, so add empties
        houses.extend([None]*int(p_empty*houses_count))
        # randomly shuffle the houses, then reshape to a grid
        houses = np.array(houses)
        np.random.shuffle(houses)
        # add the border of empties
        self.houses = np.array([[None]*(width+2)]*(height+2),dtype=Person)
        self.houses[1:-1,1:-1] = np.reshape(houses,(height,width))
        # finally, record which houses are empty
        self.empties = []
        for r in range(height):
            for c in range(width):
                if self.houses[r+1,c+1] is None:
                    self.empties.append([r+1,c+1])
        # note that all uses of looping through houses should always simply
        # loop with range(width or height) and access indices with +1
        # so as to guarantee we ignore the no-man's-land boundary
    def check_happy(self):
        '''
        check if each person is happy, returns the number and percentage
        of happy people
        '''
        happies = 0
        for r in range(self.height):
            for c in range(self.width):
                thisperson = self.houses[r+1,c+1]
                if thisperson is not None:
                    happies += thisperson.is_happy([r+1,c+1],self.houses)
        return happies,happies/self.people_cnt
    def moving_time(self):
        '''
        loop through all people and let whomever wants move
        returns number of people who moved
        '''
        moved = 0
        for r in range(self.height):
            for c in range(self.width):
                thisperson = self.houses[r+1,c+1]
                if thisperson is not None:
                    if not(thisperson.happy):
                        # person is not happy so let him move
                        self.move_empty(thisperson,[r+1,c+1])
                        moved += 1
        return moved,moved/self.people_cnt
    def move_empty(self,thisperson,cur_home):
        '''
        pick a random empty house and move the person there
        '''
        # get a random empty house and mark as not empty
        new_home = np.random.permutation(self.empties)[0].tolist()
        self.empties.remove(new_home)
        # get the current home and mark as empty
        self.empties.append(cur_home)
        # now update the houses matrix
        self.houses[cur_home[0],cur_home[1]] = None
        self.houses[new_home[0],new_home[1]] = thisperson
    def Plot(self,atitle,ax):
        '''
        plot the grid showing all the people
        statuses
        '''
        ax.set_axis_bgcolor((210/256,210/256,210/256))
        plt.hold('on')
        for r in range(self.width+2):
            for c in range(self.height+2):
                if self.houses[r,c] is None:
                    plt.scatter(r,c,marker = '.',c=(210/256,210/256,210/256),s=50)
                else:
                    plt.scatter(r,c,marker = 'xo'[self.houses[r,c].happy],\
                        c = self.houses[r,c].race.plot_color,s=50)
        plt.grid()
        plt.axis([0,self.width+1,0,self.height+1])
        plt.title(atitle)
        plt.hold('off')
    
# now run everything
# parameters - note that the proportion parameters *must* be consistent
# with the dim's, i.e, all must result in an integer number people in each
# race and integer number of empty houses
races = [Race('White','w',0.5),\
        Race('Black','k',0.7),\
        Race('Chinese','y',0.6),\
        Race('Indian','r',0.5)]
width = 50
height = 50
people_cnt = width*height
propors = [0.25,0.25,0.15,0.15]
empty = 0.20
MCmaxsims = 1000
happies = []
moves = []
convgcrit_h = 0.00001 # model converged when happiness changes less than this
convgcrit_m = 0.0001 # model converged when fewer than this % moves in last 5 iters

# create the model
thismap = Map(width,height,empty,propors,races)
plt.close('all')

# talk and plot initial situation
h,perch = thismap.check_happy()
tit = 'Initially, %d (%0.2f%%) happy people'%(h,100*perch)
print(tit)
fh = plt.figure()
ax = plt.subplot(1,2,1)
thismap.Plot(tit,ax)
# simulate
for sim in range(MCmaxsims):
    # let people move, then figure out if they're happy
    m,percm = thismap.moving_time()
    h,perch = thismap.check_happy()
    moves.append(percm)
    happies.append(perch)
    # talk, and maybe plot
    tit = 'After iteration %d (%d movers), %d (%0.2f%%) happy people'%\
        (sim,m,h,100*perch)
    print(tit)
#    if sim % 10 == 0:
#        thismap.Plot(tit)
    # if at least 10 iterations, check for termination
    # alternatively, whenever 100% of people are happy, terminate
    if happies[-1] == 1:
        print("Everyone's happy, so terminating early...")
        break
    if sim >= 10:
        if (max([abs(happies[-1] - h)for h in happies[-10:]]) <= convgcrit_h) &\
            (sum(moves[-10:]) <= convgcrit_m):
            # no more improvement in happiness & moves, so can terminate
            print('Terminating early...')
            break
        
# plot the final configuration
plt.figure(fh.number)
ax = plt.subplot(1,2,2)
thismap.Plot(tit,ax)
# plot the progress of happiness & moves
plt.figure()
plt.subplot(1,2,1).plot(range(sim+1),happies)
plt.title('% Happy People')
plt.subplot(1,2,2).plot(range(sim+1),moves)
plt.title('% People Moved')
