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
from itertools import count
plt.ion()

class Race:
  def __init__(self,name,plot_color,simil_pref):
    self.name = name
    self.plot_color = plot_color
    self.simil_pref = simil_pref
  def __repr__(self):
    return self.name

class Person:
  _ids = count(0)
  def __init__(self,race):
    self.race = race
    self.happy = None
    self.id = self._ids.__next__()
  def is_happy(self,neighbors):
    race_diff_simil = [0,0] # 1st is diff, 2nd is sim    
    # check each neighbor
    for neighbor in neighbors:
      if neighbor is not None:
        race_diff_simil[self.race == neighbor.race]+=1
    # check if similarity count is pleasing
    if sum(race_diff_simil) == 0:
      self.happy = False
    else:
      self.happy = race_diff_simil[1]/sum(race_diff_simil)>=self.race.simil_pref
    return self.happy
  def __repr__(self):
    if self.happy is None:
      mestr = '%d-%s(-)'%(self.id,self.race.name)
    else:
      mestr = '%d-%s(%s)'%(self.id,self.race.name,'UH'[self.happy])
    return mestr
    
class Map:
  def __init__(self,width,height,p_empty,p_races,races):
    '''
    note that it is assumed all proportions sum to 1!!!
    '''
    # save stuff
    self.width = width
    self.height = height
    houses_count = width*height
    self.races = races
    self.people_cnt = houses_count*(1-p_empty)
    # prepare the houses - starting with empties
    self.houses = [None]*int(p_empty*houses_count)    
    # assign each race to houses - first we do in a vector
    # later, we'll randomize, reshape and assign coords
    for i,r in enumerate(races):
      # create each agent and house him
      rcnt = int(p_races[i]*houses_count)
      rh = [None]*rcnt
      for h in range(rcnt):
        rh[h] = Person(r)
      self.houses.extend(rh)
    # randomly shuffle the houses, then reshape to a grid
    np.random.shuffle(self.houses)
    self.houses = np.reshape(self.houses,(height,width))
    # finally, record which houses are empty
    self.empties = []
    for r in range(height):
      for c in range(width):
        if self.houses[r,c] is None:
            self.empties.append([r,c])
    # finally, finally, spaceholders for plotting stuff
    self.plot_artists = None
    self.plax = None
  def WhosHappy(self):
    '''
    check if each person is happy, returns the number and percentage
    of happy people
    '''
    happies = 0
    neighbor_rows = [-1,-1,-1,0,1,1, 1, 0]
    neighbor_cols = [-1, 0, 1,1,1,0,-1,-1]
    for r in range(self.height):
      for c in range(self.width):
        thisperson = self.houses[r,c]
        if thisperson is not None:
          oldhappy = thisperson.happy
          # need to build this person's matrix of neighbors
          myneighR = [n+r for n in neighbor_rows]
          myneighC = [n+c for n in neighbor_cols]
          # compute any adjustments for edges - a person's neighbors
          # wrap around to the map
          myneighR = [[r,0][r == self.height] for r in \
            [[r,self.height-1][r<0] for r in myneighR]]
          myneighC = [[c,0][c == self.width] for c in \
            [[c,self.width-1][c<0] for c in myneighC]]
          neighbors = [self.houses[r,c] for r,c, in zip(myneighR,myneighC)]
          # measure happiness
          happies += thisperson.is_happy(neighbors)
          # if happiness status changed, let's update the plot
          if (oldhappy != thisperson.happy) & (oldhappy is not None) &\
            (self.plax is not None):
            self.plot_artists[r,c].remove()
            self.plot_artists[r,c] = self.plax.scatter(r,c,marker = 'xo'[thisperson.happy],\
              c = thisperson.race.plot_color,s=50)
    return happies,happies/self.people_cnt
  def moving_time(self):
    '''
    loop through all people and let whomever wants move
    returns number of people who moved
    '''
    moved = 0
    # need to make a copy of the houses matrix to see the status of the 
    # array before any moves, so we don't allow multiple moves
    mapHseCpy = self.houses.copy()
    for r in range(self.height):
      for c in range(self.width):
        thisperson = self.houses[r,c]
        # the thisperson =? None check is from the copy
        if mapHseCpy[r,c] is not None:
          if not(thisperson.happy):
            # person is not happy so let him move
            self.move_empty(thisperson,[r,c])
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
    # update the plot if a plot has been made
    if (self.plax is not None):
      self.PlotMove(cur_home,new_home)
  def Plot(self,atitle,ax):
    '''
    plot the grid showing all the people & statuses
    '''
    # first setup to store info about the plotted points
    self.plot_artists = []
    self.plax = ax
    self.plax.set_axis_bgcolor((210/256,210/256,210/256))
    plt.hold('on')
    for r in range(self.width):
      for c in range(self.height):
        if self.houses[r,c] is None:
          self.plot_artists.append(self.plax.scatter(r,c,marker = '.',\
            c=(210/256,210/256,210/256),s=50))
        else:
          self.plot_artists.append(self.plax.scatter(r,c,marker = 'xo'[self.houses[r,c].happy],\
            c = self.houses[r,c].race.plot_color,s=50))
    self.plax.grid('off')
    self.plax.axis([-1,self.width,-1,self.height])
    self.plax.set_title(atitle)
    # reshape plot artist list to array
    self.plot_artists = np.reshape(self.plot_artists,(self.width,self.height))
  def PlotMove(self,orig,dest):
    '''
    move a person on the plot; this means setting the origin position
    to a None point and the destination position to the person's point
    '''
    # first remove both positions
    self.plot_artists[orig[0],orig[1]].remove()
    self.plot_artists[dest[0],dest[1]].remove()
    # now reset the origin to None
    r = orig[0]; c = orig[1]
    self.plot_artists[r,c] = self.plax.scatter(r,c,marker='.',c=(210/256,210/256,210/256),s=50)
    # set the destination to the person
    # just set the marker to unhappy, since that's the only reason he'd move
    # this will be updated in the next WhosHappy
    r = dest[0]; c = dest[1]
    self.plot_artists[r,c] = self.plax.scatter(r,c,marker='x',c=self.houses[r,c].race.plot_color,s=50)
    #plt.draw(); plt.pause(0.01)
  def PlotRedraw(self,atitle,paussec):
    if atitle is not None:
      self.plax.set_title(atitle)
    plt.draw(); plt.pause(paussec)
    
# now run everything
# parameters - note that the proportion parameters *must* be consistent
# with the dim's, i.e, all must result in an integer number people in each
# race and integer number of empty houses
races = [Race('White','w',0.5), Race('Black','k',0.7),\
  Race('Chinese','y',0.6), Race('Indian','r',0.5)]
width = 50
height = 50
people_cnt = width*height
propors = [0.25,0.30,0.15,0.20]
empty = 0.10

# create the model
thismap = Map(width,height,empty,propors,races)
plt.close('all')

# talk and plot initial situation
h,perch = thismap.WhosHappy()
tit = 'Init: %d Happy (%0.2f%%)'%(h,100*perch)
print(tit)
jnk = plt.subplots(2,2)[1]
(ax_init,ax_curr,axh,axm) = jnk.flatten().tolist()

thismap.Plot(tit,ax_init)
thismap.Plot(tit,ax_curr)
axh.set_title('Happy People')
axm.set_title('People Moved')

# simulate
MCmaxsims = 500
happies = []
moves = []
convgcrit_h = 0.00001 # model converged when happiness changes less than this
convgcrit_m = 0.0001 # model converged when fewer than this % moves in last 5 iters
for sim in range(MCmaxsims):
  # let people move, then figure out if they're happy
  m,percm = thismap.moving_time()
  h,perch = thismap.WhosHappy()
  moves.append(percm)
  happies.append(perch)
  # talk, and maybe plot
  tit = 'Iter %d: %d Happy (%0.2f%%)'%(sim,h,100*perch)
  print(tit)
  if sim % 1 == 0:
    axh.plot(range(sim+1),happies,'g-')
    axh.set_title('Happy People (%d)'%h)
    axm.plot(range(sim+1),moves,'r-')
    axm.set_title('People Moved (%d)'%m)
    thismap.PlotRedraw(tit,0.005)

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
# final redraw, in case the last iteration was not redrawn
thismap.PlotRedraw(tit,0.005)