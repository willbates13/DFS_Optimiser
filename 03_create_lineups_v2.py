#!/usr/bin/env python3

import pulp
import pandas as pd
import numpy as np
import itertools
import math
from scipy.stats import norm
import pulp
solver = pulp.PULP_CBC_CMD(msg=0)

############################ Global Params #################################
week = 'week17'
slate = '/ms'
comp_type = 'classic'

number_of_lineups = 5
TS = 250 # target score for lineups
beta = 10 # scales mean
lambda_ = 0.1 # scales variance
delta = 0.1 # scales covariance
#########################################################################

cov_mat = pd.read_csv(week + slate + '/covariance_matrix.csv').set_index('index')
df = pd.read_csv(week + slate + '/projections.csv')

def main():

    cov_mat.index = pd.to_numeric(cov_mat.index, errors='coerce')
    cov_mat.columns = pd.to_numeric(cov_mat.columns, errors='coerce')
    
    lineups = create_lineups(df, cov_mat, number_of_lineups)

    teams_ID = pd.DataFrame(lineups,columns = ['QB','RB','RB','WR','WR','WR','TE','FLEX','DST'])
    print(teams_ID)
    
    teams_ID.to_csv(week + slate + '/lineups.csv', index=False) 

def create_lineups(df, cov_mat, number):
    
    groups = df.groupby(['Game Info']).agg(list)
    teams = []
    counter = 0
    for it in range(number):
    
        # Define the problem
        prob = pulp.LpProblem("DraftKings_Lineup_Optimization", pulp.LpMaximize)
            
        allId = [x for x in df["ID"]]
        pairId = [(i,j) for z in range(len(groups)) for (i,j) in itertools.combinations(groups['ID'][z],2)]
    
        # Define the decision variables
        x = pulp.LpVariable.dicts("x", allId, 0, 1, pulp.LpBinary)
        x_pair = pulp.LpVariable.dicts("x_pair", pairId, 0, 1, pulp.LpBinary)
    
        #link the pair selections to the singles
        for i, j in pairId:
            prob += x_pair[i, j] <= (x[i] + x[j]) / 2  
            prob += x_pair[i, j] >= x[i] + x[j] - 1
        
        mean = pulp.lpSum([df["projPoints"][count] * x[idx] for count,idx in enumerate(allId)])
        variance = pulp.lpSum([df["variance"][count]*x[idx] for count,idx in enumerate(allId)])
        covariance = 2*pulp.lpSum([x_pair[i,j]*cov_mat.loc[i][j] for i,j in pairId])
    
        prob += (beta * mean) + (lambda_ * variance) + (delta * covariance)
        # Salary contraint
        prob += pulp.lpSum([df["Salary"][count]*x[idx] for count,idx in enumerate(allId)]) <= 50000
        
        # max lineup size of 9
        prob += pulp.lpSum(x) == 9
    
        # overlap between teams consraint
        if counter != 0:
            for j in range(len(teams)):
                prob += pulp.lpSum(x[idx] for idx in teams[j]) <= 4
                
              
        def get_sum(player_vars, df, value, field):
            return pulp.lpSum([player_vars[idx] * (value in df[field][count]) for count,idx in enumerate(allId)])
        
        # positional restrictions for classic format
        prob += get_sum(x, df, 'QB', 'Position') == 1
        prob += get_sum(x, df, 'DST', 'Position') == 1
        prob += get_sum(x, df, 'RB', 'Position') >= 2
        prob += get_sum(x, df, 'WR', 'Position') >= 3
        prob += get_sum(x, df, 'TE', 'Position') == 1
        prob += get_sum(x, df, 'TE', 'Position') == 1
        
        # Solve the problem
        prob.solve(solver)
    
        # return the optimal lineup
        sol = []
        for i in allId:
            if x[i].value() == 1.0:
                sol.append(i)
                
        lineup = pd.DataFrame(sol,list(df[(df.ID.isin(sol))]['Position'])).rename_axis('Position').reset_index()
        teams.append(sort_lineup(lineup))
        
        # Calculate percentage chance of team reaching a given score

        perc,mu,sigma,covar = win_perc(sol, TS)
        
        teamPos = pd.DataFrame({'Name':list(df.Name[df.ID.isin(sol)]),'Pos':list(df[df.ID.isin(sol)]['Roster Position']), 'Team':list(df[df.ID.isin(sol)]['TeamAbbrev'])}).sort_values('Pos')
        
        # print info
        print(teamPos)
        print("mean = " + str(round(mu, 2)) + " Covariance = " + str(round(covar, 2)) + " Standard Deviation = " + str(round(sigma, 2)))
        print("percentage chance = "+str(round(perc*100, 8)))
        counter = counter + 1
        
    return teams


def sort_lineup(lineup):
    # sort lineup in to correct position format.
    sorted_lineup = []
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "QB"))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "RB"))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "RB" and x[0] not in sorted_lineup))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "WR"))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "WR" and x[0] not in sorted_lineup))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "WR" and x[0] not in sorted_lineup))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "TE"))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position != "DST" and x[0] not in sorted_lineup))
    sorted_lineup.append(next(x[0] for index,x in lineup.iterrows() if x.Position == "DST"))
    return sorted_lineup
   
    
def win_perc(lineup,target_score):
    # Calclulate mean variance and percentage chance of reaching target score
    # lineup = sol.copy()
    tc = 0
    for (i,j) in itertools.combinations(lineup,2):
        tc = tc + cov_mat.loc[i][j]
    var = sum(df[df.ID.isin(lineup)].variance) + (2*tc)
    mu = sum(df[df.ID.isin(lineup)].projPoints)
    sigma = math.sqrt(abs(var))
    perc = 1-(norm(loc=mu, scale=sigma).cdf(target_score))

    return perc,mu,sigma,tc


main()