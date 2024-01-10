#!/usr/bin/env python3

import pandas as pd
import numpy as np

############################ Data input #################################
# This script assumes that you have saved the raw html from this link https://www.numberfire.com/nfl/daily-fantasy/daily-football-projections
# at this relative file path and names: 'projections/projections_players.html' and 'projections/projections_D.html'

# location of salary file exported from DK website
dk = pd.read_csv('DKSalaries_week17_ms.csv')

######################################################################


def main():
    
    # get numberfire data
    data = scrape()
    
    dk['Name'] = dk.apply(normalise_names, axis=1)

    # Correct defense names
    data.loc[data['Name'] == 'JAC', 'Name'] = 'JAX'
    data.loc[data['Name'] == 'WSH', 'Name'] = 'WAS'
    data.loc[data['Name'] == 'LA', 'Name'] = 'LAR'
    data.loc[data['Name'] == 'JAC', 'Name'] = 'JAX'
    data.loc[data['Name'] == 'WSH', 'Name'] = 'WAS'
    
    # merge projected data with contest dataset:
    # merge all positions that arent defense (name columns are different)
    merge_part1 = pd.merge(data, dk, on='Name')
    # merge defense on correct columns
    merge_part2 = pd.merge(data, dk, left_on='Name', right_on='TeamAbbrev')
    merge_part2 = merge_part2[merge_part2['Position'] == 'DST']
    merge_part2 = merge_part2.rename(columns={'Name_y': 'Name'})

    # combine all positions to one dataset
    projections = pd.concat([merge_part1, merge_part2], axis=0, ignore_index=True)
    
    selected_columns = ['Name', 'ID', 'Position', 'Roster Position', 'Salary', 'Game Info', 
                        'TeamAbbrev', 'projPoints', 'FantasyValue', 'PassingC/A',
                        'PassingYds', 'PassingTDs', 'PassingInts', 'RushingAtt', 'RushingYds',
                        'RushingTDs', 'ReceivingRec', 'ReceivingYds', 'ReceivingTDs',
                        'ReceivingTGTS', 'StatsPoints Allowed', 'StatsYards Allowed',
                        'StatsSacks', 'StatsINTs', 'StatsFumbles', 'StatsTDs']
    
    
    projections = projections[selected_columns].sort_values('TeamAbbrev')
    
    # recalculate projected points
    updated_projections = calculate_projected(projections)

    updated_projections.to_csv('projections_raw.csv', index=False) 
    print("projections succesfully created at this location: " + '/projections_raw.csv')
    
def calculate_projected(proj):
    # use stats projections to get correct projected score
    formula = lambda x: (x['PassingYds'] * 0.04) + (x['PassingTDs'] * 4) \
    + (x['PassingInts'] * -1) + (x['RushingYds'] * 0.1) + (x['RushingTDs'] * 6) + \
    + (x['ReceivingRec'] * 1) + (x['ReceivingYds'] * 0.1) + (x['ReceivingTDs'] * 6)
    
    # Apply the formula only to rows where dk['Position'] != 'DST'
    condition = (proj['Position'] != 'DST') & (proj['Position'] != 'K')
    proj['projPoints'] = np.where(condition, proj.apply(formula, axis=1), proj['projPoints'])
    
    return proj


def normalise_names(row):
    extras = ["Jr.", "Sr.", "II", "III", "IV", "V"]
    name = row.Name.split()
    corrected_name = []
    for i in name:      
        if not i in extras:
            corrected_name.append(i)
    new_name = ' '.join(corrected_name)
    new_name = new_name.replace(".", "")
    return new_name


def find_name(names):
    # selects correct parts of player names
    parts = []
    for i in names:
        if i == "D/ST":
            return names[0]
        else:
            parts.append(i)
    
    return ' '.join(parts[2:4])
    

def scrape():
        
    with open('projections/projections_players.html', 'r', encoding='utf-8') as html_file:
        proj_player = html_file.read()
        
    with open('projections/projections_D.html', 'r', encoding='utf-8') as html_file:
        proj_D = html_file.read()
        
        
    table = pd.read_html(proj_player)[3]
    table2 = pd.read_html(proj_D)[3]
    table = pd.concat([table, table2], axis=0, ignore_index=True)
    table.columns = [''.join(col) for col in table.columns]
    table = table.rename(columns={'Unnamed: 0_level_0Player': 'Name'})
    
    # Split the 'FirstColumn' into multiple columns
    split_columns = table['Name'].str.split()
    
    # find name within column
    table['Name'] = split_columns.apply(find_name)
    table = table.rename(columns={'FantasyFP': 'projPoints'})
    table = table.drop_duplicates(subset='Name', keep='last')
    
    table['Name'] = table.apply(normalise_names, axis=1)
    
    return table




    
main()