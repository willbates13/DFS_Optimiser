#!/usr/bin/env python3

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
tqdm.pandas()
from statsmodels.stats.correlation_tools import cov_nearest


def main():
    # Load datasets
    data = pd.read_csv('historical_data_random_noise.csv').fillna(0)
    dk = pd.read_csv('projections_raw.csv').fillna(0)[:100]
    # filter data by projected points
    dk = dk[dk.projPoints >= 0.1]
    # data column adjustment
    data.loc[data['position'] == 'D', 'position'] = 'DST'
    
    # from the input projections and historical data return dataset with variance/covariance included.
    dk, covariance_df = find_stats(dk, data, 1)

    dk = dk.sort_values('ID')

    dk.to_csv('projections.csv', index=False)
    covariance_df.reset_index(drop=False).to_csv('covariance_matrix.csv', index=False) 
    

def find_stats(dk, data, threshold):
 
    # calculate variance and store similar data in dict for covar calc
    variance_list = []
    similar_historical_data = {}
    for _, player in dk.iterrows():
        similar_df = similar(player, data)
        
        mean = np.average(similar_df['actualPoints'])
        
        variance = np.average((similar_df['actualPoints'] - mean) ** 2)

        variance_list.append(variance)
        
        similar_historical_data[player.ID] = similar_df

    dk['variance'] = variance_list
                
    
    # Merge the 'dk' DataFrame with itself based on 'Game Info' to create all combinations with non zero covariance
    dk_pairs = dk.merge(dk, on='Game Info', suffixes=('_1', '_2')).query('ID_1 != ID_2')

    # calculate covariance
    dk_pairs['covariance'] = dk_pairs.progress_apply(lambda row: calc_covar(similar_historical_data, row), axis=1)
    dk_pairs['covariance'] = dk_pairs['covariance'].astype(float)
    
    covariance_df = create_covar_matrix(dk, dk_pairs)
    
    return dk, covariance_df


def similar(player, data):

    x = data[data.position.isin([player.Position])]
    x['diff_points'] = np.abs(x.projPoints - player.projPoints)
    
    # if enough data, take the top 1/3rd. if not, take the top 2/3rds
    if len(x) > 100:
        cutoff = int(round(len(x)*0.33,0))
    else:
        cutoff = int(round(len(x)*0.66,0))
        
    # cutting the data to not include any player duds, removes injury noise
    if player.projPoints > 1 and player.Position != 'DST':
        x = x[x.actualPoints > 0]
        
    x = x.sort_values(by=['diff_points'])[:cutoff]

    return x


def calc_covar(similar_historical_data, pair):
    # pair = dk_pairs.loc[4]
    name1, id1, pos1, t1 = pair[['Name_1', 'ID_1', 'Position_1', 'TeamAbbrev_1']]
    name2, id2, pos2, t2 = pair[['Name_2', 'ID_2', 'Position_2', 'TeamAbbrev_2']]

    if name1 == name2:
        covariance = 0
        return covariance
    teammates = t1 == t2
    same_position = pos1 == pos2

    a = similar_historical_data[id1]
    b = similar_historical_data[id2]
    
    # Merge the DataFrames based on 'eventId' and 'playerId'
    data_pairs = a.merge(b, on='eventId', suffixes=('_1', '_2'))
    

    # Filter out rows with the same playerId
    data_pairs = data_pairs[data_pairs['playerId_1'] != data_pairs['playerId_2']]
    
    if same_position:
        data_pairs = data_pairs[data_pairs['position_1'] == data_pairs['position_2']]
    else:
        data_pairs = data_pairs[data_pairs['position_1'] != data_pairs['position_2']]

    if teammates:
        data_pairs = data_pairs[data_pairs['currentTeam_1'] == data_pairs['currentTeam_2']]
    else:
        data_pairs = data_pairs[data_pairs['currentTeam_1'] != data_pairs['currentTeam_2']]
        
    data_pairs['diff_points_total'] = data_pairs['diff_points_1'] + data_pairs['diff_points_2']
    
    if len(data_pairs) > 100:
        filtered_data_pairs = data_pairs.sort_values(by=['diff_points_total'])[:100]
    else:
        print('not enough similar data between ' + name1 + " and " + name2)
        filtered_data_pairs = data_pairs.copy()
        
    # Calculate the weighted covariance
    covariance = np.cov(filtered_data_pairs['actualPoints_1'], filtered_data_pairs['actualPoints_2'])

    return covariance[0, 1] 


def is_positive_semi_definite(matrix):

    # Check if a covariance matrix is positive semi-definite.

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Check if all eigenvalues are non-negative
    eigenvalues, _ = np.linalg.eigh(matrix)
    
    return np.all(eigenvalues >= 0)


def create_covar_matrix(dk, dk_pairs):
    
    player_ids = dk['ID'].to_list()
    # Create an empty DataFrame with player IDs as row and column index
    covariance_df = pd.DataFrame(index=player_ids, columns=player_ids)
    # Populate the covariance values in the DataFrame
    for _, row in dk_pairs.iterrows():
        player_id1 = row['ID_1']
        player_id2 = row['ID_2']
        covariance_value = row['covariance']
        
        covariance_df.loc[player_id1, player_id2] = covariance_value
        covariance_df.loc[player_id2, player_id1] = covariance_value
    
    # dataframe manipulation to allow for easy indexing in to covariance dataframe
    dk = dk.sort_values('ID')
    sorted_ID = dk['ID']
    covariance_df = covariance_df.sort_index()
    covariance_df = covariance_df.reindex(columns = sorted_ID)
    variance_values = dk.set_index('ID')['variance']
    # Set the diagonal elements of the covariance matrix
    for var, idx in zip(variance_values, variance_values.index):
        covariance_df[idx][idx] = var
        
    covariance_df = covariance_df.fillna(0)
    dk = dk.fillna(0)
    
    # ensure indexs are numeric
    covariance_df.index = pd.to_numeric(covariance_df.index, errors='coerce')
    covariance_df.columns = pd.to_numeric(covariance_df.columns, errors='coerce')
    
    # ensure covariance matrix is positive semi-definite
    cov_array = covariance_df.to_numpy()
    if not is_positive_semi_definite(cov_array):
        print('Covariance matrix is not positive semi-definite')
        cov_array = cov_nearest(cov_array, method='nearest', threshold=1e-15, n_fact=10, return_all=False)
        covariance_df = pd.DataFrame(cov_array, index=covariance_df.index, columns=covariance_df.columns)
        print('Covariance matrix has been adjusted to nearest positive semi-definite form')
    else:
        print('Covariance matrix is positive semi-definite!')
        
    return covariance_df


main()




