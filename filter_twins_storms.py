#%%
import numpy as np
import pandas as pd

from functions import *

#%%

full_maps = loading_twins_maps(full_map=True)
window_maps = loading_twins_maps(full_map=False)
#%%
AUX_PATH = "/home/jmarchezi/research-projects/gic-statistics/auxData"

stormList = pd.read_csv(f"{AUX_PATH}/StormList2_manPhStart_NewBeginEnd.csv")
stormList.rename(columns={"Unnamed: 0": "stormNumber"}, inplace=True)

#%%
datetime_twins = pd.to_datetime(list(window_maps.keys()))

# Convert storm times to datetime once
stormList['new_begining_times'] = pd.to_datetime(stormList['new_begining_times'])
stormList['main_phase'] = pd.to_datetime(stormList['main_phase'])
stormList['minimumSymH'] = pd.to_datetime(stormList['minimumSymH'])
stormList['new_ending_times'] = pd.to_datetime(stormList['new_ending_times'])

# Convert datetime_twins to DataFrame
datetime_twins_df = pd.DataFrame({'datetime': datetime_twins})

# Filter storms that overlap with the datetime_twins range
overall_start = datetime_twins_df['datetime'].min()
overall_end = datetime_twins_df['datetime'].max()
stormList_filtered = stormList[
    (stormList['new_begining_times'] <= overall_end) &
    (stormList['new_ending_times'] >= overall_start)
]

# Initialize an empty list to store results
results = []

# Iterate through datetime_twins and assign storm information
for date in datetime_twins_df['datetime']:
    storm_match = stormList_filtered[
        (stormList_filtered['new_begining_times'] <= date) &
        (stormList_filtered['new_ending_times'] >= date)
    ]
    
    if not storm_match.empty:
        # Extract the first matching storm (assuming no overlaps)
        storm = storm_match.iloc[0]
        if storm['new_begining_times'] <= date <= storm['main_phase']:
            phase = 'ini'
        elif storm['main_phase'] <= date <= storm['minimumSymH']:
            phase = 'main'
        elif storm['minimumSymH'] <= date <= storm['new_ending_times']:
            phase = 'rec'
        else:
            phase = ''
        
        # Append the result
        results.append({
            'datetime': date,
            'stormNumber': storm['stormNumber'],
            'ifCME': storm['ifCME'],
            'ifHSS': storm['ifHSS'],
            'ifcomplex': storm['ifcomplex'],
            'phase': phase
        })
    else:
        # If no storm matches, add empty values
        results.append({
            'datetime': date,
            'stormNumber': None,
            'ifCME': None,
            'ifHSS': None,
            'ifcomplex': None,
            'phase': None
        })

# Convert results to a DataFrame or dictionary
results_df = pd.DataFrame(results)
#%%
results_df.to_csv(f'{AUX_PATH}/twins_maps_storms_dates.csv', index=False)
#%%
data_assoc_with_storm = results_df.dropna(subset=['stormNumber'])
#%%
data_assoc_with_storm.to_csv(f'{AUX_PATH}/twins_data_assoc_with_storm.csv', index=False)
#%%
import math
key = '2009-07-21 00:10:00'
full_img_mean, full_selected_sections = mlt_wedge(full_maps[key]['map'], mlt_min=0, mlt_max=6, center_x=full_maps[key]['map'].shape[0]-20, center_y=int(full_maps[key]['map'].shape[1]/2))

fig, ax = plt.subplots(2, 3, figsize=(20, 15))
vmin = np.nanmin(full_maps[key]['map'])
vmax = np.nanmax(full_maps[key]['map'])
ax[0,0].imshow(full_maps[key]['map'], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
ax[0,0].axhline(y=80, color='r', linestyle='--')
ax[0,2].imshow(full_selected_sections, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
ax[0,1].imshow(full_selected_sections[35:125,50:110], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')

window_img_mean, window_selected_sections = mlt_wedge(window_maps[key]['map'], mlt_min=18, mlt_max=6, center_x=window_maps[key]['map'].shape[0]-20, center_y=int(window_maps[key]['map'].shape[1]/2))
ax[1,0].imshow(window_maps[key]['map'], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
ax[1,0].axhline(y=45, color='r', linestyle='--')
ax[1,1].imshow(window_selected_sections, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
ax[1,2].imshow(full_maps[key]['map'][35:125,50:110], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
# %%
