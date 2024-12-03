#%%
import os
os.environ["CDF_LIB"] = "~/CDF/lib"
import pandas as pd
import glob
import numpy as np
import math
import scipy.io
from spacepy import pycdf
import matplotlib.pyplot as plt
from tqdm import tqdm

twins_dir = '/home/jmarchezi/python_projects/read_twins_data/data/twins/'
#%%
def loading_twins_maps(full_map=False):
	'''
	Loads the twins maps

	Returns:
		maps (dict): dictionary containing the twins maps
	'''

	times = pd.read_feather('outputs/regular_twins_map_dates.feather')
	twins_files = sorted(glob.glob(twins_dir+'*.cdf', recursive=True))

	maps = {}

	for file in twins_files:
		twins_map = pycdf.CDF(file)
		for i, date in enumerate(twins_map['Epoch']):
			if full_map:
				if len(np.unique(twins_map['Ion_Temperature'][i])) == 1:
					continue
			else:
				if len(np.unique(twins_map['Ion_Temperature'][i][35:125,50:110])) == 1:
					continue
			check = pd.to_datetime(date.strftime(format='%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
			if check in times.values:
				maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')] = {}
				if full_map:
					maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')]['map'] = twins_map['Ion_Temperature'][i]
				else:
					maps[check.round('T').strftime(format='%Y-%m-%d %H:%M:%S')]['map'] = twins_map['Ion_Temperature'][i][35:125,50:110]

	return maps
# %%


def mlt_wedge(img_data, mlt_min: int = 18, mlt_max: int = 6, center_x: int = 120, center_y: int = 80):
    """
    This function takes in a temperature ENA image and divides thems into pie slices based on the
    numbers of slices desired. The output of the function is an array of the mean different sections of the
    ENA temperature map generated and the different sections themselves, respectively.
    Parameters:
    -----------
    img_data : ndarray
        Input image data.
    mlt_min : int, optional
        Minimum MLT value. Default is 18.
    mlt_max : int, optional
        Maximum MLT value. Default is 6.
    mlt_span : int, optional
        MLT span. Default is 1.
    angle_steps : int, optional
        Number of angle steps for pie slices. Default is 8.
    center_x : int, optional
        X-coordinate of the center. Default is 120.
    center_y : int, optional
        Y-coordinate of the center. Default is 80.
    Returns:
    --------
    selected_sections : list
        List of selected pie slice sections.
    """
    # get the dimension of the image
    height, width = img_data.shape
    # Create an empty list to store the selected sections
    selected_sections = []
    img_mean = []
    # define the numbers of angles used for the pie slices
    # Create masks for each N-degree section and apply them to the image
    # Create a new blank mask as a NumPy array
    mask = np.zeros((height, width), dtype=np.uint8)
    img_copy = np.copy(img_data)
    # Calculate the coordinates of the sector's bounding box

    start_angle = (mlt_min*15) % 360
    end_angle = (mlt_max*15) % 360

    # print(f'start_angle: {start_angle}, end_angle: {end_angle}')
    # Calculate the coordinates of the sector arc
    for y in range(height):
        for x in range(width):
            # Calculate the polar coordinates of the pixel relative to the image center
            dx = center_x - x
            dy = center_y - y
            pixel_angle = math.degrees(math.atan2(dy, dx))  # Calculate the angle in radians
            if pixel_angle < 0:
                pixel_angle += 360
            # Check if the pixel is within the current 45-degree section
            if mlt_min > mlt_max:
                if start_angle <= pixel_angle < 360 or 0 <= pixel_angle < end_angle:
                    mask[y, x] = 1
            else:
                if start_angle <= pixel_angle < end_angle:
                    mask[y, x] = 1  # Set the pixel to white (255)
    # Apply the mask to the heat map to select the section
    img_copy[mask == 0] = 0
    # Append the selected section to the list
    selected_sections = img_copy
    xx = np.copy(img_copy)
    xx[xx == 0] = np.nan
    # Get the mean of the non zero values of the image
    img_mean.append(np.nanmean(xx))
    # print(f'mean: {np.nanmean(xx)}')
    return img_mean, selected_sections
# %%


def plot_polar(icmes, hss, phase):
    y_hss =hss.values    # Replace with your actual HSS y values
    y_icme = icmes.values     # Replace with your actual ICME y values

    # Correctly map MLT from 18 to 24, then 0 to 6
    mlt_hours = np.concatenate((np.linspace(18, 24, len(y_hss) // 2, endpoint=False),
                                np.linspace(0, 6, len(y_hss) // 2, endpoint=False)))

    # Convert MLT to radians, mapping 18?6 MLT to 180�?360�
    angles = np.deg2rad(mlt_hours * 15)  # Convert MLT hours to degrees and then to radians

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot the data
    ax.plot(angles, y_hss, marker='o', linestyle='-', label='HSS')
    ax.plot(angles, y_icme, marker='x', linestyle='-', label='ICME')
    ax.set_theta_zero_location('S')
    # Configure the plot to show only the 180�?360� portion
    ax.set_theta_direction(1)  # Clockwise
    # ax.set_theta_offset(np.pi)  # Start at 180� (midnight)

    # Set custom tick marks for MLT
    custom_ticks = [18, 20, 22, 0, 2, 4, 6]  # MLT labels
    custom_tick_angles = np.deg2rad([-90, -60, -30, 0, 30, 60, 90])  # Convert MLT to radians
    ax.set_xticks(custom_tick_angles)
    ax.set_xticklabels(custom_ticks)
    # ax.set_rlabel_position(5)  # Move grid labels away from other labels
    ax.set_thetamin(90) # only show top half
    ax.set_thetamax(-90)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2)
    ax.text(0.5, 0.9, f"Mean Twins Ion Temperature maps \n for HSS and ICME {phase}", 
            horizontalalignment='center', 
            verticalalignment='center', 
            size='large',
            transform=ax.transAxes)
    ax.text(0.5, 0.135, f"MLT", 
            horizontalalignment='center', 
            verticalalignment='center', 
            size='medium',
            transform=ax.transAxes)
    # ax.set_title()
    plt.savefig(f'figures/mean_density_ICME_HSS_{phase}.png', dpi=300, bbox_inches='tight')

# %%

def get_the_twins_mlt_mean(df, step, window_maps):
    dados = []
    medias = []
    for key in tqdm(df['datetime'].values):
        mlt_means = {}
        for mlt in np.arange(18,24,step):
            mlt_min = mlt
            mlt_max = mlt+step
            window_img_mean, window_selected_sections = mlt_wedge(window_maps[key]['map'], 
                                                        mlt_min=mlt_min, mlt_max=mlt_max, 
                                                        center_x=window_maps[key]['map'].shape[1], 
                                                        center_y=int(window_maps[key]['map'].shape[0]/2))
            mlt_means[f"{mlt:.1f}"] = window_img_mean[0]
        for mlt in np.arange(0,6.5,step):
            mlt_min = mlt
            mlt_max = mlt+step
            window_img_mean, window_selected_sections = mlt_wedge(window_maps[key]['map'], 
                                                        mlt_min=mlt_min, mlt_max=mlt_max, 
                                                        center_x=window_maps[key]['map'].shape[1], 
                                                        center_y=int(window_maps[key]['map'].shape[0]/2))
            mlt_means[f"{mlt:.1f}"] = window_img_mean[0]
        medias.append(mlt_means)
    return medias