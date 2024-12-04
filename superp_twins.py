#%%
import os
os.environ["CDF_LIB"] = "~/CDF/lib"
import pandas as pd
import glob
import numpy as np
import scipy.io
from spacepy import pycdf
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import *

#%%

AUX_PATH = "/home/jmarchezi/research-projects/gic-statistics/auxData"

window_maps = loading_twins_maps(full_map=False)
#%%

twins_storms = pd.read_csv(f"{AUX_PATH}/twins_data_assoc_with_storm.csv")
# %%
struc = "ICME"
maskStruc = (twins_storms[f"if{struc.replace('I','')}"] == f"{struc}")
data_ICME = twins_storms[maskStruc]
icme_ini = data_ICME[data_ICME['phase'] == 'ini']
icme_mai = data_ICME[data_ICME['phase'] == 'main']
icme_rec = data_ICME[data_ICME['phase'] == 'rec']


struc = "HSS"
maskStruc = (twins_storms[f"if{struc.replace('I','')}"] == f"{struc}")
data_HSS = twins_storms[maskStruc]
hss_ini = data_HSS[data_HSS['phase'] == 'ini']
hss_mai = data_HSS[data_HSS['phase'] == 'main']
hss_rec = data_HSS[data_HSS['phase'] == 'rec']
# %%
key= twins_storms['datetime'].values[1]
window_img_mean, window_selected_sections = mlt_wedge(window_maps[key]['map'], mlt_min=3, mlt_max=4, center_x=window_maps[key]['map'].shape[1], center_y=int(window_maps[key]['map'].shape[0]/2))
fig, ax = plt.subplots(1, 3, figsize=(20, 15))
vmin = np.nanmin(window_maps[key]['map'])
vmax = np.nanmax(window_maps[key]['map'])
ax[0].imshow(window_maps[key]['map'], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
ax[0].axhline(y=int(window_maps[key]['map'].shape[0]/2), color='r', linestyle='--')
ax[1].imshow(window_selected_sections, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
ax[2].imshow(window_maps[key]['map'][35:125,50:110], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
# %%




# # %%
# medias_icme = get_the_twins_mlt_mean(data_ICME, 0.3, window_maps)
# meansICME = pd.DataFrame(medias_icme)
# meansICME[meansICME == -1.0] = np.nan
# meansICME.to_csv(f"means_icme.csv")

# medias_icme_ini = get_the_twins_mlt_mean(icme_ini, 0.3, window_maps)
# meansICME_ini = pd.DataFrame(medias_icme_ini)
# meansICME_ini[meansICME_ini == -1.0] = np.nan
# meansICME_ini.to_csv(f"means_icme_ini.csv")

# medias_icme_mai = get_the_twins_mlt_mean(icme_mai, 0.3, window_maps)
# meansICME_mai = pd.DataFrame(medias_icme_mai)
# meansICME_mai[meansICME_mai == -1.0] = np.nan
# meansICME_mai.to_csv(f"means_icme_mai.csv")

# medias_icme_rec = get_the_twins_mlt_mean(icme_rec, 0.3, window_maps)
# meansICME_rec = pd.DataFrame(medias_icme_rec)
# meansICME_rec[meansICME_rec == -1.0] = np.nan
# meansICME_rec.to_csv(f"means_icme_rec.csv")

# medias_hss = get_the_twins_mlt_mean(data_HSS, 0.3, window_maps)
# meansHSS = pd.DataFrame(medias_hss)
# meansHSS[meansHSS == -1.0] = np.nan
# meansHSS.to_csv(f"means_hss.csv")

# medias_hss_ini = get_the_twins_mlt_mean(hss_ini, 0.3, window_maps)
# meansHSS_ini = pd.DataFrame(medias_hss_ini)
# meansHSS_ini[meansHSS_ini == -1.0] = np.nan
# meansHSS_ini.to_csv(f"means_hss_ini.csv")

# medias_hss_mai = get_the_twins_mlt_mean(hss_mai, 0.3, window_maps)
# meansHSS_mai = pd.DataFrame(medias_hss_mai)
# meansHSS_mai[meansHSS_mai == -1.0] = np.nan
# meansHSS_mai.to_csv(f"means_hss_mai.csv")

# medias_hss_rec = get_the_twins_mlt_mean(hss_rec, 0.3, window_maps)
# meansHSS_rec = pd.DataFrame(medias_hss_rec)
# meansHSS_rec[meansHSS_rec == -1.0] = np.nan
# meansHSS_rec.to_csv(f"means_hss_rec.csv")
#%%

meansICME = pd.read_csv(f"outputs/means_icme.csv")
meansICME.drop(columns=['Unnamed: 0'], inplace=True)
meansHSS = pd.read_csv(f"outputs/means_hss.csv")
meansHSS.drop(columns=['Unnamed: 0'], inplace=True)
meansICME_ini = pd.read_csv(f"outputs/means_icme_ini.csv")
meansICME_ini.drop(columns=['Unnamed: 0'], inplace=True)
meansHSS_ini = pd.read_csv(f"outputs/means_hss_ini.csv")
meansHSS_ini.drop(columns=['Unnamed: 0'], inplace=True)
meansICME_mai = pd.read_csv(f"outputs/means_icme_mai.csv")
meansICME_mai.drop(columns=['Unnamed: 0'], inplace=True)
meansHSS_mai = pd.read_csv(f"outputs/means_hss_mai.csv")
meansHSS_mai.drop(columns=['Unnamed: 0'], inplace=True)
meansICME_rec = pd.read_csv(f"outputs/means_icme_rec.csv")
meansICME_rec.drop(columns=['Unnamed: 0'], inplace=True)
meansHSS_rec = pd.read_csv(f"outputs/means_hss_rec.csv")
meansHSS_rec.drop(columns=['Unnamed: 0'], inplace=True)
#%%


plot_polar(meansICME.mean(axis=0), meansHSS.mean(axis=0), "")
plot_polar(meansICME_ini.mean(axis=0), meansHSS_ini.mean(axis=0), "- Initial")
plot_polar(meansICME_mai.mean(axis=0), meansHSS_mai.mean(axis=0), "- Main")
plot_polar(meansICME_rec.mean(axis=0), meansHSS_rec.mean(axis=0), "- Recovery")


# %%
list_ICME = [meansICME.mean(axis=0), meansICME_ini.mean(axis=0), meansICME_mai.mean(axis=0), meansICME_rec.mean(axis=0)]
list_HSS = [meansHSS.mean(axis=0), meansHSS_ini.mean(axis=0), meansHSS_mai.mean(axis=0), meansHSS_rec.mean(axis=0)]
phases = ["", "- Initial", "- Main", "- Recovery"]


figprops = dict(nrows=2, ncols=2,sharex=False,
                    constrained_layout=False, 
                    figsize=(10, 10), dpi=200,
                    subplot_kw=dict(polar=True))

fig, axs = plt.subplots(**figprops)
axs = axs.flatten()
letters = ['a)', 'b)', 'c)', 'd)']
fig.subplots_adjust(hspace=-0.2)
for ii in range(len(list_HSS)):

    y_hss =list_HSS[ii].values    
    y_icme = list_ICME[ii].values     
    mlt_hours = np.concatenate((np.linspace(18, 24, len(y_hss) // 2, endpoint=False),
                                np.linspace(0, 6, len(y_hss) // 2, endpoint=False)))

    
    angles = np.deg2rad(mlt_hours * 15)  # Convert MLT hours to degrees and then to radians

    phase = phases[ii]


    # Plot the data
    axs[ii].plot(angles, y_hss, marker='o', linestyle='-', label='HSS')
    axs[ii].plot(angles, y_icme, marker='x', linestyle='-', label='ICME')
    axs[ii].set_theta_zero_location('S')
    # Configure the plot to show only the 180�?360� portion
    axs[ii].set_theta_direction(1)  # Clockwise
    axs[ii].set_rlim(0,5.5, 1)
    # ax.set_theta_offset(np.pi)  # Start at 180� (midnight)

    # Set custom tick marks for MLT
    custom_ticks = [18, 20, 22, 0, 2, 4, 6]  # MLT labels
    custom_tick_angles = np.deg2rad([-90, -60, -30, 0, 30, 60, 90])  # Convert MLT to radians
    axs[ii].set_xticks(custom_tick_angles)
    axs[ii].set_xticklabels(custom_ticks)
    # ax.set_rlabel_position(5)  # Move grid labels away from other labels
    axs[ii].set_thetamin(90) # only show top half
    axs[ii].set_thetamax(-90)
    axs[ii].legend(loc='lower center', bbox_to_anchor=(0.5, 0.026), ncol=2)
    axs[ii].text(0.5, 0.9, f"Mean TWINS Ion Temperature \n for HSS and ICME {phase}", 
            horizontalalignment='center', 
            verticalalignment='center', 
            size='large',
            transform=axs[ii].transAxes)
    axs[ii].text(0.5, 0.14, f"MLT", 
            horizontalalignment='center', 
            verticalalignment='center', 
            size='medium',
            transform=axs[ii].transAxes)
    axs[ii].text(-0.1, 0.9, f"{letters[ii]}", 
            horizontalalignment='center', 
            verticalalignment='center', 
            size='large',
            transform=axs[ii].transAxes)

plt.savefig(f'figures/mean_density_ICME_HSS.png', dpi=300, bbox_inches='tight')

# ax.set_title()
# %%
