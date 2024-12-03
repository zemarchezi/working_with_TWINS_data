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
# # %%
# key= data_HSS['datetime'].values[10]
# window_img_mean, window_selected_sections = mlt_wedge(window_maps[key]['map'], mlt_min=5, mlt_max=6, center_x=window_maps[key]['map'].shape[1], center_y=int(window_maps[key]['map'].shape[0]/2))
# fig, ax = plt.subplots(1, 3, figsize=(20, 15))
# vmin = np.nanmin(window_maps[key]['map'])
# vmax = np.nanmax(window_maps[key]['map'])
# ax[0].imshow(window_maps[key]['map'], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
# ax[0].axhline(y=int(window_maps[key]['map'].shape[0]/2), color='r', linestyle='--')
# ax[1].imshow(window_selected_sections, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
# ax[2].imshow(window_maps[key]['map'][35:125,50:110], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
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
