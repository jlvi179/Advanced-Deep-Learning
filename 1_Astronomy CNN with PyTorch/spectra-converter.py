# This notebook packs spectra from GALAH dr4 - hermes
# Step 1: Download galah_dr4_allstar_*.fits with all 1e6 star properties.
# Step 2: Select a subset of these stars and save the ids and some physical parameters
# Step 3: Bulk-download by list of ids from https://datacentral.org.au/services/download/ (select dr4-hermes)
# Step 4: Combine cameras, pack spectra and labels into numpy files

#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

#%%

# Source: https://www.galah-survey.org/details/facilities/
plt.figure(figsize=(6,1))
plt.plot([4718,4903],[0,0],label="Blue", linewidth=5, c="tab:blue")
plt.plot([5649,5873],[0,0],label="Green",linewidth=5, c="tab:green")
plt.plot([6481,6739],[0,0],label="Red",  linewidth=5, c="tab:red")
plt.plot([7590,7890],[0,0],label="IR",   linewidth=5, c="tab:brown")
plt.title("GALAH camera coverage")
plt.xlabel("Wavelength [Å]")
plt.yticks([])
plt.savefig("galah_coverage.svg", bbox_inches="tight") # markdown does not support pdf :(
plt.show()



#%%

## Step 2
def collectStarData(sourceFile, num, labels):
	starData = {}
	with fits.open(sourceFile) as hdul:
		colID = hdul[1].data["sobject_id"]
		colFlags = np.stack([hdul[1].data["flag_red"], hdul[1].data["flag_sp"], hdul[1].data["snr_px_ccd3"]<30])
		columns  = [hdul[1].data[k] for k in labels]
		i = 0
		while len(starData) < num:
			if colFlags[:,i].sum() == 0:
				tmp = [c[i] for c in columns]
				if not np.any(np.isnan(tmp)):
					starData[str(colID[i])] = tmp
			i += 1
		# hdul.info()
		# print(hdul[1].header)
		# for i in range(1,184):
		# 	print(i, hdul[1].header["TTYPE"+str(i)], "->", hdul[1].header["TCOMM"+str(i)])
	return starData

labels = ["mass","age","lbol","r_med","teff","logg","fe_h","snr_px_ccd3"] # galah names
labelNames = ["mass","age","lbol","dist","teff","logg","feh","snr"] # friendly names
starData = collectStarData("galah_dr4_allstar_240705.fits", 100000, labels)
print(f"Selected {len(starData)} stars.")

# Save ids to a txt file for manual spectra download
with open("galah_ids.txt","w") as file:
	for id in starData.keys():
		file.write(id + ",\n")


#%%

## Check if anything correlates with index
data = np.array([starData[id] for id in starData.keys()])
corrMatrix = np.corrcoef(np.c_[np.arange(len(data)),data],rowvar=False)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corrMatrix, interpolation='nearest')
fig.colorbar(cax)
plt.xticks(np.arange(9),["#"]+labelNames)
plt.yticks(np.arange(9),["#"]+labelNames)
plt.tight_layout()
plt.savefig("label_correlation.svg")
plt.show()


#%%

## Step 4
# Every star is observed by 4 cameras, indicated by the last digit of the filename.
# Even though the wavelength windows are not contiguous, we just concatenate these individual spectra.
# Sometimes not all four files exist. We drop these targets.
def loadFitsSpectraDR4(dir:str,starData:dict):
	# Collect files per object:
	allFilesCom = np.concatenate([[os.path.join(root,f) for f in files if f.endswith(".fits")] for root,_,files in os.walk(dir) if files!=[]])
	objs = {}
	for f in allFilesCom[:]: 
		with fits.open(f) as hdul:
			he = hdul[0].header
			id = he["OBJECT"]
			if id not in objs:
				objs[id] = [f]
			else:
				objs[id].append(f)

	# Merge spectra
	allFluxes = []
	for id,files in objs.items():
		spectrum = np.zeros(4*4096)
		for i,f in enumerate(np.sort(files)):
			with fits.open(f) as hdul:
				spectrum[4096*i:4096*(i+1)] = hdul[1].data
		allFluxes.append(spectrum)
		
	spectra = np.stack(allFluxes)
	labels = np.array([starData[id] for id in objs.keys()])
	return spectra, labels

galahSpectra0, galahLabels0 = loadFitsSpectraDR4("hermes/com/", starData)

#%%

# Remove NaN stars
mask = np.all(np.isfinite(galahLabels0),axis=1)
print(f"Removed {len(galahLabels0)-mask.sum()} stars.")
galahSpectra = galahSpectra0[mask]
galahLabels  = galahLabels0[mask]

#Remove strong outliers 
p=0.01
ranges = np.percentile(galahLabels,[100*p,100*(1-p)],axis=0)
mask = np.logical_and(np.all(galahLabels<ranges[1],axis=1), np.all(ranges[0]<galahLabels,axis=1))
galahSpectra = galahSpectra[mask]
galahLabels = galahLabels[mask]

print(galahSpectra.shape)
np.save("spectra.npy", galahSpectra)
np.save("labels.npy", galahLabels)
