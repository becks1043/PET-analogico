#flood_map
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.segmentation import watershed
from matplotlib.patches import Circle
#visualizzare e discutere le flood map 
x = np.random.randn(1000) #qua vanno gli array di cosa vedono i rivelatori
y = np.random.randn(1000) #per adesso ho simulato die dati
h2d, xe, ye, = np.histogram2d(x,y, bins = (200,200), range = [(-1,1), (-1, 1)])
plt.imshow(h2d, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]]) #h2d.T fa la trasfosta degli eventi
plt.colorbar(label="Conteggi")
plt.xlabel("Asse x")
plt.ylabel("Asse y")
plt.title("Flood Map")
plt.show()

#ricerchiamo i centri

fig, ax  = plt.subplots(1,1, figsize = (6, 6))
ax.imshow(h2d, interpolation="nearest")#vmax definisce la colorazione dell'immagine

blobs = blob_log(h2d, min_sigma=0.5, max_sigma=0.6, threshold=0.15)

for y, x, s in blobs:
    ax.add_patch(Circle((x,y),s ,color = "y" , lw= 1, fill= False))#color = "y"

#se non troviamo dei punti dobbiamo aggiungerli manualmente
sigma = 0.6
additional_blobs = [(60.89,171 ,sigma), (57.98,178. ,sigma)] #list like [(..,..,..)]

for y, x, s in additional_blobs:
    ax.add_patch(Circle((x,y),s ,color = "r" , lw= 1, fill= False))

blobs = np.concatenate([blobs, np.array(additional_blobs)])
#numerazione dei blob
b = blobs[blobs[:,0].argsort()]
markers = np.zeros_like(h2d, dtype = "int8")
markers[b[:,0].astype(int),b[:,1].astype(int)] = np.arange(b[:,0].size) +1

for i, (y,x, s) in enumerate(b):
    ax.add_patch(Circle((x,y),s ,color = "r" , lw= 1, fill= False))
    ax.text(x, y, i+1, ha="center", va = "center", color="white", fontsize=5)

plt.xlabel("Asse x")
plt.ylabel("Asse y")
plt.show()

#generazione della LUT
fig, ax  = plt.subplots(1,1, figsize = (6, 5))
lut = np.zeros_like(h2d, dtype="int16")
lut = watershed(lut, markers) - 1

ax.imshow(lut*1e4, interpolation="nearest", cmap= "tab20")
plt.xlabel("Asse x")
plt.ylabel("Asse y")
plt.title("LUT")
plt.show()
"""
#identificazione dei pixel
pixel_ids = lut[y,x]

for pixel in sorted(set(pixel_ids)):
    pix_en = e[pixel_ids ==pixel] #mi s ache e è l'energia dell'evento
    n, b = np.histogram(pix_en, bins=100)
    bc = 0.5*(b[1:] + b[:-1])
    photopeak = bc[n.argamax()]
    plt.bar(bc, n, np.diff(b))
    plt.axvline(photopeak, ls="--", lw=2, c = "k")
"""



