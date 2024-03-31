#
# join 2019 census tracts with 2020 precincts
#
#
# missing point" - are there missing geometries?
#
import geopandas as gp
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

precincts_2020 = gp.read_file("/Users/charleskramer/Documents/virginia-voting-precincts-master/maps/va-precinct-shapefiles-2019-2020/va-precincts-2019")
precincts_2020 = precincts_2020.to_crs("EPSG:3447")
precincts_2020['rep_pt'] = precincts_2020.representative_point()

tracts_2019 = gp.read_file("/Users/charleskramer/Documents/Va Census Tracts/2019/tl_2019_51_tract")
tracts_2019 = tracts_2019.to_crs("EPSG:3447")
tracts_2019['rep_pt'] = tracts_2019.representative_point()

#https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas

def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.rep_pt.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.rep_pt.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)

    return gdf

tracts_2019.rename(columns={'precincts':'tract_precincts'},inplace=True)

distances = ckdnearest(precincts_2020,tracts_2019)

distances.to_csv('distances.csv')

# dist = gp.read_file('distances.csv',ignore_geometry=True)
# print("data loaded")
# dist['geometry'] = gp.GeoSeries.from_wkt(dist['geometry'])
# dist.plot()
# plt.show()
#
# dictionary to map precUid to tract
#
prec2tract = dict(zip(distances['precUid'].tolist(),distances['GEOID'].tolist()))

with open("prec2tract.txt",'wb') as file:
    pickle.dump(prec2tract,file)