import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
%matplotlib inline

#Load dataset
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv"
pdf = pd.read_csv(url)
print("Shape of dataset: ", pdf.shape)
pdf.head(5)

#Data Cleaning
print("Shape of the dataset before cleaning: ", pdf.size)
pdf[['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop = True)
print("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

#Feature Selection
featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

#Normalization
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]

#Clustering using Scipy
import scipy 
len = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
	for j in range(leng):
		D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])


import pylab
Z = hierarchy.linkage(D, 'complete')

#if needed to partition the clusters as in flat clustering we can use a cutting line
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters

#clusters can be determined directly as well
from scipy.cluster.hierarchy import fcluster
k=5
cluster = fcluster(z, k, criterion='maxclust')
clusters

#Dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
	return '[%s %s %s]' % (pdf['maunfact'][id], pdf['model'][id], int(float(pdf['type'][id])))

dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')



#Clustering using scikit-learn
dist_matrix = distance_matrix(feature_mtx, feature_mtx)
print(dist_matrix)

agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
agglom.fit(feature_mtx)
agglom.labels_

#Adding a new field to the dataframe to show the cluster of each row
pdf['cluster_'] = agglom.label_
pdf.head()

#Plotting
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0,1,n_clusters))
cluster_labels = list(range(0, n_clusters))

#Create  afigure of size 6 X 4
plt.figure(figsize=(6,4))

for color, label in zip(colors, cluster_labels):
	subset = pdf[pdf.cluster_ == label]
	for i in subset.index:
		plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), roration=25)
	plt.scatter(subset.horsepow, subset.mpg, s=subset.price*10, c=color, label='cluster'+str(label), alpha=0.5)

#plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

#Since there a 2 types of vehicles(truck, car) in the dataset, we'll distinguish them
pdf.groupby(['cluster_', 'type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_', 'type'])['horsepow', 'engine_s', 'mpg', 'price'].mean()
agg_cars

#Visualizing the data
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluslter_labels):
	subset = agg_cars.loc[(label,),]
	for i in subset.index:
		plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
	plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')