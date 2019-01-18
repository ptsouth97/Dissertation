#!/usr/bin/python3

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def main():
	'''  Main function for testing purposes '''

	samples = pd.read_csv('seeds_dataset.csv', header=None)
	samples = samples.drop(samples.columns[7], axis=1)
	samples = np.array(samples)
	print(samples.shape)
	num_of_clusters(samples)


def inertia_plot(data):
	''' Plots inertia vs number of clusters to aid in finding a good number of clusters for a dataset'''

	ks = range(1, 46)
	inertias = []

	for k in ks:
    	# Create a KMeans instance with k clusters: model
		model = KMeans(n_clusters=k)
    
    	# Fit model to samples
		model.fit(data)
    
		# Append the inertia to the list of inertias
		inertias.append(model.inertia_)
    
	# Plot ks vs inertias
	fig = plt.figure()
	plt.plot(ks, inertias, '-o', color='black')
	plt.xlabel('number of clusters, k')
	plt.ylabel('inertia')
	plt.xticks(ks)
	plt.title('K-Means Clustering')
	caption = '$\it{Fig 2.}$ The inertia values represent how far the samples are from their centroid for k clusters. Lower inertia values indicate a better clustering. The best k value is the elbow, the point where the inertia values begin to decrease less rapidly. Although there no clear elbow here, it appears to be at about k=6'
	fig.text(0.01, 0.01, caption, ha='left', wrap=True)
	
	plt.savefig('KMeans Clustering.png')
	plt.show()


if __name__ == '__main__':
	main()
