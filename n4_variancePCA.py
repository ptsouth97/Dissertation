#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
	''' Main function for testing purposes'''

	pass


def PCA_variance(samples):
	''' Makes a plot of the variances of the PCA features to help find the instrinsic dimension'''

	# Create scaler: scaler
	#scaler = StandardScaler()

	# Create a PCA instance: pca
	pca = PCA()

	# Create pipeline: pipeline
	#pipeline = make_pipeline(scaler, pca)

	# Fit the pipeline to 'samples'
	pca.fit(samples)

	# Plot the explained variances
	features = range(pca.n_components_)
	plt.bar(features, pca.explained_variance_)
	plt.xlabel('PCA feature')
	plt.ylabel('variance')
	plt.xticks(features)
	plt.show()

	return

if __name__ == '__main__':
	main()
