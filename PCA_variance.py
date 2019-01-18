#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import clean_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
	''' main function'''

	cluster_prep()


def cluster_prep():
	''' prepare data for question 1'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(finished)

	# Drop non-supervisor behaviors
	dropped_bx = clean_data.drop_bx(metadata)
	
	# Drop demographics
	dropped_demo = clean_data.drop_demographics(dropped_bx)

	# Make supervison behaviors list
	sup_list = clean_data.make_supervision_behaviors_list()

	# Replace text with integers
	integers = clean_data.text_to_int(dropped_demo)

	# Fill NA values with average
	for sup in sup_list:
		integers[sup].fillna(integers[sup].mean(), inplace=True)

	# Convert dataframe into numpy array
	data = np.array(integers)

	# PCA variance
	PCA_variance(integers)
	

def PCA_variance(samples):
	''' Makes a plot of the variances of the PCA features to help find the instrinsic dimension'''

	# Create scaler: scaler
	scaler = StandardScaler()

	# Create a PCA instance: pca
	pca = PCA()

	# Create pipeline: pipeline
	pipeline = make_pipeline(scaler, pca)

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
