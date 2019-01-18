#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data
import sklearn
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
import n1_clusters, n4_FindNumberClusters


def main():
	''' Loads and cleans data for analysis'''

	factor_analysis()


def factor_analysis():
	''' Prepares data for factor analysis'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Move back to Q2 directory
	os.chdir('./Q2_results')

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(finished)

	# Drop non-supervisor behaviors
	dropped_bx = clean_data.drop_bx(metadata)
	
	# Drop demographics
	dropped_demo = clean_data.drop_demographics(dropped_bx)

	# Replace text with integers
	integers = clean_data.text_to_int(dropped_demo)

	# Make supervison behaviors list
	sup_list = clean_data.make_supervision_behaviors_list()

	# Fill NA values with average
	for sup in sup_list:
		integers[sup].fillna(integers[sup].mean(), inplace=True)
	
	# Standardize values NOT NECESSARY
	#df = (integers-integers.mean())/integers.std(ddof=0)

	# Convert dataframe into numpy array
	data = np.array(integers)

	# Show cluster example
	n1_clusters.plot_clusters(data)
	
	# KMEANS CLUSTERING
	n4_FindNumberClusters.inertia_plot(data)

	'''
	# FACTOR ANALYSIS
	factor = FactorAnalysis(n_components=4).fit(df)
	results = pd.DataFrame(factor.components_, columns=sup_list)
	results.to_csv('FactorAnalysisResults.csv')
	print(results)

	covariance = pd.DataFrame(factor.get_covariance(), columns=sup_list, index=sup_list)
	covariance.to_csv('Covariance.csv')
	print(covariance)

	precision = pd.DataFrame(factor.get_precision(), columns=sup_list)
	print(precision)
	'''
	
	'''
	# PRINCIPAL COMPONENT ANALYSIS
	pca = PCA()
	pca.fit(df)
	PCA(copy=True, iterated_power='auto', n_components=None, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
	print(pca.get_covariance())
	'''
	
	return


if __name__ == '__main__':
	main()
