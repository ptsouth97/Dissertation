#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import clean_data
from scipy.cluster.hierarchy import linkage, dendrogram


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

	# Drop responses that did not meet BACB supervision requirements
	yes = finished.drop(finished[finished['BACB requirements'] == 'No'].index)

	# Combine columns with text-based option
	combined = clean_data.combine_columns(yes)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)
	
	# Make list of demographics
	d_list = clean_data.make_demographics_list()

	sup_list = clean_data.make_supervision_behaviors_list()

	# HIERARCHICAL CLUSTERING
	cluster(metadata, d_list, sup_list)
	plt.close()
	

def cluster(df, demo_list, bx_list):
	''' Make a hierarchical clustering'''

	# Build dataframe to hold knn scores
	scores = pd.DataFrame(index=demo_list, columns=['score', 'k'])

	for demo in demo_list:
	
		# Count number of occurences
		counts = df[demo].value_counts()
    
		# Find occurences where counts are less than 2
		less_than_2 = counts[counts<2]
    
		# Remove rows from df with demo counts < 2
		for item in less_than_2.index:
			df = df.drop(df[df[demo] == item].index)	

		# Create the target array with categorical data
		y = df[demo].astype('category')
		print(y)
		# Create feature array with float (int64) data
		X = df[bx_list]
		X = X.drop(['Credentialing requirements', \
                    'Training and supervision', \
                    'Participate in peer review'], axis=1)
	
		X = X.astype('int64')

		# Calculate the linkage: mergings
		mergings = linkage(X, method='complete')

		# Plot the dendrogram, using varieties as labels
		dendrogram(mergings,
                   labels=y,
                   leaf_rotation=90,
                   leaf_font_size=6,
	              )
		plt.title(demo)
		plt.show()
		plt.close()
	
	return
	
	





if __name__ == '__main__':
	main()
