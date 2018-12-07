#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import clean_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def main():
	''' main function'''

	q5_prep()


def q5_prep():
	''' prepare data for question 1'''

	# Move up one directory level
	os.chdir('..')

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])
	
	# Move back into Q5 folder
	os.chdir('./Q5')

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

	# RESEARCH QUESTION 5
	knn_scores = question5(metadata, d_list, sup_list)
	knn_scores.to_csv('Q5 knn scores.csv')


def question5(df, demo_list, bx_list):
	''' BONUS: How accurately can a basic knn model classify the set of survey responses by demographic?'''

	# Build dataframe to hold knn scores
	scores = pd.DataFrame(index=demo_list)

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

		# Encode the target array
		y = pd.get_dummies(y)

		# Create feature array with float (int64) data
		X = df[bx_list]
		X = X.drop(['Outside training area - credentialing requirements', \
                    'Outside training area - training and supervision', \
                    'Participate in peer review'], axis=1)
	
		X = X.astype('int64')

		# Split into training and test set
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42, stratify=y)

		# Create a k-NN classifier with 7 neighbors: knn
		knn = KNeighborsClassifier(n_neighbors=7)

		# Fit the classifier to the training data
		knn.fit(X_train, y_train)

		# Print the accuracy
		ks = knn.score(X_test, y_test)
		print(ks)

		# Assign knn score to dataframe position
		scores.loc[demo, 'score'] = round(ks, 3)
		
	# Arrange df in descending order
	scores.sort_values(by='score', ascending=False, inplace=True)

	# Make a boxplot
	_ = scores.plot.bar(color='gray', legend=False)
	_ = plt.title('k-NN Scores by Demographic')
	_ = plt.xlabel('Demographic')
	_ = plt.ylabel('k-NN accuracy')
	_ = plt.tight_layout()
	_ = plt.savefig('k-NN Scores.png')
	_ = plt.show()

	return scores


if __name__ == '__main__':
	main()
