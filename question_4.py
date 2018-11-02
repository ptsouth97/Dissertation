#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data
from sklearn.linear_model import LinearRegression


def main():
	''' main function'''

	q4_prep()


def q4_prep():
	''' Prepares data for research question 4'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Combine columns with text-based option
	combined = clean_data.combine_columns(dataframe)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)
	
	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)

	# Drop non-supervisor behaviors
	dropped = clean_data.drop_bx(metadata)

	# Calculate necessary values for question 4
	dropped['pass rate'] = dropped['100% fieldwork pass rate'] / \
                          (dropped['100% fieldwork candidates'] - dropped['Discontinued fieldwork'])

	# RESEARCH QUESTION 4
	q4_list = clean_data.get_question4_data()
	sup_list = clean_data.make_supervision_behaviors_list()
	question4(dropped, q4_list, sup_list)


def question4(df, q4_lst, bx_lst):
	''' Answers research question 4'''

	# Change folder for graphs
	os.chdir('./Q4_graphs')

	#df = [df[col].dropna() for col in df]
	#print(df)

	# Initialize list to hold p-values
	p_values = []

	for bx in bx_lst:

		# Slice a sample df for analysis
		sample = df.loc[:,[bx, 'pass rate']]

		# Get rid of NAs
		sample.dropna(inplace=True)

		# Spearman correlation calculates p-value and appends to list
		p = calculate_spearman(df[bx], df['pass rate'])
		p_values.append(p)

		# Linear regression
		reg = LinearRegression()
		prediction_space = np.linspace(1, 5).reshape(-1,1)
		X = sample[bx].values.reshape(-1,1)
		y = sample['pass rate'].values.reshape(-1,1)
		reg.fit(X, y)
		y_pred = reg.predict(prediction_space)

		# Calculate r^2 for the regression
		r2 = reg.score(X, y)

		# Make a scatter plot and line of best fit
		_ = plt.plot(prediction_space, y_pred, color='black', linewidth=1)
		_ = plt.scatter(sample[bx], sample['pass rate'], c='k', s=6, clip_on=False)
		#_ = plt.grid(b=None, axis='both')
		_ = plt.suptitle(bx)
		_ = plt.title('r^2='+str(r2)+', p='+str(p))
		_ = plt.xlabel('responses')
		_ = plt.ylabel('pass rate')
		#_ = plt.annotate('r^2='+str(r2), xy=(1.5, 0.25))
		_ = plt.xticks(np.arange(1, 5.1, 1))
		_ = plt.yticks(np.arange(0, 1.1, 0.2))
		#_ = plt.tight_layout()
		#_ = plt.margins(0.02)
		_ = plt.savefig(bx+'.png')
		_ = plt.close()


	#print('p_values:')
	#print(p_values)
	
	os.chdir('..')

	return


def calculate_spearman(x, y):
	''' Calculates p-value from Spearman correlation'''

	rho, pval = stats.spearmanr(x, y)

	return pval


if __name__ == '__main__':
	main()
