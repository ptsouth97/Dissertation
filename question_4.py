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
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)

	# Combine columns with text-based option
	combined = clean_data.combine_columns(finished)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)
	
	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)

	# Drop non-supervisor behaviors
	dropped = clean_data.drop_bx(metadata)

	# Fill in zeroes for missing values
	zeroed = clean_data.zeroes(dropped)

	# Calculate necessary values for question 4
	zeroed['pass rate'] = zeroed['100% fieldwork pass rate'] / \
                          (zeroed['100% fieldwork candidates'] - zeroed['Discontinued fieldwork'])

	no_inf = zeroed.replace([np.inf, -np.inf], np.nan).dropna(subset=['pass rate'], how='all')

	# Filter out responses with pass rates greater than 100% (reponder misread or mis-answered the question)
	filtered = no_inf[no_inf['pass rate'] <= 1]

	# RESEARCH QUESTION 4
	q4_list = clean_data.get_question4_data()
	sup_list = clean_data.make_supervision_behaviors_list()
	p_val = question4(filtered, q4_list, sup_list)
	p_val.to_csv('Q4 p-value table.csv')


def question4(df, q4_lst, bx_lst):
	''' Answers research question 4'''

	# Change folder for graphs
	os.chdir('./Q4_graphs')

	# Initialize dataframe to hold Spearman correlation rho and p-values
	results = pd.DataFrame(index=bx_lst, columns=['rho', 'p-val'])

	for bx in bx_lst:

		# Slice a sample df for analysis
		sample = df.loc[:,[bx, 'pass rate']]

		# Get rid of NAs
		sample.dropna(inplace=True)

		# Spearman correlation calculates rho and p-value and adds to dataframe
		r, p = calculate_spearman(sample[bx], sample['pass rate'])
		r = str(round(r, 3))
		p = str(round(p, 3))
		results.loc[bx, 'rho'] = r
		results.loc[bx, 'p-val'] = p

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
		_ = plt.title('rho=' + r + ', p=' + p)
		_ = plt.xlabel('responses')
		_ = plt.ylabel('pass rate')
		_ = plt.xticks(np.arange(1, 5.1, 1))
		_ = plt.yticks(np.arange(0, 1.01, 0.2))
		_ = plt.ylim(0, 1.1)
		#_ = plt.tight_layout()
		#_ = plt.margins(0.02)
		_ = plt.savefig(bx+'.png')
		_ = plt.close()

	os.chdir('..')

	return results


def calculate_spearman(x, y):
	''' Calculates p-value from Spearman correlation'''

	rho, pval = stats.spearmanr(x, y)

	return rho, pval


if __name__ == '__main__':
	main()
