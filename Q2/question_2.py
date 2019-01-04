#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data


def main():
	''' Loads and cleans data for analysis'''

	q2_prep()


def q2_prep():
	''' Prepares data for research question 3'''

	# Move up one directory
	os.chdir('..')

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Move back to Q2 directory
	os.chdir('./Q2')

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

	# RESEARCH QUESTION 2
	statistics = question2(integers, sup_list, 100)
	
	# RESEARCH QUESTION 2 ... FIGURE OUT WHERE SIGNIFICANCE BEGINS
	sorted_bx = statistics.index.tolist()

	print(sorted_bx)	
	for i in range(0,22):

		#Drop the first and last column
		integers = integers.drop(columns=[sorted_bx[0], sorted_bx[-1]])
		
		#Drop the first item in list (highest average)
		sorted_bx.pop(0)

		# Drop the last item in the list (lowest average)
		sorted_bx.pop(-1)

		question2(integers, sup_list, i)
		
		
	

def question2(df, bx_lst, n):
	''' answers research question 2'''

	# change directory
	os.chdir('./Q2_graphs')	

	# Run ANOVA
	data = [df[col].dropna() for col in df]
	f, p = stats.f_oneway(*data)

	# Account for extremely small p-value
	if p < 0.001:
		p = '<.001'

	else:
		p = '='+str(format(p, '.3e'))

	# Get the number of lists in 'data' then subtract one to get degrees of freedom numerator (dfn)
	n_lists = len(data)
	dfn = n_lists - 1

	# Make lists to hold averages, median, and standard deviation
	avg = []
	med = []
	stdev = []

	# Sum up the number of data points in each list then subtract the number of lists to get
	# degrees of freedom denominator (dfd)
	dfd = 0
	for each_list in data:
		dfd += len(each_list)
		avg.append(round(np.mean(each_list), 3))
		med.append(round(np.median(each_list), 3))
		stdev.append(round(np.std(each_list), 3))


	# Convert the lists to dataframe
	statistics = pd.DataFrame({'mean':avg, 'median':med, 'std':stdev}, index=df.columns)
	statistics = statistics.sort_values(by='mean', ascending=False)
	print(statistics)
	statistics.to_csv('stats.csv')

	dfd = dfd - n_lists

	# Calculate the critical F value
	Fcrit = stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd)

	# Sort dataframe in descending order
	df = df.reindex(df.mean().sort_values(ascending=False).index, axis=1)

	# Make a boxplot	
	color = dict(boxes='gray', whiskers='black', medians='black', caps='black')
	meanpointprops = dict(marker='s', markeredgecolor='black', markerfacecolor='black')
	_ = df.plot.box(color=color, patch_artist=True, meanprops=meanpointprops, showmeans=True)
	_ = plt.yticks(np.arange(1, 5+1, step=1))
	_ = plt.xticks(rotation=90)
	_ = plt.xlabel('Individual behaviors')
	_ = plt.ylabel('Survey response')
	_ = plt.title('Supervisory Behaviors, F('+str(dfn)+', '+str(dfd)+')=' \
                  +str(round(f, 3))+'(F critical='+str(round(Fcrit, 3))+'), p'+p)


	# Resize to larger window for bigger graph
	fig = plt.gcf()
	fig.set_size_inches(12, 10)
	_ = plt.tight_layout()

	if n == 100:
		_ = plt.savefig('SupervisoryBehaviorsBoxplot.png')

	elif n != 100:
		_ = plt.savefig('SupervisoryBehaviorsBoxplot_' +str(n)+ '.png')

	_ = plt.show()
	_ = plt.close()

	os.chdir('..')
	
	return statistics

	
if __name__ == '__main__':
	main()
