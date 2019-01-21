#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import clean_data


def main():
	''' Loads and cleans data for analysis'''

	q2_prep()


def q2_prep():
	''' Prepares data for research question 3'''

	# Move up one directory
	# os.chdir('..')

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

	# RESEARCH QUESTION 2
	statistics, avg  = question2(integers, 100)

	#q2_narrow(integers, statistics, avg)
	

def q2_narrow(integers, statistics, avg):
	''' FIGURE OUT WHERE SIGNIFICANCE BEGINS'''

	sorted_bx = statistics.index.tolist()

	# total number of behaviors
	total = len(sorted_bx)

	# number of behaviors greater than the average
	statistics_array = np.array(statistics['mean'])
	less_than = (statistics_array < avg).sum()
	print('LESS THAN ' +str(less_than))

	num_to_drop = total - (less_than * 2)

	# Slice the columns to drop from the list of behaviors
	columns_to_drop = sorted_bx[0:num_to_drop]

	# Use the list from previous step to drop the columns that won't be included
	integers = integers.drop(columns=columns_to_drop)

	# Also remove the corresponding behaviors from the list
	for i in range(0, len(columns_to_drop)):
		sorted_bx.pop(0)

	# Figure out how many times to drop columns
	num_to_process = int(len(sorted_bx) / 2 - 1)

	for i in range(0, num_to_process):

		#Drop the columns corresponding to the first and last items of the behavior list sorted by average
		integers = integers.drop(columns=[sorted_bx[0], sorted_bx[-1]])
				
		#Drop the first item in list (highest average)
		sorted_bx.pop(0)

		# Drop the last item in the list (lowest average)
		sorted_bx.pop(-1)

		question2(integers, i)
	

def question2(df, n):
	''' answers research question 2'''
	
	# change directory
	os.chdir('./Q2_graphs')	

	# Run ANOVA using Scipy stats
	data = [df[col].dropna() for col in df]
	f, p = stats.f_oneway(*data)

	# Initialize new dataframe to hold results for tukey test
	tukey_df = pd.DataFrame()

	for col_name in df.columns:
		temp = df.loc[:, [col_name]]
		temp['identity'] = col_name
		temp = temp.rename(columns={col_name:'score'})
		tukey_df = tukey_df.append(temp)

	print('HERE IS THE FINAL DF FOR TUKEY')
	tukey_df = tukey_df.dropna()
	tukey_df = tukey_df.reset_index(drop=True)
	print(tukey_df)

	posthoc = pairwise_tukeyhsd(tukey_df['score'], tukey_df['identity'])
	tukey_results = pd.DataFrame(data=posthoc._results_table.data[1:],
                              columns=posthoc._results_table.data[0])
	tukey_results.to_csv('Q2-Tukey.csv', index=False)
	print(posthoc)
	
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
	#print(statistics)

	# Find the average of the averages
	tot_avg = statistics['mean'].mean()

	# Save the stats to a .csv file
	statistics.to_csv('stats.csv')

	dfd = dfd - n_lists

	# Calculate the critical F value
	Fcrit = stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd)

	# Calculate Sum of Squares Between groups
	SSbetween = 10.533
 
	# Calculate Sum of Squares Within groups
	SSwithin = 12.8

	# Calculate Mean Square Between groups
	MSbetween = 5.267

	# Create result table
	result_table = pd.DataFrame(index=['Between Groups', 'Within Groups', 'Total'], \
                              columns=['Sum of Squares', 'df', 'Mean Square', 'F', 'Sig.'])

	result_table.loc['Between Groups', 'Sum of Squares'] = SSbetween
	result_table.loc['Between Groups', 'df'] = dfn
	result_table.loc['Between Groups', 'Mean Square'] = MSbetween
	result_table.loc['Between Groups', 'F'] = f
	result_table.loc['Between Groups', 'Sig.'] = p


	print(result_table)
	result_table.to_csv('result_table.csv')

	fig, ax = plt.subplots()

	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')

	ax.table(cellText=result_table.values, colLabels=result_table.columns, loc='center')
	#fig.tight_layout()
	fig.savefig('results.png')
	plt.show()
	plt.close()

#########################################################
	# Sort dataframe in descending order
	df = df.reindex(df.mean().sort_values(ascending=False).index, axis=1)

	# Make a boxplot	
	color = dict(boxes='gray', whiskers='black', medians='black', caps='black')
	meanpointprops = dict(marker='s', markeredgecolor='black', markerfacecolor='black')

	_ = df.plot.box(color=color, patch_artist=True, meanprops=meanpointprops, showmeans=True)

	new_yticks = ['Almost never (1)', 'Rarely (2)', 'Sometimes (3)', 'Usually (4)', 'Almost always (5)']
	
	_ = plt.axhline(y=tot_avg, linestyle='--', color='black')
	_ = plt.yticks(np.arange(1, 5+1, step=1), new_yticks)
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
	
	return statistics, tot_avg


def group_behaviors(df):
	''' Changes column labels from supervision behavior to the category it falls in'''
 
	os.chdir('./Q2_graphs')

	# SUPERVISING WITHIN YOUR SCOPE
	group1 = pd.concat([df['Literature for new competency area'], \
                        df['Supervisory study groups'], \
                        df['Professional groups'], \
                        df['Outside training area - credentialing requirements'], \
                        df['Outside training area - training and supervision'], \
                        df['Supervisory study groups']], ignore_index=True)

	group1.dropna(inplace=True)
 
	# SUPERVISORY VOLUME
	group2 = pd.concat([df['Arrive on time'], \
                        df['60% fieldwork hours'], \
                        df['Supervision schedule'], \
                        df['Schedule contacts']], ignore_index=True)

	group2.dropna(inplace=True)
 
	# SUPERVISORY DELEGATION
	group3 = pd.concat([df['Confirm required skill set'], \
                        df['Practice skill set']], ignore_index=True)

	group3.dropna(inplace=True)
 
	# DESIGNING EFFECTIVE TRAINING  
	group4 = pd.concat([df['Group supervision'], \
                        df['Create group activities'], \
                        df['Include ethics'], \
                        df['Behavior skills training'], \
                        df['Discuss how to give feedback']], ignore_index=True)

	group4.dropna(inplace=True)
 
	# COMMUNICATION OF SUPERVISION CONDITIONS
	group5 = pd.concat([df['Send agenda'], \
                        df['Performance expectations'], \
                        df['Written supervision contract'], \
                        df['Review supervision contract'], \
                        df['Supervision termination clause']], ignore_index=True)

	group5.dropna(inplace=True)

	# PROVIDING FEEDBACK TO SUPERVISEES
	group6 = pd.concat([df['Observe body language'], \
                        df['Written evaluation system'], \
                        df['Positive and corrective feedback'], \
                        df['Document feedback'], \
                        df['Immediate feedback'], \
                        df['Instructions and demonstration']], ignore_index=True)
	
	group6.dropna(inplace=True) 

	# EVALUATING THE EFFECTS OF SUPERVISION
	group7 = pd.concat([df['Self-assess interpersonal skills'], \
                        df['Supervision fidelity'], \
                        df['Evaluate client performance'], \
                        df['Evaluate supervisee performance']], ignore_index=True)

	group7.dropna(inplace=True)

	# MISC
	group8 = pd.concat([df['Peer evaluate'], \
                        df['Return communications within 48 hours'], \
                        df['Take baseline'], \
                        df['Detect barriers to supervision'], \
                        df['BST case presentation'], \
                        df['Send agenda'], \
                        df['Continue professional relationship'], \
                        df['Observe body language'], \
                        df['Maintain positive rapport'], \
                        df['Self-assess interpersonal skills'], \
                        df['Group supervision'], \
                        df['Create group activities'], \
                        df['Supervisory study groups'], \
                        df['Include ethics'], \
                        df['Arrive on time'], \
                        df['Discuss how to give feedback'], \
                        df['Schedule direct observations'], \
                        df['Schedule standing supervision appointments'], \
                        df['Review literature'], \
                        df['Meeting notes'], \
                        df['Attend conferences'], \
                        df['Participate in peer review'], \
                        df['Seek mentorship'], \
                        df['60% fieldwork hours'], \
                        df['Schedule contacts'], \
                        df['Discourage distractions']], ignore_index=True)
	
	group8.dropna(inplace=True)


	columns = ['Supervising within your scope (5.01)', \
               'Supervisory volume (5.02)', \
               'Supervisory delegation (5.03)', \
               'Designing effective training (5.04)', \
               'Communication of supervision conditions (5.05)', \
               'Providing feedback to supervisees (5.06)', \
               'Evaluating the effects of supervision (5.07)', \
               'Misc']

	
	df = pd.concat([group1, group2, group3, group4, group5, group6, group7, group8], axis=1, ignore_index=True)
	df.columns = columns

	os.chdir('..')

	return df
	
if __name__ == '__main__':
	main()
