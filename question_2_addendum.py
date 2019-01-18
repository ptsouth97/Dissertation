#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data
import question_2


def main():
	''' Loads and cleans data for analysis'''

	q2_add_prep()


def q2_add_prep():
	''' Prepares data for research question 3'''
	
	# Move up one directory level
	os.chdir('..')

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Move back into question 2 directory
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

	# Make list of supervision behaviors
	supervision_list = clean_data.make_supervision_behaviors_list()

	# Make list of demographics
	demo_list = clean_data.make_demographics_list()

	# RESEARCH QUESTION 2 ADDENUM
	df = question2add(integers)

	# SEND BACK TO ORIGINAL QUESTION
	#statistics, avg = question_2.question2(df, 100)
		
	#question_2.q2_narrow(df, statistics, avg)
	

def question2add(df):
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

	# Run ANOVA
	data = [df[col].dropna() for col in df]
	f, p = stats.f_oneway(*data)
	print(stats.f_oneway(*data)

	# Account for extremely small p-value
	if p < 0.001:
		p = '<.001'

	else:
		p = '='+str(format(p, '.3e'))
	
	# Get the number of lists in 'data' then subtract one
	n_lists = len(data)
	dfn = n_lists - 1

	# Make lists to hold averages, median, and standard deviation
	avg = []
	med = []
	stdev = []

	# Sum up the number of data points in each list then subtract the number of lists
	dfd = 0

	for col in data:
		dfd += len(col)
		avg.append(round(np.mean(col), 3))
		med.append(round(np.median(col), 3))
		stdev.append(round(np.std(col), 3))

	# Convert the lists to dataframe
	statistics = pd.DataFrame({'mean':avg, 'median':med, 'std':stdev}, index=df.columns)
	statistics = statistics.sort_values(by='mean', ascending=False)
	print(statistics)

	# Find the average of the averages
	tot_avg = statistics['mean'].mean()

	# Save the stats to a .csv file
	statistics.to_csv('addendum_stats.csv')

	dfd = dfd - n_lists

	# Calculate the critical F value
	Fcrit = stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd)

	# Sort dataframe in descending order
	df = df.reindex(df.mean().sort_values(ascending=False).index, axis=1)
	
	# Make the boxplot
	color = dict(boxes='gray', whiskers='black', medians='black', caps='black')
	meanpointprops = dict(marker='s', markeredgecolor='black', markerfacecolor='black')

	bp = df.plot.box(color=color, patch_artist=True, meanprops=meanpointprops, showmeans=True)

	_ = plt.title('Responses by Supervision Category, F('+str(dfn)+', '+str(dfd)+ \
                  ')='+str(round(f, 3))+' (F critical='+str(round(Fcrit, 3))+'), p='+p)

	new_ticks = ['Almost never (1)', 'Rarely (2)', 'Sometimes (3)', 'Usually (4)', 'Almost always (5)']

	_ = plt.yticks(np.arange(1, 5+1, step=1), new_ticks)
	_ = plt.ylim(0.9, 5.1)
	_ = plt.xlabel('Supervision categories')
	_ = plt.ylabel('responses')
	_ = plt.xticks(rotation=45)

	fig = plt.gcf()
	fig.set_size_inches(12, 10)

	_ = plt.tight_layout()
	_ = plt.savefig('supervision_categories.png')
	_ = plt.show()
	_ = plt.close()

	os.chdir('..')
	
	return
	
	
if __name__ == '__main__':
	main()
