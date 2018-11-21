#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data


def main():
	''' Loads and cleans data for analysis'''

	q2_add_prep()


def q2_add_prep():
	''' Prepares data for research question 3'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(dataframe)

	# Drop non-supervisor behaviors
	dropped_bx = clean_data.drop_bx(metadata)
	
	# Drop demographics
	dropped_demo = clean_data.drop_demographics(dropped_bx)

	# Replace text with integers
	integers = clean_data.text_to_int(dropped_demo)

	# Make list of supervision behaviors
	supervision_list = clean_data.make_supervision_behaviors_list()

	# RESEARCH QUESTION 2 ADDENUM
	question2add(integers)
	

def question2add(df):
	''' Changes column labels from supervision behavior to the category it falls in'''
 
	os.chdir('./Q2_addendum_graphs')

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
                        df['Discourage distractions']], ignore_index=True)
	
	group8.dropna(inplace=True)


	columns = ['Supervising within your scope', 'Supervisory volume', 'Supervisory delegation', \
               'Designing effective training', 'Communication of supervision conditions', \
               'Providing feedback to supervisees', 'Evaluating the effects of supervision', 'Misc']

	
	df = pd.concat([group1, group2, group3, group4, group5, group6, group7, group8], axis=1, ignore_index=True)
	df.columns = columns

	# Run ANOVA
	data = [df[col].dropna() for col in df]
	f, p = stats.f_oneway(*data)

	# Account for extremely small p-value
	if p < 0.001:
		p = '<.001'

	else:
		p = '='+str(format(p, '.3e'))
	
	# Get the number of lists in 'data' then subtract one
	n_lists = len(data)
	dfn = n_lists - 1

	# Sum up the number of data points in each list then subtract the number of lists
	dfd = 0

	for col in data:
		dfd += len(col)

	dfd = dfd - n_lists

	# Calculate the critical F value
	Fcrit = stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd)
	
	# Make the boxplot
	bp = plt.boxplot([group1, group2, group3, group4, group5, group6, group7, group8], \
                      labels=columns, \
                      patch_artist=True)

	# Chance color of boxes
	for box in bp['boxes']:
 		box.set(facecolor = 'gray')
 
 	# Change color of median line
	for median in bp['medians']:
		median.set(color = 'black')

	_ = plt.title('Responses by Supervision Category, F('+str(dfn)+', '+str(dfd)+ \
                  ')='+str(round(f, 3))+' (F critical='+str(round(Fcrit, 3))+'), p='+p)

	_ = plt.ylim(1,5)
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
