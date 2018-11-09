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
	question2add(integers, supervision_list)
	

def question2add(df, sup_list):
	''' Changes column labels from supervision behavior to the category it falls in'''
 
	os.chdir('./Q2_addendum_graphs')

	# SUPERVISING WITHIN YOUR SCOPE
	group1 = (pd.concat([df['Literature for new competency area'], \
                         df['Supervisory study groups'], \
                         df['Review literature'], \
                         df['Attend conferences'], \
                         df['Participate in peer review'], \
                         df['Seek mentorship'], \
                         df['Supervisory study groups']], ignore_index=True)).tolist()
 
	# SUPERVISORY VOLUME

	group2 = (pd.concat([df['Arrive on time'], \
                         df['60% fieldwork hours'], \
                         df['Schedule contacts']], ignore_index=True)).tolist()
 
	# SUPERVISORY DELEGATION
	group3 = (pd.concat([df['Confirm required skill set'], \
                         df['Practice skill set']], ignore_index=True)).tolist()
 
	# DESIGNING EFFECTIVE TRAINING  
	group4 = (pd.concat([df['Group supervision'], \
                         df['Create group activities'], \
                         df['Include ethics'], \
                         df['Discuss how to give feedback']], ignore_index=True)).tolist()
 
	# COMMUNICATION OF SUPERVISION CONDITIONS
	group5 = (pd.concat([df['Send agenda']], ignore_index=True)).tolist()

	# PROVIDING FEEDBACK TO SUPERVISEES
	group6 = (pd.concat([df['Observe body language'], \
                         df['Maintain positive rapport']], ignore_index=True)).tolist()
 
	# EVALUATING THE EFFECTS OF SUPERVISION
	group7 = (pd.concat([df['Self-assess interpersonal skills']], ignore_index=True)).tolist()

 
	_ = plt.boxplot([group1, group2, group3, group4, group5, group6, group7])
	_ = plt.suptitle('Responses by Supervision Category')
	_ = plt.title('p-value=')
	_ = plt.xlabel('Supervision categories')
	_ = plt.ylabel('responses')
	_ = plt.savefig('supervision_categories.png')
	_ = plt.show()

	os.chdir('..')

	return

	
if __name__ == '__main__':
	main()
