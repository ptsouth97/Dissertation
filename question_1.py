#!/usr/bin/python3

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def main():
	''' main function'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Combine columns with text-based option
	combined = combine_columns(dataframe)

	# Replace text with integers
	integers = replace_text(combined)
	
	# Drop all the survey metadata
	metadata = drop_metadata(integers)

	# Drop non-supervisor behaviors
	dropped = drop_bx(metadata)
	print(dropped)

	# RESEARCH QUESTION 1
	# demographics = question1(dataframe)

	# RESEARCH QUESTION 2
	question2(dropped)
	
	# RESEARCH QUESTION 3
	#demo_list = make_demographics_list()
	#sup_list = make_supervision_behaviors_list()
	#question3(dropped, demo_list, sup_list)


def question1(df):
	''' answers research question 1'''

	demo = df.iloc[:, 17:32]
	print(demo)


def question2(df):
	''' answers research question 2'''

	# Run ANOVA
	data = [df[col].dropna() for col in df]
	f, p = stats.f_oneway(*data)
	
	# Make a boxplot	
	_ = behaviors.boxplot()
	_ = plt.xticks(rotation=90)
	_ = plt.xlabel('Individual behaviors')
	_ = plt.ylabel('Survey response')
	_ = plt.title('Supervisory Behavior Boxplot')
	_ = plt.annotate('p='+str(p), xy=(30,1))
	_ = plt.grid(b=None)
	#_ = plt.tight_layout()
	_ = plt.savefig('SupervisoryBehaviorsBoxplot.png')
	_ = plt.show()

	return

	
def question3(df, demo_lst, bx_lst):
	''' Answers research question 3'''

	# Change folder for graphs
	os.chdir('./Q3_graphs')

	# Initialize list to hold p-values
	p_values = []

	# Loop through each demographic
	for demo in demo_lst:

		# For each demographic, go through the behaviors			
		for bx in bx_lst:
			grouped = df.groupby(demo)
			groupby_to_df = grouped.describe().squeeze()
			names = groupby_to_df.index.tolist()
			length = len(names) + 1
			positions = list(range(1,length))
			grouped_list = list(grouped[bx])
			grouped_df = pd.DataFrame.from_items(grouped_list)
	
			n_cols = len(grouped_df.columns)
			col_list = []			

			for col in range(0, n_cols):
				column=grouped_df.iloc[:,col].dropna().tolist()
				col_list.append(column)

			if len(col_list) == 2:
				f, p = stats.f_oneway(col_list[0], col_list[1])

			elif len(col_list) == 3:
				f, p = stats.f_oneway(col_list[0], col_list[1], col_list[2])

			elif len(col_list) == 4:
				f, p = stats.f_oneway(col_list[0], col_list[1], col_list[2], col_list[3])

			p_values.append(p)

			_ = plt.boxplot(col_list)
			_ = plt.xticks(positions, names, rotation=45)
			_ = plt.yticks(np.arange(1, 5, step=1))
			_ = plt.grid(b=None)
			_ = plt.title(bx)
			_ = plt.xlabel(demo)
			_ = plt.ylabel('responses')
			_ = plt.ylim(1, 5)
			_ = plt.annotate('p='+str(p), xy=(0.6, 1.5))
			_ = plt.tight_layout()
			_ = plt.savefig(demo+'-'+bx+'.png')
			_ = plt.close()


	print('p_values:')
	print(p_values)

	os.chdir('..')

	return


def replace_text(df):
	''' Changes text to integers'''

	df.replace('Almost never (0-20%)', 1, inplace=True)
	df.replace('Rarely (21-40%)', 2, inplace=True)
	df.replace('Sometimes (41-60%)', 3, inplace=True)
	df.replace('Usually (61-80%)', 4, inplace=True)
	df.replace('Almost always (81-100%)', 5, inplace=True)

	return df
	

def drop_bx(df):
	''' drop non-supervisor behaviors'''

	df.drop(['Outside training area (yes/no)', \
             'Number of candidates', \
             'Past 12 months candidates', \
             'Allotted hours', \
             'Scheduled hours', \
             'Number of clients', \
             'Who dictates caseload', \
             'Supervise RBTs (yes/no)', \
             'RBT supervision %', \
             'Peer review opportunity', \
             'Supervision fieldwork protocol source - Other - Text', \
             'Supervision fieldwork protocol source - Selected Choice'], \
             inplace=True, axis=1)

	return df


def make_demographics_list():
	''' Create list of demographics'''

	demog = ['Area of study', \
             'Job classification - Selected Choice', \
             'Place of employment - Selected Choice', \
             'State', \
             'Supervision mode - Selected Choice', \
             'Supervision format']

	return demog


def make_supervision_behaviors_list():
	''' Create the list of supervision behaviors'''

	sup = ['Literature for new competency area', \
           'Professional groups', \
           'Outside training area - credentialing requirements', \
           'Supervision schedule']

	return sup


def combine_columns(df):
	''' Combines columns that have text-based options'''

	df['Area of study'] = \
          (df['Area of study - Selected Choice'].fillna('') + \
           df['Area of study - Other - Text'].fillna('')).str.strip()

	df['Job classification'] = \
          (df['Job classification - Selected Choice'].fillna('') + \
           df['Job classification - Other - Text'].fillna('')).str.strip()

	df['Place of employment'] = \
          (df['Place of employment - Selected Choice'].fillna('') + \
           df['Place of employment - Other - Text'].fillna('')).str.strip()

	df['Supervision mode'] = \
          (df['Supervision mode - Selected Choice'].fillna('') + \
           df['Supervision mode - Other - Text'].fillna('')).str.strip()

	df['Supervision training'] = \
          (df['Supervision training - Selected Choice'].fillna('') + \
           df['Supervision training - Other - Text'].fillna('')).str.strip()

	df['Supervision resources'] = \
          (df['Supervision resources - Selected Choice'].fillna('') + \
           df['Supervision resources - Other - Text'].fillna('')).str.strip()

	df['Supervision fieldwork protocol source'] = \
          (df['Supervision fieldwork protocol source - Selected Choice'].fillna('') + \
           df['Supervision fieldwork protocol source - Other - Text'].fillna('')).str.strip()

	return df


def drop_metadata(df):
	''' drops the survey metadata'''

	df.drop(['Start Date', \
             'End Date', \
             'Response Type', \
             'IP Address', \
             'Progress', \
             'Duration (in seconds)', \
             'Finished', \
             'Recorded Date', \
             'Response ID', \
             'Recipient Last Name', \
             'Recipient First Name', \
             'Recipient Email', \
             'External Data Reference', \
             'Location Latitude', \
	         'Location Longitude', \
             'Distribution Channel', \
             'User Language'], \
             inplace=True, axis=1)

	return df

if __name__ == '__main__':
	main()
