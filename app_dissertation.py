#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def main():
	''' main function'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Replace text with integers
	integers = replace_text(dataframe)
	
	# Slice all columns that are survey questions
	sliced = integers.iloc[:, 17:95]

	# Drop non-supervisor behaviors
	dropped = drop_bx(sliced)

	# RESEARCH QUESTION 1
	# demographics = question1(dataframe)

	# RESEARCH QUESTION 2
	# question2(dataframe)
	
	# RESEARCH QUESTION 3
	demo_list = make_demographics_list()
	sup_list = make_supervision_behaviors_list()
	question3(dropped, demo_list, sup_list)


def question1(df):
	''' answers research question 1'''

	demo = df.iloc[:, 17:32]
	print(demo)


def question2(df):
	''' answers research question 2'''

	#behaviors.dropna(axis=1, inplace=True)

	# Run ANOVA
	data = [behaviors[col].dropna() for col in behaviors]
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

	os.chdir('./graphs')

	for demo in demo_lst:
		for bx in bx_lst:
			df.boxplot(column=bx, by=demo)
			_ = plt.xticks(rotation=90)
			_ = plt.grid(b=None)
			_ = plt.title(demo)
			_ = plt.xlabel(bx)
			_ = plt.ylabel('responses')
			_ = plt.ylim(1, 5)
			_ = plt.suptitle('')
			_ = plt.tight_layout()
			_ = plt.savefig(demo+'-'+bx+'.png')
			_ = plt.close()
			#_ = plt.show()

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

	demog = ['Area of study - Selected Choice', \
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


if __name__ == '__main__':
	main()
