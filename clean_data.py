#!/usr/bin/python3

import pandas as pd
import numpy as np


def main():
	''' Main function for testing'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Combine columns with text-based option
	combined = combine_columns(dataframe)

	# Replace text with integers
	integers = text_to_int(combined)
	
	# Drop all the survey metadata
	metadata = drop_metadata(integers)

	# Make a list of all the demographics
	demo_list = make_demographics_list()

	# Drop demographics
	no_demo = drop_demographics(metadata, demo_list)

	# Combine columns with text-based option
	combined = combine_columns()

	# Drop non-supervisor behaviors
	dropped = drop_bx(metadata)
	print(dropped)


def text_to_int(df):
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
             'Job classification', \
             'Place of employment', \
             'State', \
             'Supervision mode', \
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


def drop_demographics(df):
	'''drops the demographics columns'''

	df.drop(['BACB requirements', \
             'Years certified', \
             'Years supervisor', \
             'Area of study - Selected Choice', \
             'Area of study - Other - Text', \
             'Job classification - Selected Choice', \
             'Job classification - Other - Text', \
             'Place of employment - Selected Choice', \
             'Place of employment - Other - Text', \
             'State', \
             'Supervision mode - Selected Choice', \
             'Supervision mode - Other - Text', \
             'Supervision format', \
             'Supervision training - Selected Choice', \
             'Supervision training - Other - Text', \
             '100% fieldwork candidates', \
             '100% fieldwork pass rate', \
             'Discontinued fieldwork', \
             'Supervision resources - Selected Choice', \
             'Supervision resources - Other - Text', \
             'Q4 - Topics'], \
             inplace=True, axis=1)

	return df


def get_question4_data():
	''' returns list of needed column headings for research question 4'''

	q4 = ['100% fieldwork candidates', \
          '100% fieldwork pass rate', \
          'Discontinued fieldwork', \
          'pass rate']


	return q4

	
if __name__ == '__main__':
	main()
