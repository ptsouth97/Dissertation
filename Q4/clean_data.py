#!/usr/bin/python3

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


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


def combine_text(df, demo):
	''' Replaces user text in Other fields with common values'''

	#print(df[demo])

	if demo == 'Supervision mode':
		df[demo].replace(to_replace='.*[Ss]chool.*', value='Other-school', inplace=True, regex=True)
		df[demo].replace(to_replace='.*district.*', value='Other-school', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Cc]linic.*', value='Other-clinic', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Ss]tate.*', value='Other-state', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Cc]ommunity.*', value='Other-community program', inplace=True, regex=True)
		df[demo].replace(to_replace='.*No longer provide.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Work place.*', value='Other-clinic', inplace=True, regex=True)

	if demo == 'Supervision training':
		df[demo].replace(to_replace='.*[Ll]iterature.*', value='Other-literature', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Jj]ournal.*', value='Other-literature', inplace=True, regex=True)
		df[demo].replace(to_replace='.*employer.*', value='Other-company based', inplace=True, regex=True)
		df[demo].replace(to_replace='.*agency.*', value='Other-company based', inplace=True, regex=True)
		df[demo].replace(to_replace='.*training at work.*', value='Other-company based', inplace=True, regex=True)
		df[demo].replace(to_replace='.*etc.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Materials.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*BACB.*', value='Other-BACB requirements', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Mentoring.*', value='Individual mentoring', inplace=True, regex=True)
		df[demo].replace(to_replace='.*experience.*', value='Other-personal experience', inplace=True, regex=True)
		df[demo].replace(to_replace='.*BCBA.*', value='Other-BCBA requirements', inplace=True, regex=True)

	if demo == 'Supervision resources':
		df[demo].replace(to_replace='.*[Nn]one.*', value='None', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Nn]othing.*', value='None', inplace=True, regex=True)
		df[demo].replace(to_replace='.*paid.*', value='Monetary compensation', inplace=True, regex=True)
		df[demo].replace(to_replace='.*funds.*', value='Monetary compensation', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Ss]elf.*', value='Other-self', inplace=True, regex=True)
		df[demo].replace(to_replace='.*mentor ship.*', value='Other-mentorship', inplace=True, regex=True)     
		df[demo].replace(to_replace='.*meeting.*', value='Other-meetings', inplace=True, regex=True)
		df[demo].replace(to_replace='.*40 hours.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*email.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*school ID.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*unfortunately.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*organization.*', value='Administrative assistance', inplace=True, regex=True)
		df[demo].replace(to_replace='.*CE.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*conference.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Data based.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*directly for.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*receive no.*', value='None', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Interms.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*OtherNa.*', value='None', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Team-based.*', value='Other-meetings', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Other BCBAs.*', value='Other-meetings', inplace=True, regex=True)
		df[demo].replace(to_replace='.*experience.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*online training.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*web-based.*', value='Other', inplace=True, regex=True)

	if demo == 'Supervision fieldwork protocol source':
		df[demo].replace(to_replace='.*[Nn]one.*', value='None', inplace=True, regex=True) 
		df[demo].replace(to_replace='.*lit.*', value='Other-literature', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Research.*', value='Other-literature', inplace=True, regex=True)
		df[demo].replace(to_replace='.*doctoral studies.*', value='Other-literature', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Bb][Aa][Cc][Bb].*', value='Other-BACB materials', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Task list.*', value='Other-BACB materials', inplace=True, regex=True)
		df[demo].replace(to_replace='.*articles.*', value='Other-iterature', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Cc]ooper.*', value='Other-Cooper book', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Pp]odcasts.*', value='Other-podcasts', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Cc]ompany.*', value='Other-company developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Aa]gency.*', value='Other-company developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*[Ee]mployer.*', value='Other-company developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*clinic.*', value='Other-company developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*own.*', value='Other-company developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*experience.*', value='Other-self developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*self.*', value='Other-self developed', inplace=True, regex=True)
		df[demo].replace(to_replace='^Other$', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*I did not.*', value='Other', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Not sure.*', value='Other-company developed', inplace=True, regex=True)
		df[demo].replace(to_replace='.*Professional.*', value='Other-professional collaboration', inplace=True, regex=True)	

	#print(df[demo])

	return df

def zeroes(df):
	''' fills in zeroes for missing values'''

	# Drop rows where BACB requirements are NOT met
	df = df.drop(df[df['BACB requirements'] == 'No'].index)

	# Fill in zeroes
	df['100% fieldwork pass rate'].fillna(0, inplace=True)
	df['100% fieldwork candidates'].fillna(0, inplace=True)

	return df


def text_to_int(df):
	''' Changes text to integers'''

	df.replace('Almost never (0-20%)', 1, inplace=True)
	df.replace('Rarely (21-40%)', 2, inplace=True)
	df.replace('Sometimes (41-60%)', 3, inplace=True)
	df.replace('Usually (61-80%)', 4, inplace=True)
	df.replace('Almost always (81-100%)', 5, inplace=True)

	df.replace('Performance feedback from another BCBA Supervisor', 'Performance feedback', regex=True, inplace=True)
	df.replace('Supervise RBTs (yes/no)', 'Supervise RBTs', inplace=True)

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
             'Supervision format', \
             'Number of candidates', \
             'Past 12 months candidates', \
             'Allotted hours', \
             'Scheduled hours', \
             'Number of clients', \
             'Who dictates caseload', \
             'RBT supervision %']

	return demog


def make_supervision_behaviors_list():
	''' Create the list of supervision behaviors'''

	sup = ['Literature for new competency area', \
           'Professional groups', \
           'Outside training area - credentialing requirements', \
           'Supervision schedule', \
           'Outside training area - training and supervision', \
           'Schedule contacts', \
           '60% fieldwork hours', \
           'Confirm required skill set', \
           'Practice skill set', \
           'Behavior skills training', \
           'Written supervision contract', \
           'Supervision termination clause', \
           'Performance expectations', \
           'Instructions and demonstration', \
           'Positive and corrective feedback', \
           'Written evaluation system', \
           'Document feedback', \
           'Immediate feedback', \
           'Evaluate supervisee performance', \
           'Evaluate client performance', \
           'Supervision fidelity', \
           'Peer evaluate', \
           'Take baseline', \
           'Detect barriers to supervision', \
           'BST case presentation', \
           'Send agenda', \
           'Meeting notes', \
           'Return communications within 48 hours', \
           'Discourage distractions', \
           'Observe body language', \
           'Maintain positive rapport', \
           'Self-assess interpersonal skills', \
           'Group supervision', \
           'Create group activities', \
           'Include ethics', \
           'Arrive on time', \
           'Discuss how to give feedback', \
           'Schedule direct observations', \
           'Schedule standing supervision appointments', \
           'Continue professional relationship', \
           'Review literature', \
           'Attend conferences', \
           'Participate in peer review', \
           'Seek mentorship', \
           'Supervisory study groups']

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
             'Supervision resources - Other - Text'], \
             #'Q4 - Topics'], \
             inplace=True, axis=1)

	return df


def separate_text(df, demographic):
	''' finds cells with multiple entries and makes new rows for each'''

	# Get the length of the dataframe
	length = len(df)

	# Get the correct column for each demographic
	if demographic == 'Supervision mode':
		col = 81

	if demographic == 'Supervision training':
		col = 82

	elif demographic == 'Supervision resources':
		col = 83

	elif demographic == 'Supervision fieldwork protocol source':
		col = 84

	# Check the top row of the df of interest 'length' number of times
	for row in range(0, length):

        #######################
		#print('Values for the top row:')
		#print(df.iloc[0, col])
		#print('')
        #######################

		# Separate the strings on the comma
		split = re.split(',', df.iloc[0, col])

		########################################
		#print('Here is the split string:')
		#print(split)
		#print('')
		########################################

		# Calculate how many new rows need to be appended
		num = len(split)

		# Make a dataframe to hold separated text: new_rows
		new_rows = pd.DataFrame([df.iloc[0]]*num)

		###################################
		#print('Make '+str(num)+' new rows:')
		#print('')
		###################################

		# Change the values of the new rows to the separated strings
		for n in range(0,num):
			new_rows.iloc[n, col] = split[n]
			############################
			#print(new_rows.iloc[n, col])
			############################
	
		# Ignore index just 'pastes' the dataframes together instead of joining on index
		df = pd.concat([df, new_rows], ignore_index=True)

		#print('Here is the first thing:')
		#print(df.iloc[0, 0])
		#print('')
		#new_df = new_df.reset_index(drop=True)
		
		if 'index' in df.columns:
			df.drop(['index'], inplace=True, axis=1)

		########################################
		#print('This is the new dataframe:')
		#print(df)
		#print('')
		#print(df.index)
		########################################
		#print('')
		#print('Here is the appended dataframe:')
		#print(df.iloc[:, col])
		#print('')
		########################################

		#print('I am dropping:')
		#print(df.iloc[0, col])
		#print('')
		# Drop the original row
		df = df.drop(0)

		##########################################
		#print('Here is the dropped row dataframe')		
		#print(df.loc[:, demographic])
		#print('')
		##########################################
	
    #################################
	#print('And here is what gets returned...')
	#print(df.loc[:, demographic])
    #################################  
  
	return df


if __name__ == '__main__':
	main()
