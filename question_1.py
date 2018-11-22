#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import clean_data


def main():
	''' main function'''

	q1_prep()


def q1_prep():
	''' prepare data for question 1'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])
	print('TOTAL RESPONSES = ' + str(len(df)))

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)
	print('FINISHED RESPONSES = ' + str(len(finished)))

	# Combine columns with text-based option
	combined = clean_data.combine_columns(finished)
	#print(combined)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)
	#print(integers)	

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)
	#print(metadata)
	
	# Combine similar text from user generated 'Other' fields
	#combined = clean_data.combine_text(metadata)

	# Make list of demographics
	d_list = clean_data.make_demographics_list()

	# RESEARCH QUESTION 1
	demographics = question1(metadata, d_list)
	print('DEMOGRAPHICS')
	#print(demographics)

	# RESEARCH QUESTION 1 -- Special cases (multiple answers in one cell)
	print('NOW WORKING ON SPECIAL CASES...')
	d_list_special = ['Supervision mode', \
                      'Supervision training', \
                      'Supervision resources', \
                      'Supervision fieldwork protocol source']

	for item in d_list_special:
		#print('NOW WORKING ON ' +item)
		special = clean_data.separate_text(metadata, item)
		combined = clean_data.combine_text(special, item)
		demographics_special = question1(combined, [item])
		del(special)



def question1(df, demo_list):
	''' answers research question 1: classify responses by demographic'''

	os.chdir('./Q1_graphs')
	#print('question1 df...')
	#print(df)

	for demo in demo_list:

		# replace blank cells with 'NaN'
		df_current = df.replace('', np.nan)

		# drop rows with NaN values
		df_current.dropna(subset=[demo], inplace=True)

		_ = df_current[demo].value_counts().plot(kind='bar', color='gray')
		manager = plt.get_current_fig_manager()
		manager.resize(*manager.window.maxsize())
		_ = plt.title('Demographic: ' +demo)
		_ = plt.xlabel(demo)
		_ = plt.ylabel('number of responses')

		_ = plt.xticks(rotation=90)
		#_ = plt.tight_layout()
		_ = plt.savefig(demo+'.png', bbox_inches='tight')
		#_ = plt.show()
		_ = plt.close()

		# Special case: calculate percentage of responders by state 
		if demo == 'State':
			grouped = df_current.groupby('State')
			groupby_to_df = grouped.describe().squeeze()
			states = groupby_to_df.index.tolist()
			length = len(states) + 1
			positions = list(range(1,length))
			grouped_list = list(grouped['100% fieldwork candidates'])
			grouped_df = pd.DataFrame.from_items(grouped_list)

			number=[]

			for state in states:
				column = len(grouped_df[state].dropna())
				number.append(column)
			
			state_calc = pd.DataFrame(number, columns=['Count'], index=states)
			
			for state in states:
				state_calc.loc[state, 'BCBAs'] = get_BCBAs(state)
				

			state_calc['percentage'] = state_calc.apply(calc_perc , axis=1)
			state_calc.sort_values(by='percentage', ascending=False, inplace=True)
			print(state_calc)
			state_calc.plot.bar(y='percentage')
			_ = plt.title('Demographic: State by percentage of BCBAs (November 2018)')
			_ = plt.xlabel('State')
			_ = plt.ylabel('percentage of BCBA responses')
			_ = plt.savefig('State by percentage')
			_ = plt.show()
			_ = plt.close()

	os.chdir('..')
	

	return


def calc_perc(row):
	''' calculates percentage'''

	return row['Count'] / row['BCBAs'] * 100


def get_BCBAs(st):
	''' returns number of BCBAs in state'''

	if st == 'Alabama':
		num = 238

	elif st == 'Alaska':
		num = 50

	elif st == 'Arizona':
		num = 363

	elif st == 'Arkansas':
		num = 72

	elif st == 'California':
		num = 5154

	elif st == 'Colorado':
		num = 660

	elif st == 'Connecticut':
		num = 751

	elif st == 'Delaware':
		num = 52

	elif st == 'Florida':
		num = 3296

	elif st == 'Georgia':
		num = 471

	elif st == 'Hawaii':
		num = 182

	elif st == 'Idaho':
		num = 39

	elif st == 'Illinois':
		num = 960

	elif st == 'Indiana':
		num = 597

	elif st == 'Iowa':
		num = 121

	elif st == 'Kansas':
		num = 164

	elif st == 'Kentucky':
		num = 247

	elif st == 'Louisiana':
		num = 286

	elif st == 'Maine':
		num = 184

	elif st == 'Maryland':
		num = 449

	elif st == 'Massachusetts':
		num = 2203

	elif st == 'Michigan':
		num = 880

	elif st == 'Minnesota':
		num = 193

	elif st == 'Mississippi':
		num = 75

	elif st == 'Missouri':
		num = 455

	elif st == 'Montana':
		num = 46

	elif st == 'Nebraska':
		num = 123

	elif st == 'Nevada':
		num = 192

	elif st == 'New Hampshire':
		num = 263

	elif st == 'New Jersey':
		num = 1597

	elif st == 'New Mexico':
		num = 71

	elif st == 'New York':
		num = 1823

	elif st == 'North Carolina':
		num = 434

	elif st == 'North Dakota':
		num = 28

	elif st == 'Ohio':
		num = 514

	elif st == 'Oklahoma':
		num = 87

	elif st == 'Oregon':
		num = 171

	elif st == 'Pennsylvania':
		num = 1368

	elif st == 'Rhode Island':
		num = 188

	elif st == 'South Carolina':
		num = 301

	elif st == 'South Dakota':
		num = 33
	
	elif st == 'Tennessee':
		num = 535

	elif st == 'Texas':
		num = 1646

	elif st == 'Utah':
		num = 283

	elif st == 'Vermont':
		num = 137

	elif st == 'Virginia':
		num = 1040

	elif st == 'Washington':
		num = 685

	elif st == 'West Virginia':
		num = 78

	elif st == 'Wisconsin':
		num = 240

	elif st == 'Wyoming':
		num = 17

	else: # st == 'I live outside of the United States':
		num = 495

	return num


if __name__ == '__main__':
	main()
