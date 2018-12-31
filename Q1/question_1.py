#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import clean_data


def main():
	''' main function'''

	q1_prep()


def q1_prep():
	''' prepare data for question 1'''

	# Make dataframe to hold metadata info
	index = ['Total', 
             'Finished', 
             'Met Supervision Requirements', 
             'Average Time to Complete Survey', 
             'Median Time to Complete Survey',
             'Start Date',
             'End Date',
             'Total Days',
             'BACB %']

	columns = ['']
	data = pd.DataFrame(index=index, columns=columns)

	# Move up one directory level
	os.chdir('..')

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])
	data.iloc[0, 0] = str(len(df)) + ' responses'
	
	# Move back into Q1 folder
	os.chdir('./Q1')

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)
	data.iloc[1, 0] = str(len(finished)) + ' responses'

	# Count responders who answered 'Yes' to meeting BACB requirements for supervision
	yes = finished.drop(finished[finished['BACB requirements'] == 'No'].index)
	data.iloc[2, 0] = str(len(yes)) + ' responses'

	# Calculate average time to complete survey
	avg = round(finished['Duration (in seconds)'].mean() / 60, 1)
	data.iloc[3, 0] = str(avg) + ' minutes'

	med = round(finished['Duration (in seconds)'].median() / 60, 1)
	data.iloc[4, 0] = str(med) + ' minutes'

	# Calculate survey start date, end date, and days between
	data.iloc[5, 0] = finished['Start Date'].min()
	data.iloc[6, 0] = finished['End Date'].max()
	data.iloc[7, 0] = pd.to_datetime(data.iloc[6, 0]) - pd.to_datetime(data.iloc[5, 0])

	# Calculate overall percentate of BCBAs who responded
	bcbas = float(30540)
	overall = round(len(finished)/bcbas, 3)*100
	data.iloc[8, 0] = overall	

	# Save metadata to .csv
	print(data)
	data.to_csv('Survey metadata.csv')

	# Make a graph showing distribution of when responses were submitted
	time_graph(finished)
	
	# Combine columns with text-based option
	combined = clean_data.combine_columns(finished)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)
	
	# Make list of demographics
	d_list = clean_data.make_demographics_list()

	# RESEARCH QUESTION 1
	demographics = question1(metadata, d_list, overall)

	# RESEARCH QUESTION 1 -- Special cases (multiple answers in one cell)
	print('NOW WORKING ON SPECIAL CASES...')
	d_list_special = ['Supervision mode', \
                      'Supervision training', \
                      'Supervision resources', \
                      'Supervision fieldwork protocol source']

	for item in d_list_special:
		special = clean_data.separate_text(metadata, item)
		merged = clean_data.combine_text(special, item)
		question1(merged, [item], overall)
	


def question1(df, demo_list, overall):
	''' answers research question 1: classify responses by demographic'''

	os.chdir('./Q1_graphs')

	for demo in demo_list:

		# replace blank cells with 'NaN'
		df_current = df.replace('', np.nan)

		# drop rows with NaN values
		df_current.dropna(subset=[demo], inplace=True)

		# make bar plot of current demographic
		ax = df_current[demo].value_counts().plot(kind='bar', color='gray')
		manager = plt.get_current_fig_manager()
		manager.resize(*manager.window.maxsize())
		_ = plt.title('Demographic: ' +demo)
		_ = plt.xlabel(demo)
		_ = plt.ylabel('number of responses')
		_ = plt.xticks(rotation=90)

		_ = bar_label(ax)
	
		_ = plt.savefig(demo+'.png', bbox_inches='tight')
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
			state_calc.plot.bar(y='percentage', color='gray', legend=False)
			_ = plt.title('Demographic: State by percentage of BCBAs (November 2018)')
			_ = plt.xlabel('State')
			_ = plt.ylabel('percentage of BCBA responses')
			_ = plt.axhline(y=overall, linestyle='--', color='black')
			dash = mlines.Line2D([], [], color='black', marker='_', label='Overall BCBA response %')
			_ = plt.legend(handles=[dash])
			_ = plt.tight_layout()
			_ = plt.savefig('State by percentage.png')
			_ = plt.close()


	oddballs = ['Years certified', 'Years supervisor']

	for odd in oddballs:
	
		x = df[odd]
		x = x.dropna()
		bins = np.linspace(0, 30, 7)
		_ = plt.hist(x, normed=True, bins=bins, color='gray', histtype='bar', ec='black')
		_ = plt.xlabel(odd)
		_ = plt.ylabel('Percentage of responders')

		plt.savefig(odd + '.png')
		plt.close()
	'''
	# Years supervisor
	x = df['Years supervisor']
	x = x.dropna()
	
	_ = plt.hist(x, normed=True, bins=bins, color='gray', histtype='bar', ec='black')
	_ = plt.xlabel('Years supervisor')
	_ = plt.ylabel('Percentage of responders')

	plt.savefig('Years supervisor.png')
	plt.close()
	'''
	os.chdir('..')
	

	return


def bar_label(ax):
	''' labels bars in bar plot'''
	
	rects = ax.patches

	# for each bar: place a label
	for rect in rects:
		y_value = rect.get_height()
		x_value = rect.get_x() + rect.get_width() / 2

		# number of points between bar and label
		space = 2
		# vertical alignment for positive values
		va = 'bottom'
	
		# If value of bar is negative: place label below bar
		if y_value < 0:
			space *= -1
			va = 'top'
	
		# Use Y value as label and format number with one decimal place
		label = "{:.0f}".format(y_value)

		# Create annotation
		plt.annotate(
			label,
			(x_value, y_value),
			xytext=(0, space),
			textcoords="offset points",
			ha='center',
			va=va)

	return


def time_graph(df):
	''' plots a graph of number of survey responses received by day over survey period'''

	# Split the date-time strings on the spaces
	dt_list = df['End Date'].apply(lambda x: x.split(' '))

	# Get rid of the time part of the date-time string
	dates = dt_list.apply(lambda x: x.pop(0))

	# Convert to pandas datetime format and grab just the date
	data = dates.apply(lambda x: pd.to_datetime(x)).dt.date

	# Group data by date
	grouped = data.value_counts()

	# Order by date
	grouped.sort_index(ascending=True, inplace=True)

	# Graph results
	_ = grouped.plot.bar(color='gray', legend=False)
	_ = plt.title('Dates of submitted surveys')
	_ = plt.xlabel('Date')
	_ = plt.ylabel('Number of submitted surveys')
	_ = plt.tight_layout()
	_ = plt.savefig('Survey endtimes.png')
	#_ = plt.show()

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
