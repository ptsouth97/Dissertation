#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data


def main():
	''' main function'''

	q3_prep()


def q3_prep():
	''' prepares data for research question 3'''

	# Move up one directory level
	os.chdir('..')

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Move back into question 3 directory
	os.chdir('./Q3')

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)

	# Combine columns with text-based option
	combined = clean_data.combine_columns(finished)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)
	
	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)

	# RESEARCH QUESTION 3
	demo_list = ['Years certified', 'Years supervisor'] 
	sup_list = clean_data.make_supervision_behaviors_list()

	# Get the 'oddball' data into a useable format
	new_df = oddballs(metadata, demo_list)
	print(new_df['Years certified'])

	p_val = question3(new_df, demo_list, sup_list, 'regular')

	p_val.to_csv('Q3 p-value table.csv')	


def oddballs(df, d_list):
	''' puts numerical data into bins'''

	for demo in d_list:
		df[demo] = np.where(df[demo].between(0,5), 0, df[demo])
		df[demo] = np.where(df[demo].between(5,10), 5, df[demo])
		df[demo] = np.where(df[demo].between(10,15), 10, df[demo])
		df[demo] = np.where(df[demo].between(15,20), 15, df[demo])
		df[demo] = np.where(df[demo].between(20,25), 20, df[demo])
		df[demo] = np.where(df[demo].between(25,30), 25, df[demo])

	return df

def question3(df, demo_lst, bx_lst, plot):
	''' Answers research question 3'''
	
	# Change folder for graphs
	os.chdir('./Q3_graphs')
	
	# Build dataframe to hold p-values
	results = pd.DataFrame(index=bx_lst, columns=demo_lst)

	# Loop through each demographic
	for demo in demo_lst:
		
		# replace blank cells with 'Nan'
		df_current = df.replace('', np.nan)

		# drop rows with NaN values
		df_current.dropna(subset=[demo], inplace=True)

		# For each demographic, go through the behaviors			
		for bx in bx_lst:
			grouped = df_current.groupby(demo)
			groupby_to_df = grouped.describe().squeeze()
			names = groupby_to_df.index.tolist()
			print(names)
			length = len(names) + 1
			positions = list(range(1,length))
			grouped_list = list(grouped[bx])
			grouped_df = pd.DataFrame.from_items(grouped_list)
			print(grouped_df)	
			# How many columns: n_cols
			n_cols = len(grouped_df.columns)
			
			# Intialize empty list to store column data
			col_list = []			

			# Initialize degrees of freedom denominator (dfd)
			dfd = 0

			# Make list to hold the average
			#avg = []

			# Make a list of columns from the dataframe and add up how many values are in each column (dfd)
			for col in range(0, n_cols):
				column=grouped_df.iloc[:,col].dropna().tolist()
				col_list.append(column)
				dfd += len(column)
				#avg.append(round(np.mean(col), 3))

			# Convert list to dataframe
			#statistics = pd.DataFrame({'mean':avg}, index=df.columns)
			#print(statistics)

			# Drop empty lists
			col_list = list(filter(None, col_list))

			dfd = dfd - len(col_list)

			# Get the degrees of freedom numerator (dfn)
			dfn = len(col_list) - 1

			# Calculate the critical F value
			Fcrit = stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd)
			
			# Get the F and p values
			f, p = p_value(col_list)
				
			# Assign p value to dataframe position
			results.loc[bx, demo] = round(p, 3)

			# Handle very small p-values
			if p < 0.001:
				p = '<.001' 

			else:
				p = '='+str(round(p, 3))

			f = str(round(f, 3))
			
			# Resize to larger window for bigger graph
			fig = plt.gcf()
			fig.set_size_inches(12, 10)

			# Create boxplot from the lists made from the dataframe columns: col_list
			# bp = plt.boxplot(col_list, patch_artist=True)
			
			############# NEW #################
			# Sort dataframe in descending order
			grouped_df = grouped_df.reindex(grouped_df.mean().sort_values(ascending=False).index, axis=1)
			
			color = dict(boxes='gray', whiskers='black', medians='black', caps='black')
			meanpointprops = dict(marker='s', markeredgecolor='black', markerfacecolor='black')
			_ = grouped_df.plot.box(color=color, patch_artist=True, meanprops=meanpointprops, showmeans=True)
			'''
			# Chance color of boxes
			for box in bp['boxes']:
				box.set(facecolor = 'gray')

			# Change color of median line
			for median in bp['medians']:
				median.set(color = 'black')
			'''
			if plot == 'regular':
				_ = plt.xticks(positions, names, rotation=90)

			else:
				_ = plt.xticks(positions, names, rotation=90)

			new_yticks = ['Almost never (1)', 'Rarely (2)', 'Sometimes (3)', 'Usually (4)', 'Almost always (5)']
			new_xticks = ['0-5 years', '6-10 years', '11-15 years', '16-20 years', '21-25 years', '26-30 years']

			_ = plt.xticks(np.arange(1, 5+1, step=1), new_xticks)
			_ = plt.yticks(np.arange(1, 5+1, step=1), new_yticks)
			#_ = plt.suptitle(bx + ' responses grouped by ' + demo)
			_ = plt.title(bx + ' responses grouped by ' + demo + \
                          ' (F=' + f + ' (F critical='+str(round(Fcrit, 3))+'), p' + p + ')')
			_ = plt.xlabel(demo)
			_ = plt.ylabel(bx + ' responses')
			_ = plt.ylim(0.9, 5.1)

			#if plot == 'regular':
			_ = plt.tight_layout()

			fig = plt.gcf()
			fig.set_size_inches(12, 10)
			_ = plt.savefig(demo+'-'+bx+'.png', dpi=100)
			_ = plt.close()

	os.chdir('..')

	return results


def p_value(columns):
	''' Peform ANOVA based on the number of columns'''

	if len(columns) == 2:
		f, p = stats.f_oneway(columns[0], columns[1])
 
	elif len(columns) == 3:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2])
 
	elif len(columns) == 4:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3])

	elif len(columns) == 5:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4])

	elif len(columns) == 6:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5])

	elif len(columns) == 7:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6])

	elif len(columns) == 8:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7])

	elif len(columns) == 9:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8])

	elif len(columns) == 10:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9])

	elif len(columns) == 11:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10])

	elif len(columns) == 12:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11])

	elif len(columns) == 13:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12])

	elif len(columns) == 14:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13])

	elif len(columns) == 15:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14])

	elif len(columns) == 16:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15])

	elif len(columns) == 17:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16])

	elif len(columns) == 18:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17])

	elif len(columns) == 19:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18])
                              
	elif len(columns) == 20:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19])

	elif len(columns) == 21:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20])

	elif len(columns) == 22:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21])

	elif len(columns) == 23:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22])

	elif len(columns) == 24:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23])

	elif len(columns) == 25:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24]) 

	elif len(columns) == 26:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25])

	elif len(columns) == 27:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26])
 
	elif len(columns) == 28:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27])
 
	elif len(columns) == 29:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28])

	elif len(columns) == 30:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29])

	elif len(columns) == 31:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30])

	elif len(columns) == 32:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31])

	elif len(columns) == 33:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32])

	elif len(columns) == 34:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33])

	elif len(columns) == 35:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34])

	elif len(columns) == 36:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35])
 
	elif len(columns) == 37:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36])
 
	elif len(columns) == 38:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37])
                               
	elif len(columns) == 39:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38])
 
	elif len(columns) == 40:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39])

	elif len(columns) == 41:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40])

	elif len(columns) == 42:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41])

	elif len(columns) == 43:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42])

	elif len(columns) == 44:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43])

	elif len(columns) == 45:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43], columns[44])

	elif len(columns) == 46:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43], columns[44], \
                              columns[45])

	elif len(columns) == 47:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43], columns[44], \
                              columns[45], columns[46])

	elif len(columns) == 48:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43], columns[44], \
                              columns[45], columns[46], columns[47])
 
	elif len(columns) == 49:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43], columns[44], \
                              columns[45], columns[46], columns[47], columns[48])
                              
	elif len(columns) == 50:
		f, p = stats.f_oneway(columns[0], columns[1], columns[2], columns[3], columns[4], \
                              columns[5], columns[6], columns[7], columns[8], columns[9], \
                              columns[10], columns[11], columns[12], columns[13], columns[14], \
                              columns[15], columns[16], columns[17], columns[18], columns[19], \
                              columns[20], columns[21], columns[22], columns[23], columns[24], \
                              columns[25], columns[26], columns[27], columns[28], columns[29], \
                              columns[30], columns[31], columns[32], columns[33], columns[34], \
                              columns[35], columns[36], columns[37], columns[38], columns[39], \
                              columns[40], columns[41], columns[42], columns[43], columns[44], \
                              columns[45], columns[46], columns[47], columns[48], columns[49])

	return f, p

if __name__ == '__main__':
	main()
