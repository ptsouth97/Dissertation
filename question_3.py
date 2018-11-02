#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data


def main():
	''' main function'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Combine columns with text-based option
	combined = clean_data.combine_columns(dataframe)

	# Replace text with integers
	integers = clean_data.text_to_int(combined)
	
	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)

	# Drop non-supervisor behaviors
	dropped = clean_data.drop_bx(metadata)

	# RESEARCH QUESTION 3
	#demo_list = clean_data.make_demographics_list()
	sup_list = clean_data.make_supervision_behaviors_list()
	#question3(dropped, demo_list, sup_list)

	# Special cases (multiple answers in one cell)
	d_list_special = ['Supervision training', \
                      'Supervision resources', \
                      'Supervision fieldwork protocol source']

	for item in d_list_special:
		special = clean_data.separate_text(dropped, item)
		question3(special, [item], sup_list)


def question3(df, demo_lst, bx_lst):
	''' Answers research question 3'''

	# Change folder for graphs
	os.chdir('./Q3_graphs')

	# Build dataframe to hold p-values
	results = pd.DataFrame(index=bx_lst, columns=demo_lst)

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

			p = p_value(col_list)
			# p_values.append(p)

			results.loc[bx, demo] = p

			# Create boxplot
			bp = plt.boxplot(col_list, patch_artist=True)

			# Chance color of boxes
			for box in bp['boxes']:
				box.set(facecolor = 'gray')

			# Change color of median line
			for median in bp['medians']:
				median.set(color = 'black')

			_ = plt.xticks(positions, names, rotation=45)
			_ = plt.yticks(np.arange(1, 5+1, step=1))
			_ = plt.title(bx+' (p='+str(p)+')')
			_ = plt.xlabel(demo)
			_ = plt.ylabel('responses')
			_ = plt.ylim(1, 5)
			#_ = plt.annotate('p='+str(p), xy=(0.6, 1.5))
			_ = plt.tight_layout()
			_ = plt.savefig(demo+'-'+bx+'.png')
			_ = plt.close()


	#print('p_values:')
	#print(p_values)

	#print(results)
	results.to_csv('p-value table.csv')

	os.chdir('..')

	return


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

	return p

if __name__ == '__main__':
	main()
