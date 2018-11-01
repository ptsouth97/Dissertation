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
	demo_list = clean_data.make_demographics_list()
	sup_list = clean_data.make_supervision_behaviors_list()
	question3(dropped, demo_list, sup_list)


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
			# color = dict(boxes='black', whiskers='black', medians='white', caps='black')
			#_ = col_list.plot.box(color=color, patch_artist=True)
			_ = plt.xticks(positions, names, rotation=45)
			_ = plt.yticks(np.arange(1, 5, step=1))
			_ = plt.title(bx+' (p='+str(p)+')')
			_ = plt.xlabel(demo)
			_ = plt.ylabel('responses')
			_ = plt.ylim(1, 5)
			#_ = plt.annotate('p='+str(p), xy=(0.6, 1.5))
			_ = plt.tight_layout()
			_ = plt.savefig(demo+'-'+bx+'.png')
			_ = plt.close()


	print('p_values:')
	print(p_values)

	os.chdir('..')

	return


if __name__ == '__main__':
	main()
