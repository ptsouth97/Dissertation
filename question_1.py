#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
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

	print(dropped.loc[:, 'Supervision resources'])
	# RESEARCH QUESTION 1
	#d_list = clean_data.make_demographics_list()
	#demographics = question1(dropped, d_list)

	# RESEARCH QUESTION 1 -- Special cases (multiple answers in one cell)
	d_list_special = ['Supervision training', \
                      'Supervision resources', \
                      'Supervision fieldwork protocol source']

	for item in d_list_special:
		special = clean_data.separate_text(dropped, item)
		demographics_special = question1(special, [item])
		del(special)



def question1(df, demo_list):
	''' answers research question 1'''

	os.chdir('./Q1_graphs')

	for demo in demo_list:
		_ = df[demo].value_counts().plot(kind='bar', colormap='gray')
		manager = plt.get_current_fig_manager()
		manager.resize(*manager.window.maxsize())
		_ = plt.title('Demographic: ' +demo)
		_ = plt.xlabel(demo)
		_ = plt.ylabel('number of responses')

		'''
		if len(demo_list) > 1:
			_ = plt.tight_layout()
			_ = plt.xticks(rotation=45)
		'''

		_ = plt.xticks(rotation=45)
		_ = plt.tight_layout()
		_ = plt.savefig(demo+'.png', bbox_inches='tight')
		_ = plt.show()
		_ = plt.close()

	os.chdir('..')
	

	return


if __name__ == '__main__':
	main()
