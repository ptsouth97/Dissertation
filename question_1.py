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

	# RESEARCH QUESTION 1
	demographics = question1(dropped)


def question1(df):
	''' answers research question 1'''

	os.chdir('./Q1_graphs')

	demo_list = clean_data.make_demographics_list()

	for demo in demo_list:
		_ = df[demo].value_counts().plot(kind='bar', colormap='gray')
		_ = plt.title('Demographic: ' +demo)
		_ = plt.xlabel(demo)
		_ = plt.ylabel('number of responses')
		_ = plt.xticks(rotation=45)
		_ = plt.tight_layout()
		_ = plt.savefig(demo+'.png')
		_ = plt.show()
		_ = plt.close()

	os.chdir('..')
	

	return


if __name__ == '__main__':
	main()
