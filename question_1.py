#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
	''' main function'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Combine columns with text-based option
	combined = combine_columns(dataframe)

	# Replace text with integers
	integers = replace_text(combined)
	
	# Drop all the survey metadata
	metadata = drop_metadata(integers)

	# Drop non-supervisor behaviors
	dropped = drop_bx(metadata)
	print(dropped)

	# RESEARCH QUESTION 1
	# demographics = question1(dataframe)


def question1(df):
	''' answers research question 1'''

	demo = df.iloc[:, 17:32]
	print(demo)


if __name__ == '__main__':
	main()
