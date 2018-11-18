#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import clean_data


def main():
	''' Loads and cleans data for analysis'''

	q2_prep()


def q2_prep():
	''' Prepares data for research question 3'''

	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	dataframe = pd.read_csv(file_name, header=1, skiprows=[2])

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(dataframe)

	# Drop non-supervisor behaviors
	dropped_bx = clean_data.drop_bx(metadata)
	
	# Drop demographics
	dropped_demo = clean_data.drop_demographics(dropped_bx)

	# Replace text with integers
	integers = clean_data.text_to_int(dropped_demo)

	# RESEARCH QUESTION 2
	question2(integers)
	

def question2(df):
	''' answers research question 2'''

	# change directory
	os.chdir('./Q2_graphs')	

	# Run ANOVA
	data = [df[col].dropna() for col in df]
	f, p = stats.f_oneway(*data)
	
	# Make a boxplot	
	color = dict(boxes='gray', whiskers='black', medians='black', caps='black')
	_ = df.plot.box(color=color, patch_artist=True) #notch=1)
	#_ = df.boxplot(patch_artist=True)
	_ = plt.yticks(np.arange(1, 5+1, step=1))
	_ = plt.xticks(rotation=90)
	_ = plt.xlabel('Individual behaviors')
	_ = plt.ylabel('Survey response')
	_ = plt.suptitle('Supervisory Behaviors')
	_ = plt.title('p-value='+str(p))
	#_ = plt.annotate('p='+str(p), xy=(30,1))
	#_ = plt.grid(b=None)

	# Resize to larger window for bigger graph
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	_ = plt.tight_layout()
	_ = plt.savefig('SupervisoryBehaviorsBoxplot.png')
	_ = plt.show()
	_ = plt.close()

	os.chdir('..')
	
	return

	
if __name__ == '__main__':
	main()
