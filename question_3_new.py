#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import numpy as np
import clean_data
import question_2


def main():
	''' Loads and cleans data for analysis'''

	q3_new_prep()


def q3_new_prep():
	''' Prepares data for research question 3'''
	
	# Read the survey results file into a pandas dataframe
	file_name = 'responses.csv'
	df = pd.read_csv(file_name, header=1, skiprows=[2])

	# Drop un-finished responses
	finished = df.drop(df[df['Finished'] == False].index)

	# Drop responses that didn't meet BACB requirements
	finished = df.drop(df[df['BACB requirements'] == 'No'].index)

	# Combine columns with text-based option
	combined = clean_data.combine_columns(finished)
	
	# Change states to regions
	regions = clean_data.state_to_region(combined)

	# Replace text with integers
	integers = clean_data.text_to_int(regions)

	# Drop all the survey metadata
	metadata = clean_data.drop_metadata(integers)

	# Get the oddballs ready ('Years certified' and 'Years supervisor')
	oddities = ['Years certified', 'Years supervisor']
	metadata = oddballs(metadata, oddities)

	# Make supervision behavior list
	sup_list = clean_data.make_supervision_behaviors_list()

	# Impute missing values
	for sup in sup_list:
		metadata[sup].fillna(metadata[sup].mean(skipna=True), inplace=True)	

	# Make the demographic list
	demo_list = clean_data.make_demographics_list() 

	# Add PEC columns that average behaviors based on domain
	df = add_pecc_columns(metadata)
	df.to_csv('q3_new.csv')

	# Run stats and make charts for new question 3
	q3_new(df, demo_list)


def add_pecc_columns(df):
	''' Creates new columns that average responder's supervision behaviors for each PECC domain'''


	
	# SUPERVISING WITHIN YOUR SCOPE
	df['PECC 5.01'] = df.apply(lambda row:(row.loc['Literature for new competency'] + \
                                       row.loc['Professional groups'] + \
                                       row.loc['Credentialing requirements'] + \
                                       row.loc['Training and supervision'] + \
                                       row.loc['Supervisory study groups'])/5, axis=1)

 
	# SUPERVISORY VOLUME
	df['PECC 5.02'] = df.apply(lambda row:(row.loc['Arrive on time'] + \
                                       row.loc['60% fieldwork hours'] + \
                                       row.loc['Supervision schedule'] + \
                                       row.loc['Schedule contacts'])/4, axis=1)

 
	# SUPERVISORY DELEGATION
	df['PECC 5.03'] = df.apply(lambda row:(row.loc['Confirm required skill set'] + \
                                       row.loc['Practice skill set'])/2, axis=1)

 
	# DESIGNING EFFECTIVE TRAINING  
	df['PECC 5.04'] = df.apply(lambda row:(row.loc['Group supervision'] + \
                                       row.loc['Create group activities'] + \
                                       row.loc['Include ethics'] + \
                                       row.loc['Behavior skills training'] + \
                                       row.loc['Discuss how to give feedback'])/5, axis=1)

 
	# COMMUNICATION OF SUPERVISION CONDITIONS
	df['PECC 5.05'] = df.apply(lambda row:(row.loc['Send agenda'] + \
                                       row.loc['Performance expectations'] + \
                                       row.loc['Written supervision contract'] + \
                                       row.loc['Review supervision contract'] + \
                                       row.loc['Supervision termination clause'])/5, axis=1)


	# PROVIDING FEEDBACK TO SUPERVISEES
	df['PECC 5.06'] = df.apply(lambda row:(row.loc['Observe body language'] + \
                                       row.loc['Written evaluation system'] + \
                                       row.loc['Positive and corrective feedback'] + \
                                       row.loc['Document feedback'] + \
                                       row.loc['Immediate feedback'] + \
                                       row.loc['Instructions and demonstration'])/6, axis=1)
	

	# EVALUATING THE EFFECTS OF SUPERVISION
	df['PECC 5.07'] = df.apply(lambda row:(row.loc['Self-assess interpersonal skills'] + \
                                       row.loc['Supervision fidelity'] + \
                                       row.loc['Evaluate client performance'] + \
                                       row.loc['Evaluate supervisee performance'])/4, axis=1)


	# MISC
	df['MISC'] = df.apply(lambda row:(row.loc['Peer evaluate'] + \
                                       row.loc['Return communications within 48'] + \
                                       row.loc['Take baseline'] + \
                                       row.loc['Detect barriers to supervision'] + \
                                       row.loc['BST case presentation'] + \
                                       row.loc['Send agenda'] + \
                                       row.loc['Continue relationship'] + \
                                       row.loc['Observe body language'] + \
                                       row.loc['Maintain positive rapport'] + \
                                       row.loc['Self-assess interpersonal skills'] + \
                                       row.loc['Group supervision'] + \
                                       row.loc['Create group activities'] + \
                                       row.loc['Supervisory study groups'] + \
                                       row.loc['Include ethics'] + \
                                       row.loc['Arrive on time'] + \
                                       row.loc['Discuss how to give feedback'] + \
                                       row.loc['Schedule direct observations'] + \
                                       row.loc['Schedule standing appointments'] + \
                                       row.loc['Review literature'] + \
                                       row.loc['Meeting notes'] + \
                                       row.loc['Attend conferences'] + \
                                       row.loc['Participate in peer review'] + \
                                       row.loc['Seek mentorship'] + \
                                       row.loc['60% fieldwork hours'] + \
                                       row.loc['Schedule contacts'] + \
                                       row.loc['Discourage distractions'])/26, axis=1)
	
	df.to_csv('q3_new.csv')
	
	return df

	
def q3_new(df, demo_lst):
	''' Calculate stats and make graphs for new question 3'''

	domains = ['PECC 5.01', 'PECC 5.02', 'PECC 5.03', 'PECC 5.04', 'PECC 5.05', 'PECC 5.06', 'PECC 5.07', 'MISC']

	# Build dataframe to FINAL DOMAIN RESULTS
	domain_results = pd.DataFrame()

	for domain in domains:


		for demo in demo_lst:

			df_current = df.replace('', np.nan)
			df_current.dropna(subset=[demo], inplace=True)

			# Group the current domain by the current demographic			
			grouped = df_current.groupby(demo)
			groupby_to_df = grouped.describe().squeeze()
			names = groupby_to_df.index.tolist()
			length = len(names) + 1
			positions = list(range(1,length))
			grouped_list = list(grouped[domain])
			grouped_df = pd.DataFrame.from_items(grouped_list)

			# How many columns: n_cols
			n_cols = len(grouped_df.columns)
			
			# Intialize empty list to store column data
			col_list = []			

			# Initialize degrees of freedom denominator (dfd)
			dfd = 0

			# Make list to hold the average
			avg = []

			# Make list to hold the standard deviation
			std = []

			# Make a list of columns from the dataframe and add up how many values are in each column (dfd)
			for col in range(0, n_cols):
				column=grouped_df.iloc[:,col].dropna().tolist()
				col_list.append(column)
				dfd += len(column)
				avg.append(round(np.mean(column), 2))
				std.append(round(np.std(column), 2))

			# Convert list to dataframe
			statistics = pd.DataFrame({'M':avg, 'SD':std}, index=names)
				
			# Drop empty lists
			col_list = list(filter(None, col_list))

			dfd = dfd - len(col_list)

			# Get the degrees of freedom numerator (dfn)
			dfn = len(col_list) - 1

			# Calculate the critical F value
			Fcrit = stats.f.ppf(q=1-0.05, dfn=dfn, dfd=dfd)
			
			# Get the F and p values
			f, p = p_value(col_list)
			print('THIS IS THE GROUPED DF')
			print(grouped_df)

			# Initialize new dataframe to hold results for tukey test
			tukey_df = pd.DataFrame()			

			# Check for significance
			if p < .05:
				for col_name in grouped_df.columns:
					temp = grouped_df.loc[:, [col_name]]
					temp['identity'] = col_name
					temp = temp.rename(columns={col_name:'score'})
					tukey_df = tukey_df.append(temp)
			
				print('HERE IS THE FINAL DF FOR TUKEY')	
				tukey_df = tukey_df.dropna()
				tukey_df = tukey_df.reset_index(drop=True)
				print(tukey_df)

				posthoc = pairwise_tukeyhsd(tukey_df['score'], tukey_df['identity'])
				tukey_results = pd.DataFrame(data=posthoc._results_table.data[1:],
                                          columns=posthoc._results_table.data[0])
				tukey_results.to_csv('Tukey-'+domain+'-'+demo+'.csv', index=False)
				print(posthoc)
				
				#mod = MultiComparison(tukey_df['score'], tukey_df['identity'])
				#print(mod.tukeyhsd())

			# Round F value
			f = str(round(f, 2))
				
			# Handle very small p-values
			if p < 0.001:
				p = '.00' 

			else:
				p = str(round(p, 2))
			
			# Create unique name for statistics df
			# unique_name = domain + '-' + demo

			# Assign F and p values to dataframe position
			statistics = statistics.assign(F=f, p=p)
			#statistics.to_csv(domain + '-' + demo + '.csv')
			print(demo)
			print(statistics)

			# Append statistics to final domain results
			domain_results = domain_results.append(statistics)

			# Sort dataframe in descending order
			grouped_df = grouped_df.reindex(grouped_df.mean().sort_values(ascending=False).index, axis=1)

			#print('Sorted grouped_df for ' + domain)
			#print(grouped_df)			
			color = dict(boxes='gray', whiskers='black', medians='black', caps='black')
			meanpointprops = dict(marker='s', markeredgecolor='black', markerfacecolor='black')
			_ = grouped_df.plot.box(color=color, patch_artist=True, meanprops=meanpointprops, showmeans=True)

			_ = plt.xticks(rotation=90)

			new_yticks = ['Almost never (1)', 'Rarely (2)', 'Sometimes (3)', 'Usually (4)', 'Almost always (5)']

			
			_ = plt.yticks(np.arange(1, 5+1, step=1), new_yticks)
			#_ = plt.suptitle(bx + ' responses grouped by ' + demo)
			_ = plt.title(domain + ' responses grouped by ' + demo + \
                          ', F=' + f + ' (F critical='+str(round(Fcrit, 3))+'), p' + p)
			_ = plt.xlabel(demo)
			_ = plt.ylabel(domain + ' responses')
			_ = plt.ylim(0.9, 5.1)

			#if plot == 'regular':
			if (domain != 'Credentialing requirements' and domain != 'Training and supervision'):
				_ = plt.tight_layout()

			
			fig = plt.gcf()
			fig.set_size_inches(12, 10)
			_ = plt.savefig(demo+'-'+domain+'.png', dpi=100)
			#_ = plt.show()
			_ = plt.close()

		domain_results.to_csv(domain + '.csv')
		domain_results.drop(domain_results.index, inplace=True)

	return 
	

def oddballs(df, d_list):
	'''Handles 'Years certified' and 'Years supervisor' that need histogram instead of bar chart'''
	
	for demo in d_list:
		df[demo] = np.where(df[demo].between(0,5), 0, df[demo])
		df[demo] = np.where(df[demo].between(5,10), 5, df[demo])
		df[demo] = np.where(df[demo].between(10,15), 10, df[demo])
		df[demo] = np.where(df[demo].between(15,20), 15, df[demo])
		df[demo] = np.where(df[demo].between(20,25), 20, df[demo])
		df[demo] = np.where(df[demo].between(25,30), 25, df[demo])

		df[demo].replace(0, '0-5 years', inplace=True)
		df[demo].replace(5, '6-10 years', inplace=True)
		df[demo].replace(10, '11-15 years', inplace=True)
		df[demo].replace(15, '16-20 years', inplace=True)
		df[demo].replace(20, '21-25 years', inplace=True)
		df[demo].replace(25, '26-30 years', inplace=True)

	return df

	
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
