#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ecdf import ecdf
from simulation import gen_ran_nums
from hypothesis_testing import diff_of_means, draw_perm_reps
from matplotlib.offsetbox import AnchoredText


def main():

	# Get simulated data
	df = gen_ran_nums()

	#df = pd.read_csv('Numerical Responses.csv', header=1)

	bx = df['Area of study'] == 'Behavior'
	data_1 = df[bx]
	data1 = data_1.iloc[:, 0]
	
	ed = df['Area of study'] == 'Education'
	data_2 = df[ed]
	data2 = data_2.iloc[:, 0]
	
	empirical_diff_means = diff_of_means(data1, data2)
	print('The difference in means is', empirical_diff_means)

	perm_replicates = draw_perm_reps(data1, data2, diff_of_means, 10000)

	if(empirical_diff_means > 0):
		p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

	else:
		p = np.sum(perm_replicates <= empirical_diff_means) / len(perm_replicates)

	print('The p-value is', p)
	
	print(df)
	sns.set()
	plt.margins(0.02)
	sns.boxplot(x='Area of study', y='Literature review', data=df)
	plt.ylabel('Literature review responses')
	plt.annotate('difference of means = {}'.format(round(empirical_diff_means, 4)), xy=(0.05, 0.95), xycoords='axes fraction')
	plt.annotate('p-value = {}'.format(p), xy=(0.05, 0.9), xycoords='axes fraction')
	plt.show()
	# plt.savefig('boxplot.png')

	
	'''plt.subplot(2, 2, 1)
	sns.swarmplot(x='Certification', y='Literature review', data=df, palette='Blues')
	plt.ylim(1, 5)
	plt.margins(0.02)

	plt.subplot(2, 2, 2)
	sns.boxplot(x='Certification', y='Attend conferences', data=df, palette='BuGn_r')
	plt.ylim(1, 5)
	plt.margins(0.02)

	plt.subplot(2, 2, 3)
	sns.boxplot(x='Certification', y='Peer review', data=df, palette='GnBu_d')
	plt.ylim(1, 5)
	plt.margins(0.02)

	plt.subplot(2, 2, 4)
	sns.violinplot(x='Certification', y='Mentorship', data=df)
	plt.ylim(1, 5)
	plt.margins(0.02)

	plt.tight_layout()
	plt.show()'''

	# df.head()
	# df.info()

'''	# Slice dataframes
	behavior = df['Area of study'] == 'Behavior analysis'
	bx_df = df[behavior]
	lit_bx = bx_df['Literature review']

	education = df['Area of study'] == 'Education'
	ed_df = df[education]
	lit_ed = ed_df['Literature review']

	# Get random data to compare
	rnd = gen_ran_nums()

	# Compute ECDFs
	x_bx, y_bx = ecdf(lit_bx)
	x_ed, y_ed = ecdf(lit_ed)
	x_rn, y_rn = ecdf(rnd)	

	# Plot ECDF
	_ = plt.plot(x_bx, y_bx, marker='.', linestyle='none')
	_ = plt.plot(x_ed, y_ed, marker='+', linestyle='none')
	_ = plt.plot(x_rn, y_rn, marker='.', linestyle='none')

	# Calculate normal distribution statistics
	mean = np.mean(rnd)
	std = np.std(rnd)
	samples = np.random.normal(mean, std, size=10000)
	x_nm , y_nm = ecdf(samples)
	_ = plt.plot(x_nm, y_nm, marker='.', linestyle='none')

	plt.margins(0.02)
	_ = plt.xticks(np.arange(1, 6, 1))
	_ = plt.xlabel('Literature review')
	_ = plt.ylabel('ECDF')
	plt.legend(('behavior', 'education'), loc='lower right')
	plt.show()'''

if __name__ == "__main__":
	main()
