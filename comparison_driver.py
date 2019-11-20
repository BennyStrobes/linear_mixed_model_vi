import numpy as np 
import os
import sys
import pdb
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linear_mixed_model_vi 
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


def load_data(file_name):
	f = open(file_name)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split(',')
		if head_count == 0:
			head_count = head_count + 1
			header = data
			continue
		pdb.set_trace()


################
# Load in data
################
data = sm.datasets.get_rdataset("dietox", "geepack").data
z = np.asarray(data["Pig"])
X = np.transpose(np.vstack(( np.ones(len(np.asarray(data["Time"]))), np.asarray(data["Time"]))))
y = np.asarray(data["Weight"])
#file_name = 'InstEval.csv'
#y, X, z = load_data(file_name)

##################
# Fit LMM using home-built variational inference
##################
lmm_vi = linear_mixed_model_vi.LMM_VI(alpha=1e-3, beta=1e-3, n_iter=2000)
lmm_vi.fit(X=X, y=y, z=z)

##################
# Fit LMM using built in python package
##################
md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()

pdb.set_trace()
# Make some plots
plt.plot(range(2000), lmm_vi.tau_list[:2000], color='blue')
plt.plot(range(2000), [40.394]*2000, color='black')
plt.xlabel('VI Iteration')
plt.ylabel('Random effects variance')
plt.show()