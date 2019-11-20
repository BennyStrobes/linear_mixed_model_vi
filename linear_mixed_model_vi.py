import numpy as np 
import os
import sys
import pdb






class LMM_VI(object):
	def __init__(self, alpha, beta, n_iter):
		self.alpha_prior = alpha
		self.beta_prior = beta
		self.n_iter = n_iter
	def fit(self, *, X, y, z):
		""" Fit the model.
			Args:
			X: A of floats with shape [num_examples, num_features].
			y: An array of floats with shape [num_examples].
			z: groupings
		"""
		self.X = X
		self.y = y
		self.z = z
		self.initialize_variables()
		self.beta_1_list = []
		self.tau_list = []
		self.var_list = []
		# Loop through VI iterations
		for vi_iter in range(self.n_iter):
			print(vi_iter)
			self.update_beta()
			self.update_alpha()
			self.update_residual_var()
			self.update_tau_var()
			self.beta_1_list.append(self.beta_mu[1])
			self.tau_list.append(self.expected_tau_var)
			self.var_list.append(self.expected_residual_var)
	def initialize_variables(self):
		# Initialize mapping from z-label to individual
		self.z_mapping = {}
		self.z_inverse_mapping = {}
		for i, label in enumerate(np.unique(self.z)):
			self.z_mapping[label] = i
			self.z_inverse_mapping[i] = label
		# Add model dimensions to object
		self.N = len(self.y)
		self.I = len(np.unique(self.z))
		self.D = (self.X).shape[1]
		# Initialize variational parameters
		self.beta_mu = np.zeros(self.D)
		self.beta_var = np.ones(self.D)
		self.alpha_mu = np.zeros(self.I)
		self.alpha_var = np.ones(self.I)
		self.residual_var_alpha = 2.0
		self.residual_var_beta = 1.0
		self.expected_residual_var = self.residual_var_beta/(self.residual_var_alpha-1.0)
		self.tau_var_alpha = 2.0
		self.tau_var_beta = 1.0
		self.expected_tau_var = self.tau_var_beta/(self.tau_var_alpha-1.0)
	def update_beta(self):
		for d in range(self.D):
			# Initialize terms to keep track of
			x_squared = 0
			y_x = 0
			x_x = 0
			x_alpha = 0
			for n in range(self.N):
				individual_index = self.z_mapping[self.z[n]]
				x_squared = x_squared + np.square(self.X[n,d])
				y_x = y_x + self.y[n]*self.X[n,d]
				x_alpha = x_alpha + self.X[n,d]*self.alpha_mu[individual_index]
				for d2 in range(self.D):
					if d != d2:
						x_x = x_x + self.beta_mu[d2]*self.X[n,d]*self.X[n,d2]
			self.beta_var[d] = self.expected_residual_var/x_squared
			self.beta_mu[d] = (self.beta_var[d]/self.expected_residual_var)*(y_x - x_x - x_alpha)
	def update_alpha(self):
		for i in range(self.I):
			indices = np.where(self.z_inverse_mapping[i] == self.z)[0]
			self.alpha_var[i] = 1.0/((len(indices)/self.expected_residual_var) + (1.0/self.expected_tau_var))
			temp_diff = 0.0
			for index in indices:
				predicted_mean = 0
				for d in range(self.D):
					predicted_mean = predicted_mean + self.beta_mu[d]*self.X[index,d]
					#print(str(predicted_mean) + ' ' + str(self.y[index]))
				temp_diff = temp_diff + self.y[index] - predicted_mean
			self.alpha_mu[i] = (self.alpha_var[i]/self.expected_residual_var)*temp_diff
	def update_residual_var(self):
		self.residual_var_alpha = self.alpha_prior + (self.N/2.0)
		temp_residual_var_beta = 0.0
		for n in range(self.N):
			individual_index = self.z_mapping[self.z[n]]
			temp_residual_var_beta = temp_residual_var_beta + self.y[n]*self.y[n]
			temp_residual_var_beta = temp_residual_var_beta - 2.0*self.y[n]*self.alpha_mu[individual_index]
			temp_residual_var_beta = temp_residual_var_beta + (self.alpha_var[individual_index] + np.square(self.alpha_mu[individual_index]))
			feature_weight = 0.0
			feature_product = 0.0
			for d in range(self.D):
				feature_weight = feature_weight + self.X[n,d]*self.beta_mu[d]
				for d2 in range(self.D):
					if d == d2:
						feature_product = feature_product + self.X[n,d]*self.X[n,d2]*(self.beta_var[d] + np.square(self.beta_mu[d]))
					else:
						feature_product = feature_product + self.X[n,d]*self.X[n,d2]*self.beta_mu[d]*self.beta_mu[d2]
			temp_residual_var_beta = temp_residual_var_beta - 2.0*self.y[n]*feature_weight
			temp_residual_var_beta = temp_residual_var_beta + 2.0*feature_weight*self.alpha_mu[individual_index]
			temp_residual_var_beta = temp_residual_var_beta + feature_product
		self.residual_var_beta = self.beta_prior + temp_residual_var_beta/2.0
		self.expected_residual_var = self.residual_var_beta/(self.residual_var_alpha-1.0)
	def update_tau_var(self):
		self.tau_var_alpha = self.alpha_prior + (self.I/2.0)
		temp_residual_var_tau = 0.0
		for i in range(self.I):
			temp_residual_var_tau = temp_residual_var_tau + (self.alpha_var[i] + np.square(self.alpha_mu[i]))
		self.tau_var_beta = self.beta_prior + (temp_residual_var_tau/2.0)
		self.expected_tau_var = self.tau_var_beta/(self.tau_var_alpha-1.0)