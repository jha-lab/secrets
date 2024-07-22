from scipy.stats import norm, t
import numpy as np
import math
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import time
import os
from collections import OrderedDict


"""
Notes:

-Below methods assume 2-sided test based on the z-statistic.
-Assumes that we're working on large samples (n >> 30), in which case the z-statistic is OK. [@ Fundamentals of Biostatistics].
-improve estimates in the low-data regimes by swapping with t-stat. This will be important for establishing the min. sampling complexity.



These equations were adopted from Fundamental of Biostatistics 
"""

"""
Min. sample size to detect the delta
	If 2 variances are provided, then this is the sample size PER ARM
	Else, total sample size


Fundamental of Biostatistics 
	Eq. 7.23 (paired samples reduces to one sample testing). var_1=variance in the deltas.
	Eq. 8.24 (independent samples), var_1, var_2 rep. the variance of each group


p.229

Note, this assumes the distribution is normal. 
If the estimates of the var_1, var_2 are based on samples >200, 
then this formula holds (see comment under get_z_test)
"""
def get_sample_size(alpha, beta, var_1, var_2, delta, k=None):
	if k is not None:
		n1=((var_1+var_2/k)*(norm.ppf(1-(alpha/2))+norm.ppf(1-beta))**2)/(delta**2)
		n2=((k*var_1+var_2)*(norm.ppf(1-(alpha/2))+norm.ppf(1-beta))**2)/(delta**2)
		return n1, n2
	else:
		return ((var_1+var_2)*(norm.ppf(1-(alpha/2))+norm.ppf(1-beta))**2)/(delta**2)


"""
Min. magnitude of delta under the alternative hypothesis that can be detected 
"""
def get_min_delta(alpha, beta, var_1, var_2, group_size):
	return (((var_1+var_2)*(norm.ppf(1-alpha/2)+norm.ppf(1-beta))**2)/group_size)**0.5

"""
Returns 'valid' alpha/beta pairs i.e., since alpha indexes the T1 err rate of the 2-sided test, alpha is constrained to lie [0,1]

See eq. 8.24 for derivation

"""
def get_valid_alphas_betas(var_1,var_2,group_size,delta, sweep_beta=True):
	alphas=[]
	betas=[]

	# the positive intercept yields lower alphas for given betas
	intercept=1*((group_size*(delta**2)/(var_1+var_2))**0.5)

	# for beta in np.concatenate([np.array([0.01,0.05]),np.linspace(0.1, 0.9, 9)]):
	if sweep_beta:
		for beta in np.concatenate([np.array([0.01,0.05,0.1,0.2])]):
			alpha=2*(1-norm.cdf(intercept-norm.ppf(1-beta)))
			if alpha >=0 and alpha <=1: 
				alphas.append(alpha)
				betas.append(beta)
	else:
		for alpha in np.array([0.05]):
			beta=1-norm.cdf(intercept-norm.ppf(1-(alpha/2)))
			alphas.append(alpha)
			betas.append(beta)

	return np.array(alphas), np.array(betas)


"""
Fundamentals of Biostatistics, p.229

z_test is valid for group_size > 200
otherwise, use the t-test
"""
def get_z_test(group_size, alpha, true_var_ctrl=None, true_var_treat=None, 
				true_var_delta=None, one_sample=False,
				ctrl_size=None, treat_size=None):
	if one_sample:
		def get_test_outcome(sample_mean):
			sample_stat=sample_mean/((true_var_delta/group_size)**0.5)
			return abs(sample_stat) > norm.ppf(1-(alpha/2))
	else:
		def get_test_outcome(sample_mean):
			if ctrl_size is not None and treat_size is not None:
				sample_var=(true_var_ctrl/ctrl_size) + (true_var_treat/treat_size)
				sample_stat=sample_mean/(sample_var**0.5)
			else:
				sample_stat=sample_mean/(((true_var_ctrl+true_var_treat)/group_size)**0.5)
			return abs(sample_stat) > norm.ppf(1-(alpha/2))
	return get_test_outcome


"""
n1, n2: arm size (for our purposes equal). n2 is ignored when one_sample=True. Note that n1 is the TOTAL # samples used in the one sample.
loc: 0 should be 0 (was just introduced for debugging)
"""
def get_t_test(n1,n2=None,alpha=0.05,var_equal=False,loc=0,one_sample=False,cv=None):

	if one_sample:
		dof=n1-1
		cv=cv if cv is not None else t.ppf(1-(alpha/2),dof)
		def get_test_outcome(sample_mean,sample_var):
			sample_stat=sample_mean/((sample_var/n1)**0.5)
			# print(sample_stat,t.ppf(1-(alpha/2),dof))
			return abs(sample_stat) > cv
	else:
		def get_test_outcome(mean_1,mean_2,sample_var_1,sample_var_2):
			if var_equal:
				s=(((n1-1)*sample_var_1+(n2-1)*sample_var_2)/(n1+n2-2))**0.5
				sample_stat=(mean_1-mean_2)/(s*((1/n1 +1/n2))**0.5)
				dof=n1+n2-2
			else:
				sample_stat=(mean_1-mean_2)/(((sample_var_1/n1) + (sample_var_2/n2))**0.5)
				dof_num=((sample_var_1/n1) + (sample_var_2/n2))**2
				dof_den=((sample_var_1/n1)**2)/(n1-1)+((sample_var_2/n2)**2)/(n2-1)
				dof=math.floor(dof_num/dof_den)
				# print(sample_stat, dof)


			return abs(sample_stat) > t.ppf(1-(alpha/2),dof,loc=loc)
	return get_test_outcome


"""
sids: unique patient ids
"""
def get_data_splits(data, labels, sids, data_counter=None, size=0, debug=0, seed=0, normalize=0, train_ratio=0.7):	
	data,labels,sids=np.copy(data),np.copy(labels),np.copy(sids)
	if data_counter is not None: data_counter=np.copy(data_counter)

	# equal-sized groups
	min_num=min(np.sum(labels),np.sum(labels==0))
	assert np.sum(labels)==min_num
	assert np.sum(labels==0)==min_num
	
	np.random.seed(0) # fix the complete dataset
	ind_ctrl=np.random.choice(np.where(labels==0)[0],min_num,replace=False)
	ind_treat=np.random.choice(np.where(labels==1)[0],min_num,replace=False)

	ind_keep=np.concatenate((ind_ctrl,ind_treat))
	data,labels,sids=data[ind_keep],labels[ind_keep],sids[ind_keep]
	if data_counter is not None: data_counter=data_counter[ind_keep]
	
	size=size if size else len(sids)

	num_train=int(size*train_ratio) - int(size*train_ratio)%2 # correct to avoid removing data as the group should be divisible into 2 equal-sized arms
	num_val=size-num_train

	np.random.seed(seed) # randomize sampling

	pids_ctrl=sids[np.random.choice(np.where(labels==0)[0],num_train//2,replace=False)]
	pids_treat=sids[np.random.choice(np.where(labels==1)[0],num_train//2,replace=False)]
	train_pids=np.concatenate((pids_ctrl,pids_treat))

	non_train_pids=np.setdiff1d(sids,train_pids)
	non_train_ind=np.array(list(map(lambda sid: np.where(sids==sid)[0][0], non_train_pids)))
	non_train_labels=labels[non_train_ind]
	assert np.array_equal(sids[non_train_ind],non_train_pids)

	pids_ctrl=non_train_pids[np.random.choice(np.where(non_train_labels==0)[0],num_val//2,replace=False)]
	pids_treat=non_train_pids[np.random.choice(np.where(non_train_labels==1)[0],num_val//2,replace=False)]
	val_pids=np.concatenate((pids_ctrl,pids_treat))

	# np.random.seed(seed) # randomize sampling
	# orig_target_pids=sids[:] #np.unique(sids)

	# np.random.shuffle(orig_target_pids)

	# size=size if size else len(orig_target_pids)
	# target_pids=orig_target_pids[:size]

	# train_pids=np.random.choice(target_pids,int(0.7*len(target_pids)),replace=False)
	# val_pids=np.setdiff1d(target_pids,train_pids)

	ind_tpids=np.array(list(map(lambda s: np.where(sids==s)[0], train_pids))).squeeze()
	ind_vpids=np.array(list(map(lambda s: np.where(sids==s)[0], val_pids))).squeeze()

	train_data,val_data=data[ind_tpids],data[ind_vpids]
	train_data_counter, val_data_counter=(data_counter[ind_tpids], data_counter[ind_vpids]) if data_counter is not None else ([],[])
	train_labels,val_labels=labels[ind_tpids],labels[ind_vpids]
	train_sids,val_sids=sids[ind_tpids], sids[ind_vpids]

	assert np.array_equal(train_pids,train_sids)
	assert np.array_equal(val_pids,val_sids)

	assert np.allclose(len(train_data)+len(val_data),size,atol=2)
	assert len(np.intersect1d(train_pids,val_pids))==0

	if debug: 
		print("Frac. placebo: orig: %.4f, train: %.4f, val: %.4f"\
		  %(labels.sum()/len(labels), train_labels.sum()/len(train_labels),val_labels.sum()/len(val_labels)))

	return [train_data, train_data_counter, train_labels, train_sids], [val_data, val_data_counter, val_labels, val_sids]



def normalize(data, labels, scheme, normalize_by=None):

	ns,nt,nf=data.shape

	norm_data=np.copy(data)

	scalers_by_label_type=[]

	if normalize_by == "label":
		for label_type in [0,1]:

			ind_label_type=np.where(labels==label_type)[0]
			data_label_type=norm_data[ind_label_type]

			scalers=[]
			for cfi in np.arange(nf):
				scaler=StandardScaler() if scheme == "standard" else MinMaxScaler((0,1))
				norm_data[ind_label_type,:,cfi]=scaler.fit_transform(data_label_type[:,:,cfi].reshape(-1,1)).reshape(len(data_label_type),-1)
				scalers.append(scaler)
			scalers_by_label_type.append(scalers)

	elif normalize_by == "row": 
		scalers=[]
		for ri in np.arange(ns):
			scaler=StandardScaler() if scheme == "standard" else MinMaxScaler((0,1))
			norm_data[ri]=scaler.fit_transform(norm_data[ri].flatten().reshape(-1,1)).reshape(nt,nf)
			scalers.append(scaler)
		scalers_by_label_type.append(scalers)
	else: # normalize across entire group
		# print("normalize")
		scalers=[]
		for cfi in np.arange(nf):
			scaler=StandardScaler() if scheme == "standard" else MinMaxScaler((0,1))
			norm_data[:,:,cfi]=scaler.fit_transform(norm_data[:,:,cfi].reshape(-1,1)).reshape(len(norm_data),-1)
			scalers.append(scaler)
		scalers_by_label_type.append(scalers)

	return norm_data, scalers_by_label_type


def unnormalize(data, labels, scalers_by_label_type, normalize_by=None):


	ns,nt,nf=data.shape

	unnorm_data=np.copy(data)


	if normalize_by == "label":
		for label_type in [0,1]:

			scalers=scalers_by_label_type[label_type]

			ind_label_type=np.where(labels==label_type)[0]
			data_label_type=unnorm_data[ind_label_type]

			for cfi in np.arange(nf):
				# technically should ignore missing but small fraction
				# for ti in tis:
				# 	scaler=StandardScaler()
				# 	train_data[:,ti,cfi]=scaler.fit_transform(train_data[:,ti,cfi].reshape(-1,1)).squeeze()
				# 	val_data[:,ti,cfi]=scaler.transform(val_data[:,ti,cfi].reshape(-1,1)).squeeze()
				# 	scalers[cf].append(scaler)

				# scaler=StandardScaler()
				scaler=scalers[cfi]
				unnorm_data[ind_label_type,:,cfi]=scaler.inverse_transform(data_label_type[:,:,cfi].reshape(-1,1)).reshape(len(data_label_type),-1)
	elif normalize_by=="row":
		scalers=scalers_by_label_type[0]
		for ri in np.arange(ns):
			scaler=scalers[ri]
			unnorm_data[ri]=scaler.inverse_transform(unnorm_data[ri].flatten().reshape(-1,1)).reshape(nt,nf)
	else:
		# print("unnormalize")
		scalers=scalers_by_label_type[0]
		for cfi in np.arange(nf):
			scaler=scalers[cfi]
			unnorm_data[:,:,cfi]=scaler.inverse_transform(unnorm_data[:,:,cfi].reshape(-1,1)).reshape(len(unnorm_data),-1)

	return unnorm_data



def get_sample(data, labels, sids, arm_size, seed=0, eval_alpha=False, with_replacement=False, data_counter=None):

	data,labels,sids=np.copy(data),np.copy(labels),np.copy(sids)
	
	if data_counter is not None: data_counter=np.copy(data_counter)

	np.random.seed(seed)
	ind_ctrl=np.where(labels==0)[0]
	ind_treat=np.where(labels==1)[0]
	sind_ctrl=np.random.choice(ind_ctrl,arm_size,replace=with_replacement)
	sind_treat=np.random.choice(ind_ctrl if eval_alpha else ind_treat,arm_size,replace=with_replacement)
	data=np.concatenate((data[sind_ctrl],data[sind_treat]))
	labels=np.array([0]*arm_size+[1]*arm_size) # there is always a 'treated' and a 'control' arm 
	sids=np.arange(2*arm_size) # actuals ids don't matter as long as unique i.e., get_data_splits
	
	# print(sind_ctrl)
	# print(sind_treat)

	if data_counter is not None: data_counter=np.concatenate((data_counter[sind_ctrl],data_counter[sind_treat]))

	if data_counter is not None:
		return data,data_counter,labels,sids

	return data,labels,sids



def parse_helper(inp_str):
		if inp_str == "None":
			return None
		return float(inp_str)

def parse_str_helper(inp_str):
	if inp_str == "None":
		return None
	return inp_str
