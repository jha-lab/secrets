import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, r2_score

from functools import partial

import multiprocessing as mp

import argparse
import os, shutil

from types import SimpleNamespace

sys.path.append("..")
import src
from src.utils import *

import pickle

import time

import scipy.integrate

import pandas as pd

from src import critical_value as cv

from types import SimpleNamespace

import copy


def get_estimator_helper(train_data,val_data,iof,estimator):
	estimator.fit(train_data[:,0],train_data[:,1:,iof].squeeze())

	val_r2=r2_score(val_data[:,1:,iof].squeeze(),estimator.predict(val_data[:,0]), multioutput="uniform_average") 
	train_r2=r2_score(train_data[:,1:,iof].squeeze(),estimator.predict(train_data[:,0]), multioutput="uniform_average")

	return estimator, val_r2, train_r2

"""
train_data: ns x nt x nf
val_data: ""

Fits an estimator on the baseline, including multiple covariates, data to the post intervention for iof feature
"""
def get_estimator(train_data, val_data, iof, model_type):
	model_kwargs={}

	best_estimator, best_val_r2, best_train_r2=None,-1*float("inf"),-1*float("inf")

	if model_type == "RandomForestRegressor":	

		for n_trees in [1, 5, 10, 100, 1000]:
			nmk=model_kwargs.copy()
			nmk['n_estimators']=n_trees
			estimator,val_r2,train_r2=get_estimator_helper(train_data,val_data,iof,RandomForestRegressor(**nmk))

			if val_r2>best_val_r2:
				best_estimator=estimator
				best_val_r2=val_r2
				best_train_r2=train_r2


	elif model_type == "NSLinearRegression":

		for fit_intercept in [True,False]:
			nmk=model_kwargs.copy()
			nmk['fit_intercept']=fit_intercept
			estimator,val_r2,train_r2=get_estimator_helper(train_data,val_data,iof,lm.LinearRegression(**nmk))

			if val_r2>best_val_r2:
				best_estimator=estimator
				best_val_r2=val_r2
				best_train_r2=train_r2



	elif model_type == "NSRidge":

		for fit_intercept in [True,False]:
			for alpha in np.logspace(-3,3,7):
				nmk=model_kwargs.copy()
				nmk['fit_intercept']=fit_intercept
				nmk['alpha']=alpha
				estimator,val_r2,train_r2=get_estimator_helper(train_data,val_data,iof,lm.Ridge(**nmk))

				if val_r2>best_val_r2:
					best_estimator=estimator
					best_val_r2=val_r2
					best_train_r2=train_r2
	

	return best_estimator, best_val_r2, best_train_r2





def get_truncated_data(u,s,vh,coverage,true_data):
	s_norm=s/np.sum(s)
	cdf=np.cumsum(s_norm)
	if coverage==1: # precision issues
		ind_thres=len(s)
	else:
		ind_thres=np.where(cdf>=coverage)[0][0]+1 # min coverage
	ut=u[:,:ind_thres]
	st=s[:ind_thres]
#     assert np.allclose(np.sum(st),coverage) 
	vht=vh[:ind_thres]
	trunc_data=np.matmul(ut,np.matmul(np.diag(st),vht))
	
	return trunc_data, mean_squared_error(true_data,trunc_data,squared=False)

"""
data: [nt]

Currently outcome_type: twa assumes mgtx
"""
def get_outcome(data, outcome_type="delta"):
	data=np.copy(data)

	if outcome_type=="delta":
		return data[-1]-data[0]

	if outcome_type=="twa":
		cardinal_vids=np.array([3,6,7,8,9,10,11,12,13,14,15,16,17,18])
		return scipy.integrate.trapezoid(data,cardinal_vids)/(cardinal_vids[-1]-cardinal_vids[0])


"""
data: [nt x nf] 
"""
def get_ite(data, data_counter, label, ind_outcome_ft, outcome_type="delta"):
	data,data_counter=np.copy(data),np.copy(data_counter)
	return (1 if label else -1)*(get_outcome(data[:,ind_outcome_ft],outcome_type) - get_outcome(data_counter[:,ind_outcome_ft],outcome_type))


"""
data: [ns x nt x nf]
"""
def get_ate(full_data, full_labels, ind_outcome_ft, outcome_type="delta"):
	treat_response=np.apply_along_axis(get_outcome,1,full_data[full_labels==1,:,ind_outcome_ft], outcome_type)
	ctrl_response=np.apply_along_axis(get_outcome,1,full_data[full_labels==0,:,ind_outcome_ft], outcome_type)
	# print(treat_response)
	# print(ctrl_response)
	# print(np.mean(treat_response),np.mean(ctrl_response))
	return np.mean(treat_response)-np.mean(ctrl_response)


class SI:
	def __init__(self, threshold=None, k=None, model_type="LinearRegression", model_kwargs={}, use_pcr=False):
		self.threshold=threshold
		self.k=k
		self.model_type=model_type
		self.model_kwargs=model_kwargs
		self.use_pcr=use_pcr

	"""
	train_data: # donors x timepoints x features
	unit_data: timepoints x features
	it0: 0

	Returns 2d matrix containing only POST-timepoint predictions (i.e., timepoints-1) x features
	"""
	def fit_predict(self, train_data, unit_data, it0=0, debug=0):
		train_data,unit_data=np.copy(train_data),np.copy(unit_data)
		if self.use_pcr:
			return self.fit_predict_pcr(train_data, unit_data, it0)
		return self.fit_predict_mrsc(train_data, unit_data, it0, debug)


	def fit_predict_pcr(self, train_data, unit_data, it0):

		nd,nt,nf=train_data.shape

		# covariates must be bounded 
		complete_data=np.concatenate((train_data,np.expand_dims(unit_data,0)))
		# ft_scalers=[]
		# for fi in np.arange(nf):
		# 	scaler=MinMaxScaler((-1,1))
		# 	complete_data[:,:,fi]=scaler.fit_transform(complete_data[:,:,fi].reshape(-1,1)).reshape((nd+1,nt))
		# 	ft_scalers.append(scaler)

		train_data,unit_data=complete_data[:-1],complete_data[-1]

		baseline_train_data,post_train_data=train_data[:,:it0+1],train_data[:,it0+1:]

		# pre-intervention fit
		# switch to matrix form where shape is (nt0*nf)xnd
		u,s,vh=np.linalg.svd(baseline_train_data.reshape(-1,nd), full_matrices=False)
		# unique solution under l2 norm
		inv_baseline_train=np.zeros((nd,nf*(it0+1)))
		for ir in range(self.k):
			inv_baseline_train+=((1/s[ir])*np.outer(vh[ir],u[:,ir]))
		weight=np.matmul(inv_baseline_train,unit_data[:it0+1].reshape(-1,1))

		# unnormalize
		post_pred=np.matmul(post_train_data.reshape(nd,-1).transpose(1,0),weight).reshape(post_train_data.shape[1:])
		# for fi,scaler in enumerate(ft_scalers):
		# 	post_pred[:,fi]=scaler.inverse_transform(post_pred[:,fi].reshape(-1,1)).squeeze()
		return post_pred

	def fit_predict_mrsc(self, train_data, unit_data, it0, debug=False):
		if debug: print("fit_predict_mrsc")
		
		nd,nt,nf=train_data.shape


		train_data=train_data.reshape(nd,-1)


		u,s,vh=np.linalg.svd(train_data, full_matrices=False)


		r=len(s)-1
		if self.threshold is not None and self.threshold<1:
			s_norm=s/np.sum(s)
			coverage=np.cumsum(s_norm)
#             print(np.where(coverage>=threshold))
			# r=np.where(coverage>=self.threshold)[0][0]+1
			r=np.where(coverage>=self.threshold)[0][0]
		elif self.threshold is not None and self.threshold==1:
			r=len(s)-1
		elif self.k is not None:
			r=self.k

		# truncate
		trunc_train_data=np.zeros(train_data.shape)
		for ir in range(r+1):
			trunc_train_data+=s[ir]*np.outer(u[:,ir],vh[ir])
		trunc_train_data=trunc_train_data.reshape(nd,nt,nf)
		
		# fit on all the baseline data (1 timepoint)
		model=getattr(lm,self.model_type)(**self.model_kwargs)
		model.fit(trunc_train_data[:,:it0+1].reshape(nd,-1).transpose(1,0), 
					unit_data[:it0+1].flatten())
		

		if debug:
			print(np.amin(trunc_train_data[:,:it0+1],axis=0))
			print(np.amax(trunc_train_data[:,:it0+1],axis=0))
			print(unit_data[:it0+1].flatten())
			# print(trunc_train_data[:,:it0+1].reshape(nd,-1).min(),trunc_train_data[:,:it0+1].reshape(nd,-1).max(),unit_data[:it0+1].flatten().min(),unit_data[:it0+1].flatten().max()) 
			print(model.coef_)

		return model.predict(trunc_train_data[:,it0+1:,:].reshape(nd,-1).transpose(1,0)).reshape(nt-(it0+1),nf), \
				model.predict(trunc_train_data[:,:it0+1].reshape(nd,-1).transpose(1,0)) # assess baseline fit



"""
Penalize only non-baseline (assumed to be at timepoint 0)
"""
def get_config(train_data, val_data, ind_outcome_ft, model_type, model_kwargs, use_pcr):
	train_data,val_data=np.copy(train_data),np.copy(val_data)
	_,nt,nf=train_data.shape

	best_val_r2=-1*float("inf")
	best_config={'threshold':None, 'model_type': model_type, 'model_kwargs': model_kwargs, 'use_pcr':use_pcr, 'k': None}

	if use_pcr:
		for k in np.arange(1,min(len(train_data), train_data.shape[2])+1):
			config={'threshold':None, 'model_type':model_type,'model_kwargs':model_kwargs, 'use_pcr':use_pcr, 'k':k}
			val_preds=np.zeros(val_data.shape)
			for i in range(len(val_data)):
				val_preds[i,1:]=SI(**config).fit_predict(train_data, val_data[i])
			val_preds=np.array(val_preds)
			r2=r2_score(val_data[:,-1,ind_outcome_ft],val_preds[:,-1,ind_outcome_ft])
			if r2>best_val_r2:
				best_val_r2=r2
				best_config['k']=k
	else:
		for threshold in np.linspace(0.1,1,10):

			alphas=np.array([-1])
			if model_type == "Ridge":
				alphas=np.logspace(-3,3,7)
				max_iter=None
			elif model_type == "Lasso":
				alphas=np.logspace(-3,3,7)
				max_iter=10000


			for alpha in alphas:
				
				if model_type in ["Lasso", "Ridge"]:
					model_kwargs['alpha']=alpha
					model_kwargs['max_iter']=max_iter


				config={'threshold':threshold, 'model_type':model_type,'model_kwargs':model_kwargs, 'use_pcr':use_pcr, 'k':None}
				val_preds=np.zeros(val_data.shape)
				val_baseline_preds=np.zeros((len(val_data),nf))
				for i in range(len(val_data)):
					val_preds[i,1:],val_baseline_preds[i]=SI(**config).fit_predict(train_data, val_data[i])
				val_preds=np.array(val_preds)
				val_baseline_preds=np.array(val_baseline_preds)
				r2=r2_score(val_data[:,1:,ind_outcome_ft],val_preds[:,1:,ind_outcome_ft], multioutput="uniform_average") 
				if r2>best_val_r2:
					best_val_r2=r2
					best_config['threshold']=threshold
					best_train_r2=r2_score(val_data[:,0,ind_outcome_ft],val_baseline_preds[:,ind_outcome_ft])

	return best_config, best_val_r2, best_train_r2

"""
path_feature_scores: str Note the scores exclude the outcome feature assumed to be index 0. 


"""
def get_dataset_topk(path_feature_scores, topk, eval_alpha, seed, data):
	with open(path_feature_scores,'rb') as f:
		res=pickle.load(f)

	val_ctrl_r2=res[eval_alpha][0][:,seed]
	val_treat_r2=res[eval_alpha][1][:,seed]

	avg_val_r2=np.mean(np.array([val_ctrl_r2,val_treat_r2]),axis=0)

	
	ind_sort=np.argsort(avg_val_r2)[::-1][:topk]


	ind_keep=np.array([0]+list(ind_sort+1))

	return data[:,:,ind_keep]




"""
ind_outcome_ft: deprecated i.e., it should be 0
"""

def get_ate_from_ite(full_data, full_labels, full_sids, arm_size, seed, ind_outcome_ft, eval_alpha, var_ctrl=None, var_treat=None, 
					model=None, normalize_flag=0, train=False, model_type="LinearRegression", model_kwargs={}, debug=False, use_pcr=False,
					normalize_scheme="standard",normalize_by=None,with_replacement=True,outcome_type="delta",use_unbiased_var=1,
					random_k=-1, path_feature_scores=None, topk=-1
					):
	# print("get_ate_from_ite")
	
	if arm_size>0:
		merged_data,merged_labels,merged_sids=get_sample(full_data,full_labels,full_sids,arm_size,seed,
													eval_alpha=eval_alpha,with_replacement=with_replacement)
	else:
		merged_data,merged_labels,merged_sids=np.copy(full_data),np.copy(full_labels),np.copy(full_sids)

	if topk!=-1:
		merged_data=get_dataset_topk(path_feature_scores, topk, eval_alpha, seed, merged_data)
		# print(merged_data.shape)
		
	elif random_k != -1:
		# print("random_k")
		np.random.seed(seed)
		feature_indices=np.random.choice(np.arange(1,merged_data.shape[2]),random_k,replace=False)
		feature_indices=np.append(np.array([0]),feature_indices)
		merged_data=merged_data[:,:,feature_indices]
		# print(feature_indices)
		# print(merged_data.shape)

	# we can normalize on the entire donor set b/c practically have this
	merged_data,scalers=normalize(merged_data, merged_labels, normalize_scheme, normalize_by) if normalize_flag else (merged_data,[])

	train_ds,val_ds=get_data_splits(merged_data,merged_labels,merged_sids)[:2]
	train_data,_,train_labels,train_sids=train_ds
	val_data,_,val_labels,val_sids=val_ds

	# if model is None and train: model=train_model(train_data,train_labels,val_data,val_labels,seed,ind_outcome_ft, debug=debug)

	# print(merged_data.shape)
	merged_data_counter_preds=np.zeros(merged_data.shape) 
	merged_data_counter_preds[:,0]=np.copy(merged_data[:,0]) # baseline across all features is given

	r2s=[]
	train_r2s=[]
	baseline_r2s=[]

	if model_type in ["LinearRegression", "Ridge"]: # SI
		for donor_label in [0,1]:
			config,val_r2,avg_train_r2=get_config(train_data[train_labels==donor_label], val_data[val_labels==donor_label], ind_outcome_ft, model_type, model_kwargs, use_pcr)
			r2s.append(val_r2)
			train_r2s.append(avg_train_r2)
			# print("donor_label: %d, val_r2: %.4f"%(donor_label,val_r2))
			# print("config", config)

			ind_donor=np.where(merged_labels==donor_label)[0]
			ind_target=np.where(merged_labels==(1-donor_label))[0]
			
			donor_data=merged_data[ind_donor]
			target_data=merged_data[ind_target]

			baseline_preds=[]
			for i in range(len(ind_target)):

				merged_data_counter_preds[ind_target[i],1:],baseline_pred=SI(**config).fit_predict(donor_data, target_data[i])
				baseline_preds.append(baseline_pred)

			baseline_preds=np.array(baseline_preds)
			baseline_r2s.append(r2_score(target_data[:,0,ind_outcome_ft],baseline_preds[:,ind_outcome_ft]))

	elif model_type in ["RandomForestRegressor", "NSLinearRegression", "NSRidge"]:
		for donor_label in [0,1]:
			estimator,val_r2,avg_train_r2=get_estimator(train_data[train_labels==donor_label], val_data[val_labels==donor_label], ind_outcome_ft, model_type)

			r2s.append(val_r2)
			train_r2s.append(avg_train_r2)
			# print("donor_label: %d, val_r2: %.4f"%(donor_label,val_r2))
			# print("config", config)

			# ind_donor=np.where(merged_labels==donor_label)[0]
			ind_target=np.where(merged_labels==(1-donor_label))[0]
			
			# donor_data=merged_data[ind_donor]
			target_data=merged_data[ind_target]

			target_preds=estimator.predict(target_data[:,0])

			if len(target_preds.shape)==1: # 1d array due to 1 post intervention timepoint as with icare,champ 
				target_preds=target_preds.reshape(len(target_preds),1)
			merged_data_counter_preds[ind_target,1:,ind_outcome_ft]=target_preds
			baseline_r2s=[-1,-1]
	else:
		baseline_r2s=train_r2s=r2s=[-1,-1]
		
	if normalize_flag:
		merged_data=unnormalize(merged_data, merged_labels, scalers, normalize_by)
		merged_data_counter_preds=unnormalize(merged_data_counter_preds, merged_labels, scalers, normalize_by)

	# print(merged_data.min(),merged_data.max(),merged_data_counter_preds.min(),merged_data_counter_preds.max())
	# print(np.where(merged_data_counter_preds==merged_data_counter_preds.min()))

	
	ites=np.array(list(map(partial(get_ite,ind_outcome_ft=ind_outcome_ft,outcome_type=outcome_type),merged_data, merged_data_counter_preds, merged_labels)))

	# print("finished get_ate_from_ite")

	return np.mean(ites), np.var(ites, ddof=use_unbiased_var), get_ate(merged_data, merged_labels, ind_outcome_ft, outcome_type), \
			merged_data, merged_labels, merged_data_counter_preds, ites, np.array(r2s), np.array(train_r2s), np.array(baseline_r2s)


"""
The variance is estimated by assuming the treated and control groups are independent (due to the standard RCT randomization)
"""
def get_true_dataset_parameters(data,labels,sids,iof, outcome_type="delta",use_unbiased_var=1):
	ate=get_ate(data,labels,iof, outcome_type)

	treat_response=np.apply_along_axis(get_outcome,1,data[labels==1,:,iof], outcome_type)
	ctrl_response=np.apply_along_axis(get_outcome,1,data[labels==0,:,iof], outcome_type)

	var_ctrl=np.var(ctrl_response,ddof=use_unbiased_var)
	var_treat=np.var(treat_response,ddof=use_unbiased_var)

	mean_ctrl=np.mean(ctrl_response)
	mean_treat=np.mean(treat_response)

	# group_size=len(data)//2
	# alphas,betas=get_valid_alphas_betas(var_ctrl,var_treat,group_size,ate)
	return ate, var_ctrl, var_treat, mean_ctrl, mean_treat


def get_model_kwargs(args):
	model_kwargs={}
	if args.model_type in ['Lasso', 'RidgeRegression']:
		model_kwargs['alpha']=args.reg
	model_kwargs['fit_intercept']=bool(args.fit_intercept)
	return model_kwargs


def get_reject_rate(cvs, test_function, itess, num_seeds, size, use_unbiased_var=1):
	o=0
	for s in np.arange(num_seeds):
		test=test_function(cv=cvs[s])
		sitess=itess[size*s:size*(s+1)]
		o+=test(np.mean(sitess),np.var(sitess,ddof=use_unbiased_var))

	assert size*(s+1) == len(itess)
	
	return o/(num_seeds)
	

def main_with_data_seeds(args, full_ds, features, ind_outcome_ft):
	full_data,full_labels,full_sids=full_ds

	main_results_path=args.results_path


	# num_data_seeds=args.data_seed_end-args.data_seed_start+1
	args.seed_start,args.seed_end=0,args.num_tune_seeds
	# args.eval_alpha=1

	for data_seed in np.arange(args.data_seed_start,args.data_seed_end+1):
		full_data_seed,full_labels_seed,full_sids_seed=get_sample(full_data,full_labels,full_sids,args.arm_size,data_seed,
																	eval_alpha=args.eval_alpha,with_replacement=1)
		args.results_path=os.path.join(main_results_path,"data_seed_%d/eval_alpha_%d"%(data_seed,args.eval_alpha))

		# if data_seed != 63: continue
		try:
			main(copy.deepcopy(args), [full_data_seed, full_labels_seed, full_sids_seed], features, ind_outcome_ft, 0)
		except Exception as e:
			print("failed on data_seed %d"%data_seed, e)

		


def main(args, full_ds, features, ind_outcome_ft, print_results=True, save_results=True):

	full_data, full_labels, full_sids=full_ds

	if print_results: 
		print(full_data.shape)
	# print(features)

	num_processes=int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 6 # 10?

	# use the 'test'/full set to estimate these params
	true_delta_non_zero,true_var_ctrl,true_var_treat,true_mean_ctrl,true_mean_treat=get_true_dataset_parameters(full_data,full_labels,full_sids,ind_outcome_ft,args.outcome_type,args.use_unbiased_var)
	true_var_delta=true_var_ctrl+true_var_treat # upperbound under iid assumption
	
	# for the given sample size, what are theoretical error rates
	alphas,betas=get_valid_alphas_betas(true_var_ctrl,true_var_treat,args.arm_size,true_delta_non_zero)
	if args.beta==-1:
		alpha,beta=args.alpha,args.beta
	else:
		alpha,beta=alphas[betas==args.beta][0],args.beta

	if args.beta>0:
		n1=round(get_sample_size(alpha,beta,true_var_delta,0,true_delta_non_zero)) # theoretical max. # subjects needed under paired testing
		n2=round(get_sample_size(alpha,beta,true_var_ctrl,true_var_treat,true_delta_non_zero)) # min # subjects per arm under two independent sample
		assert n1==n2==args.arm_size
		print(true_delta_non_zero,true_var_delta,true_var_ctrl,true_var_treat,true_mean_treat,true_mean_ctrl,n1,n2,alpha,beta)
	else:
		if print_results: print(true_delta_non_zero,true_mean_ctrl,true_mean_treat,true_var_ctrl,true_var_treat,alpha)

	# test with the desired alpha, beta levels
	if args.use_z_test:
		one_sample_test=get_z_test(2*args.arm_size, alpha, true_var_delta=true_var_delta, one_sample=True)
		two_sample_test=get_z_test(args.arm_size,alpha,true_var_ctrl,true_var_treat,one_sample=False)
	else:
		one_sample_test=get_t_test(2*args.arm_size,alpha=alpha,one_sample=True,cv=args.critical_value) # note we evaluate SI on the full dataset so the # samples is the total across arms
		two_sample_test=get_t_test(args.arm_size,args.arm_size,alpha,var_equal=False,one_sample=False)

	
	os.makedirs(args.results_path,exist_ok=True)

	si_ate_errs=[]
	naive_ate_errs=[]

	test_outcome_si=[]
	test_outcome_naive=[]

	baseline_ctrl_r2s, baseline_treat_r2s=[],[]
	train_ctrl_r2s,train_treat_r2s=[],[]
	val_ctrl_r2s,val_treat_r2s=[],[]

	itess=[]

	# si_vars=[]
	mean_ctrls,mean_treats,var_ctrls,var_treats=[],[],[],[]

	orig_seed_start=args.seed_start

	if args.resume_from_state:
		try:
			with open(os.path.join(args.results_path,'state.pickle'),'rb') as f:
				state=pickle.load(f)

			args.seed_start=state['seed']+1
			si_ate_errs=state['si_ate_errs']
			naive_ate_errs=state['naive_ate_errs']
			test_outcome_si=state['test_outcome_si']
			test_outcome_naive=state['test_outcome_naive']
			baseline_ctrl_r2s,baseline_treat_r2s=state['baseline_ctrl_r2s'],state['baseline_treat_r2s']
			train_ctrl_r2s,train_treat_r2s=state['train_ctrl_r2s'],state['train_treat_r2s']
			val_ctrl_r2s,val_treat_r2s=state['val_ctrl_r2s'],state['val_treat_r2s']
			itess=state['itess']

			if print_results: print("resumed from state")
		except:
			pass
	
	
	save_every=math.ceil((args.seed_end+1)//args.num_checkpoints) if args.num_checkpoints else args.save_every


	if args.use_mp:
		with (mp.get_context("spawn").Pool() if args.use_spawn else mp.Pool(processes=num_processes)) as pool:
			results=[pool.apply_async(get_ate_from_ite, 
										(full_data,full_labels,full_sids,args.arm_size,
											seed,ind_outcome_ft,args.eval_alpha,true_var_ctrl,true_var_treat), 
						{'train':args.train, 'model_type': args.model_type, 'model_kwargs': get_model_kwargs(args),
						'normalize_flag': args.normalize, 'debug': args.debug, 'use_pcr':args.use_pcr,
						'model': None,
						'normalize_scheme': args.normalize_scheme,
						'normalize_by': args.normalize_by,
						'with_replacement':args.with_replacement,
						'outcome_type':args.outcome_type,
						'use_unbiased_var':args.use_unbiased_var,
						'random_k':args.random_k,
						'path_feature_scores': args.path_feature_scores,
						'topk': args.topk
						}) \
					for seed in np.arange(args.seed_start,args.seed_end+1 if args.seed_interrupt == -1 else args.seed_interrupt)]

			for result_i, result in enumerate(results):
				
				
				si_ate,si_var,naive_ate,data,labels,data_counter_preds,ites,r2,avg_train_r2,baseline_r2=result.get()

				si_ate_errs.append(si_ate)
				naive_ate_errs.append(naive_ate)

				train_ctrl_r2s.append(avg_train_r2[0])
				train_treat_r2s.append(avg_train_r2[1])

				val_ctrl_r2s.append(r2[0])
				val_treat_r2s.append(r2[1])

				baseline_ctrl_r2s.append(baseline_r2[0])
				baseline_treat_r2s.append(baseline_r2[1])

				itess.extend(ites)

				# si_vars.append(si_var)

				
				if args.use_z_test:
					test_outcome_si.append(one_sample_test(si_ate))
					test_outcome_naive.append(two_sample_test(naive_ate))
				else:
					test_outcome_si.append(one_sample_test(si_ate,si_var))
					naive_ate,var_ctrl,var_treat,mean_ctrl,mean_treat=get_true_dataset_parameters(data,labels,np.arange(len(data)),ind_outcome_ft,args.outcome_type,args.use_unbiased_var)
					test_outcome_naive.append(two_sample_test(mean_ctrl,mean_treat,var_ctrl,var_treat))
					mean_ctrls.append(mean_ctrl)
					mean_treats.append(mean_treat)
					var_ctrls.append(var_ctrl)
					var_treats.append(var_treat)

				if save_every > 0 and (result_i % save_every) == 0:
					with open(os.path.join(args.results_path,'state.pickle'),'wb') as f:
						pickle.dump({'seed':args.seed_start+result_i,
									'si_ate_errs':si_ate_errs,
									'naive_ate_errs':naive_ate_errs,
									'test_outcome_si':test_outcome_si,
									'test_outcome_naive':test_outcome_naive,
									'baseline_ctrl_r2s':baseline_ctrl_r2s,
									'baseline_treat_r2s':baseline_treat_r2s,
									'train_ctrl_r2s':train_ctrl_r2s,
									'train_treat_r2s':train_treat_r2s,
									'val_ctrl_r2s':val_ctrl_r2s,
									'val_treat_r2s':val_treat_r2s,
									'itess':itess,
									# 'si_vars':si_vars,
									'mean_ctrls':mean_ctrls,
									'mean_treats':mean_treats,
									'var_ctrls':var_ctrls,
									'var_treats':var_treats
									},f)

				if args.print_every>0 and result_i%args.print_every==0: print(result_i)


	else:
		for seed in np.arange(args.seed_start,args.seed_end+1):
			si_ate,si_var,naive_ate,\
			data,labels,data_counter_preds,ites,r2,avg_train_r2, baseline_r2=get_ate_from_ite(full_data,full_labels,full_sids,
															args.arm_size,seed,ind_outcome_ft,args.eval_alpha, 
															true_var_ctrl, true_var_treat, 
															model=None, normalize_flag=args.normalize, train=args.train, 
															model_type=args.model_type, model_kwargs=get_model_kwargs(args), 
															debug=False, use_pcr=args.use_pcr,
															normalize_scheme=args.normalize_scheme,
															normalize_by=args.normalize_by,
															with_replacement=args.with_replacement,
															outcome_type=args.outcome_type,
															use_unbiased_var=args.use_unbiased_var,
															random_k=args.random_k,
															path_feature_scores=args.path_feature_scores,
															topk=args.topk
															)

			si_ate_errs.append(si_ate)
			naive_ate_errs.append(naive_ate)

			train_ctrl_r2s.append(avg_train_r2[0])
			train_treat_r2s.append(avg_train_r2[1])

			val_ctrl_r2s.append(r2[0])
			val_treat_r2s.append(r2[1])

			baseline_ctrl_r2s.append(baseline_r2[0])
			baseline_treat_r2s.append(baseline_r2[1])

			itess.extend(ites)

			if args.use_z_test:
				test_outcome_si.append(one_sample_test(si_ate))
				test_outcome_naive.append(two_sample_test(naive_ate))
			else:
				test_outcome_si.append(one_sample_test(si_ate,si_var))
				naive_ate,var_ctrl,var_treat,mean_ctrl,mean_treat=get_true_dataset_parameters(data,labels,np.arange(len(data)),ind_outcome_ft)
				# print(naive_ate, var_ctrl, var_treat, mean_ctrl, mean_treat)
				test_outcome_naive.append(two_sample_test(mean_ctrl,mean_treat,var_ctrl,var_treat))

			if save_every>0 and (seed%save_every)==0:
				with open(os.path.join(args.results_path,'state.pickle'),'wb') as f:
					pickle.dump({'seed':seed,
								'si_ate_errs':si_ate_errs,
								'naive_ate_errs':naive_ate_errs,
								'test_outcome_si':test_outcome_si,
								'test_outcome_naive':test_outcome_naive,
								'baseline_ctrl_r2s':baseline_ctrl_r2s,
								'baseline_treat_r2s':baseline_treat_r2s,
								'train_ctrl_r2s':train_ctrl_r2s,
								'train_treat_r2s':train_treat_r2s,
								'val_ctrl_r2s':val_ctrl_r2s,
								'val_treat_r2s':val_treat_r2s,
								'itess':itess},f)

			if args.print_every>0 and seed%args.print_every==0: print(seed)


	true_delta=0 if args.eval_alpha else true_delta_non_zero
	true_outcome=1-args.eval_alpha

	si_ates=np.array(si_ate_errs[:])
	naive_ates=np.array(naive_ate_errs[:])


	si_rmse=mean_squared_error(np.array([true_delta]*len(si_ates)), si_ates, squared=False)
	naive_rmse=mean_squared_error(np.array([true_delta]*len(naive_ates)), naive_ates, squared=False)


	# si_rmse=mean_squared_error(abs(np.array([true_delta]*len(si_ates))), abs(si_ates), squared=False)
	# naive_rmse=mean_squared_error(abs(np.array([true_delta]*len(naive_ates))), abs(naive_ates), squared=False)

	test_outcome_si=np.array(test_outcome_si)
	test_outcome_naive=np.array(test_outcome_naive)

	itess=np.array(itess)

	train_ctrl_r2s=np.array(train_ctrl_r2s)
	train_treat_r2s=np.array(train_treat_r2s)

	val_ctrl_r2s=np.array(val_ctrl_r2s)
	val_treat_r2s=np.array(val_treat_r2s)

	baseline_ctrl_r2s=np.array(baseline_ctrl_r2s)
	baseline_treat_r2s=np.array(baseline_treat_r2s)

	# si_vars=np.array(si_vars)
	mean_ctrls,mean_treats,var_ctrls,var_treats=np.array(mean_ctrls),np.array(mean_treats),np.array(var_ctrls),np.array(var_treats)

	threshold,emp_alpha=-1,-1
	if args.eval_alpha:
		test_function=partial(get_t_test,n1=args.arm_size*2,one_sample=True)

		try:
			threshold,emp_alpha=cv.get_search_bounds(test_function,itess,args.arm_size*2,
												args.cvl,args.cvu,args.num_search_samples,args.err_threshold,
												args.alpha,
												args.seed_end-orig_seed_start+1)
		except Exception as e:
			pass
	else:
		# load tuned threshold
		try:
			with open(os.path.join(args.results_path,"../eval_alpha_1/res_true_outcome_0.pickle"),"rb") as f:
				threshold=pickle.load(f)['threshold']
			test_function=partial(get_t_test,n1=args.arm_size*2,one_sample=True)
			# yes, misnomer but just to keep interface simple
			emp_alpha=cv.get_reject_rate(threshold,test_function,itess,args.seed_end,2*args.arm_size,args.use_unbiased_var)
		except Exception as e:
			pass
		

	if print_results:
		print("TEST with effect_size: %.5f, true_outcome: %d"%(true_delta, true_outcome))
		# print("si: RMSE: %.4f +/- (%.4f), R**2: %.4f +/- (%.4f)"%(np.mean(rmses), np.std(rmses), np.mean(r2s),np.std(r2s)))

		print("test accuracy: si: %.2f"%(np.sum(test_outcome_si==true_outcome)/len(test_outcome_si)))
		print("test_accuracy: standard: %.2f"%(np.sum(test_outcome_naive==true_outcome)/len(test_outcome_naive)))

		# print("ATE RMSE (across all seeds): si %.3f" %si_rmse)
		# print("ATE RMSE : standard %.3f" %naive_rmse)

		# print("abs(si_ate): %.3f +/- %.3f"%(np.mean(abs(si_ates)), np.std(abs(si_ates))))
		# print("abs(standard_ate): %.3f +/- %.3f" %(np.mean(abs(naive_ates)), np.std(abs(naive_ates))))

		# if args.eval_alpha:
		# 	print("critical_value: %.3f, empirical alpha: %.5f"%(threshold,emp_alpha))

	if save_results:
		with open(os.path.join(args.results_path,'res_true_outcome_%s.pickle'%true_outcome), 'wb') as f:
			pickle.dump({'true_delta': true_delta, 
						'si_ates':si_ates, 
						'naive_ates':naive_ates,
						# 'si_ate_errs':si_ate_errs,'naive_ate_errs':naive_ate_errs,
						'si_rmse':si_rmse, 
						'naive_rmse': naive_rmse,
						'test_outcome_si':test_outcome_si,
						'test_outcome_naive':test_outcome_naive,
						'true_outcome':true_outcome,
						'train_ctrl_r2s': train_ctrl_r2s,
						'train_treat_r2s': train_treat_r2s,
						'val_ctrl_r2s': val_ctrl_r2s,
						'val_treat_r2s': val_treat_r2s,
						'baseline_ctrl_r2s': baseline_ctrl_r2s,
						'baseline_treat_r2s': baseline_treat_r2s,
						'itess':itess,
						'threshold': threshold,
						'emp_alpha': emp_alpha,
						# 'si_vars':si_vars,
						'mean_ctrls':mean_ctrls,
						'mean_treats':mean_treats,
						'var_ctrls':var_ctrls,
						'var_treats':var_treats
						},
						f)

	train_r2s=np.array([train_ctrl_r2s,train_treat_r2s])
	val_r2s=np.array([val_ctrl_r2s,val_treat_r2s])
	baseline_r2s=np.array([baseline_ctrl_r2s,baseline_treat_r2s])
	train_r2s,val_r2s,baseline_r2s=np.array(train_r2s),np.array(val_r2s),np.array(baseline_r2s)


	# return itess, val_ctrl_r2s, val_treat_r2s, train_ctrl_r2s, train_treat_r2s, baseline_ctrl_r2s, baseline_treat_r2s
	return threshold, emp_alpha, itess

def add_arguments(parser):

	def parse_helper(inp_str):
		if inp_str == "None":
			return None
		return float(inp_str)

	def parse_str_helper(inp_str):
		if inp_str == "None":
			return None
		return inp_str


	parser.add_argument("--results_path",type=str,default='output/')
	parser.add_argument('--data_seeds_results_path',type=str,default=None)



	parser.add_argument('--seed_start', type=int,default=0)
	parser.add_argument('--seed_end',type=int,default=999)
	parser.add_argument('--seed_interrupt',type=int,default=-1)

	# test parameters 
	parser.add_argument('--beta',type=float,default=-1) # don't specify this (the sample size should be calculated apriori)
	parser.add_argument('--alpha',type=float,default=0.05)
	parser.add_argument('--eval_alpha',type=int,default=0)
	parser.add_argument('--use_z_test',type=int,default=0)
	parser.add_argument('--critical_value',type=parse_helper,default=None)
	parser.add_argument('--outcome_type',type=str,default='delta')
	parser.add_argument('--use_unbiased_var',type=int,default=1)

	parser.add_argument('--tune_cv_per_data_seed',type=int,default=0)
	parser.add_argument('--num_tune_seeds',type=int,default=1000)
	parser.add_argument('--data_seed_start',type=int,default=-1)
	parser.add_argument('--data_seed_end',type=int,default=-1)
	
	parser.add_argument('--cvl',type=int,default=3)
	parser.add_argument('--cvu',type=int,default=5)
	parser.add_argument('--num_search_samples',type=int,default=10)
	parser.add_argument('--err_threshold',type=float,default=1e-3)


	# data
	parser.add_argument('--cache_dir',type=parse_str_helper,default=None)
	parser.add_argument('--arm_size',type=int)
	parser.add_argument('--with_replacement',type=int,default=1)
	parser.add_argument('--include_extra',type=int,default=0)
	parser.add_argument('--feature_indices',type=parse_str_helper,default=None)
	parser.add_argument('--path_feature_scores',type=parse_str_helper,default=None)
	parser.add_argument('--topk',type=int,default=-1)
	parser.add_argument('--random_k',type=int,default=-1)
	parser.add_argument('--random_k_seed',type=int,default=0)


	# SI
	parser.add_argument('--train',type=int,default=0)
	parser.add_argument('--model_type',type=parse_str_helper,default="Ridge")
	parser.add_argument('--use_pcr',type=int,default=0)
	parser.add_argument('--reg',type=float,default=0)
	parser.add_argument('--normalize',type=int,default=1)
	parser.add_argument('--normalize_scheme',type=str,default="min_max")
	parser.add_argument('--normalize_by',type=parse_str_helper,default=None)
	parser.add_argument('--fit_intercept',type=int,default=0)
	

	parser.add_argument('--debug',type=int,default=0)
	parser.add_argument('--use_mp',type=int,default=0)
	parser.add_argument('--use_spawn',type=int,default=0)
	parser.add_argument('--num_checkpoints',type=int,default=0)
	parser.add_argument('--resume_from_state',type=int,default=0)
	parser.add_argument('--print_every',type=int,default=0)
	parser.add_argument('--save_every',type=int,default=0)


	


	return parser


	
	


