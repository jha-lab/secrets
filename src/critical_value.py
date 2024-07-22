import pickle
import os
import numpy as np
import argparse
import sys
sys.path.append("..")
import src
from src.utils import get_t_test
from functools import partial
from scipy.stats import t, bootstrap
import scipy.stats


def get_reject_rate(cv, test_function, itess, num_seeds, size, use_unbiased_var=1):
	o=0
	test=test_function(cv=cv)
	for s in np.arange(num_seeds):
		sitess=itess[size*s:size*(s+1)]
		o+=test(np.mean(sitess),np.var(sitess,ddof=use_unbiased_var))

	# assert size*(s+1) == len(itess)
	
	return o/(num_seeds)

def get_reject_rate_by_data_seed(cvs, test_function, itess, num_seeds, size, use_unbiased_var=1):
	o=0
	for s in np.arange(num_seeds):
		test=test_function(cv=cvs[s])
		sitess=itess[size*s:size*(s+1)]
		o+=test(np.mean(sitess),np.var(sitess,ddof=use_unbiased_var))

	assert size*(s+1) == len(itess)
	
	return o/(num_seeds)

def get_search_bounds(test_function,itess,size,lower,upper,num_search_samples=10,err_threshold=1e-7,target_alpha=0.05,num_seeds=1000,
						use_unbiased_var=1):

	cvs=np.linspace(lower, upper, num=num_search_samples)
	alphas_by_cv=[]
	for cv in cvs:
		cv_alpha=get_reject_rate(cv, test_function, itess, num_seeds,size, use_unbiased_var)
		if abs(cv_alpha-target_alpha)<=err_threshold:
			return cv, cv_alpha
		alphas_by_cv.append(cv_alpha)
	alphas_by_cv=np.array(alphas_by_cv)

	# print(lower,upper, alphas_by_cv[0],alphas_by_cv[-1])

	# this is we don't have a wide enough initial search space?
	if len(cvs)==1:
		return cvs[0], alphas_by_cv[0]

	alpha_threshold=alphas_by_cv[len(alphas_by_cv)//2]
	if target_alpha<alpha_threshold: 
		lower=cvs[(len(alphas_by_cv)//2) + 1]
		if target_alpha<alphas_by_cv[-1]: upper=upper+2 
	else:
		upper=cvs[(len(alphas_by_cv)//2) + 1]
		if target_alpha>alphas_by_cv[0]: lower=lower-2

	return get_search_bounds(test_function,itess,size,lower,upper,num_search_samples,err_threshold,target_alpha,num_seeds,use_unbiased_var)
	

def run_bootstrap_test(itess, num_seeds, size, alpha, null_delta=0, n_resamples=9999):
	o=0

	dist=np.zeros(n_resamples) # default bootstraps

	for s in np.arange(num_seeds):

	    ites=itess[size*s:size*(s+1)]
	    
	#     ts=time.time()
	    res=bootstrap((ites,),lambda sample: np.mean(sample), vectorized=False, random_state=0, n_resamples=n_resamples,confidence_level=1-alpha)
	#     print(time.time()-ts)
	    o+=(null_delta < res.confidence_interval.low or null_delta > res.confidence_interval.high)
	    dist=dist+res.bootstrap_distribution

	return o/num_seeds, dist/num_seeds



# def get_data_seed_cv(ites, num_bootstraps, use_unbiased_var, alpha):
# 	null_dist=[]
# 	n1=len(ites)
# 	for b in np.arange(num_bootstraps):
# 		ites_sampled=np.random.choice(ites,len(ites),replace=True)
# 		sample_mean=np.mean(ites_sampled)
# 		sample_var=np.var(ites_sampled,ddof=use_unbiased_var)
# 		stat=abs(sample_mean/((sample_var/n1)**0.5))
# 		null_dist.append(stat)
# 	null_dist=np.array(null_dist)
# 	hist,bins=np.histogram(null_dist)
# 	cdf=np.cumsum(hist)
# 	cdf=cdf/num_bootstraps
# 	return bins[:-1][cdf>=1-alpha][0]

# def get_data_seed_cvs(res, num_seeds, size, num_bootstraps, use_unbiased_var, alpha):

# 	cvs=[]
# 	for data_seed in np.arange(num_seeds):
# 		ites=res['itess'][data_seed*size:(data_seed+1)*size]
# 		cvs.append(get_data_seed_cv(ites, num_bootstraps, use_unbiased_var, alpha))

# 	return cvs

def main(args):
	# print("critical_value.py")

	os.makedirs(args.results_path,exist_ok=True)


	if args.use_data_seed_cvs:

		num_data_seeds=args.data_seed_end-args.data_seed_start+1

		
		if args.load_data_seed_cvs:
			cvs=[]
			for data_seed in np.arange(args.data_seed_start,args.data_seed_end+1):
				data_seed_dir=os.path.join(args.data_seed_prefix_dir,"data_seed_%d/eval_alpha_1"%data_seed)
				with open(os.path.join(data_seed_dir,"res_true_outcome_0.pickle"),'rb') as f:
					cvs.append(pickle.load(f)['threshold'])
			cvs=np.array(cvs)
		else: # calculate
			pass
			# try:
			# 	with open(os.path.join(args.prefix_file,'res_true_outcome_0.pickle'),"rb") as f:
			# 		res=pickle.load(f)
			# except:
			# 	with open(os.path.join(args.prefix_file,'eval_alpha_1/res_true_outcome_0.pickle'),"rb") as f:
			# 		res=pickle.load(f)

			# cvs=get_data_seed_cvs(res, num_data_seeds, args.arm_size*2, args.num_bootstraps, args.use_unbiased_var, args.target_alpha)

		test_function=partial(get_t_test,n1=args.arm_size*2,one_sample=True, alpha=args.target_alpha)

		with open(os.path.join(args.prefix_file,'eval_alpha_1/res_true_outcome_0.pickle'),'rb') as f:
			res=pickle.load(f)
		alpha=get_reject_rate_by_data_seed(cvs,test_function,res['itess'],num_data_seeds,args.arm_size*2,args.use_unbiased_var)

		with open(os.path.join(args.prefix_file,'eval_alpha_0/res_true_outcome_1.pickle'),'rb') as f:
			res=pickle.load(f)
		power=get_reject_rate_by_data_seed(cvs,test_function,res['itess'],num_data_seeds,args.arm_size*2,args.use_unbiased_var)

		with open(os.path.join(args.results_path,'seed_specific_critical_value_%d.pickle'%args.load_data_seed_cvs),'wb') as f:
			pickle.dump({'cvs':cvs,'alpha':alpha,'power':power},f)

		print("seed-specific: critical value: %.5f +/- %.5f: alpha: %.5f, power: %.5f"\
				%(np.mean(cvs),np.std(cvs,ddof=1),alpha,power))

	elif args.use_bootstrap:
		try:
			with open(os.path.join(args.prefix_file,'res_true_outcome_0.pickle'),"rb") as f:
				null_res=pickle.load(f)
			with open(os.path.join(args.prefix_file,'res_true_outcome_1.pickle'),"rb") as f:
				alt_res=pickle.load(f)
		except:
			with open(os.path.join(args.prefix_file,'eval_alpha_1/res_true_outcome_0.pickle'),"rb") as f:
				null_res=pickle.load(f)
			with open(os.path.join(args.prefix_file,'eval_alpha_0/res_true_outcome_1.pickle'),"rb") as f:
				alt_res=pickle.load(f)


		alpha,null_dist=run_bootstrap_test(null_res['itess'],args.num_seeds,args.arm_size*2,args.target_alpha,n_resamples=args.n_resamples)
		power,alt_dist=run_bootstrap_test(alt_res['itess'],args.num_seeds,args.arm_size*2,args.target_alpha,n_resamples=args.n_resamples)

		shift_factor=abs(np.mean(alt_dist)-np.mean(null_dist))
		shift_factor/=np.mean([np.std(alt_dist,ddof=1),np.std(null_dist,ddof=1)])
		est_power=scipy.stats.norm.cdf(-1*scipy.stats.norm.ppf(1-(args.target_alpha/2))+shift_factor)

		with open(os.path.join(args.results_path,"bootstrap_test.pickle"),"wb") as f:
			pickle.dump({'alpha': alpha ,'power': power, 'null_dist': null_dist, 'alt_dist': alt_dist, 'shift_factor': shift_factor, 'est_power':est_power},f)

		print("alpha: %.5f, power: %.5f, sep. bet. means: %5f, avg. stdev: %.5f +/- %.5f, shift_factor: %.5f, est_power: %.5f"\
																	%(alpha,power, 
																	abs(np.mean(alt_dist)-np.mean(null_dist)), 
																	np.mean([np.std(alt_dist,ddof=1),np.std(null_dist,ddof=1)]),
																	np.std([np.std(alt_dist,ddof=1),np.std(null_dist,ddof=1)],ddof=1),
																	shift_factor,
																	est_power))




	else:

		try:
			with open(os.path.join(args.prefix_file,'res_true_outcome_0.pickle'),"rb") as f:
				res=pickle.load(f)
		except:
			with open(os.path.join(args.prefix_file,'eval_alpha_1/res_true_outcome_0.pickle'),"rb") as f:
				res=pickle.load(f)

		# note by our convention, it arm_size*2 due to the stacking of the data

		test_function=partial(get_t_test,n1=args.arm_size*2,one_sample=True, alpha=args.target_alpha)
		
		try:
			cv,alpha=get_search_bounds(test_function,res['itess'],args.arm_size*2,
												args.cvl,args.cvu,args.num_search_samples,args.err_threshold,args.target_alpha,
												args.num_seeds, args.use_unbiased_var)
		except Exception as e:
			cv,alpha=-1,-1
		
		alpha_t=get_reject_rate(None,test_function,res['itess'],args.num_seeds,args.arm_size*2,args.use_unbiased_var)
		cv_t=t.ppf(1-(args.target_alpha/2),args.arm_size*2-1)

		power,power_t=-1,-1
		if args.eval_alpha_0:
			try:
				with open(os.path.join(args.prefix_file,'res_true_outcome_1.pickle'),"rb") as f:
					res=pickle.load(f)
			except:
				with open(os.path.join(args.prefix_file,'eval_alpha_0/res_true_outcome_1.pickle'),"rb") as f:
					res=pickle.load(f)


			power=get_reject_rate(cv,test_function,res['itess'],args.num_seeds,args.arm_size*2,args.use_unbiased_var)

			power_t=get_reject_rate(None,test_function,res['itess'],args.num_seeds,args.arm_size*2,args.use_unbiased_var)


		with open(os.path.join(args.results_path,"critical_value.pickle"),"wb") as f:
			pickle.dump({'critical_value':cv, 'alpha': alpha ,'power': power, 
						'critical_value_t':cv_t, 'alpha_t':alpha_t, 'power_t':power_t},f)

		print("critical_value (modified): %.5f, alpha (modified): %.5f, power (modified): %.5f, \
				critical_value (t-test): %.5f, alpha (t-test): %.5f, power (t-test): %.5f"\
				%(cv, alpha, power, cv_t, alpha_t, power_t))

		return np.array([cv, alpha, power, cv_t, alpha_t, power_t])


def add_arguments(parser):
	parser.add_argument('--prefix_file',type=str)
	parser.add_argument('--data_seed_prefix_dir',type=str,default=None)
	parser.add_argument('--results_path',type=str)

	# test params
	parser.add_argument('--num_seeds',type=int,default=1000)
	parser.add_argument('--arm_size',type=int)
	parser.add_argument('--eval_alpha_0',type=int,default=1)
	parser.add_argument('--use_unbiased_var',type=int,default=1)
	parser.add_argument('--use_data_seed_cvs',type=int,default=1)
	parser.add_argument('--load_data_seed_cvs',type=int,default=1)
	parser.add_argument('--num_bootstraps',type=int,default=1000) # deprecated
	parser.add_argument('--data_seed_start',type=int,default=-1)
	parser.add_argument('--data_seed_end',type=int,default=-1)

	parser.add_argument('--use_bootstrap',type=int,default=0)
	parser.add_argument('--n_resamples',type=int,default=9999)

	# search params
	parser.add_argument('--target_alpha',type=float,default=0.05)
	parser.add_argument('--err_threshold',type=float,default=1e-7)
	parser.add_argument('--cvl',type=float,default=3)
	parser.add_argument('--cvu',type=float,default=5)
	parser.add_argument('--num_search_samples',type=int,default=10)



	return parser

if __name__ == "__main__":
	parser=argparse.ArgumentParser()

	parser=add_arguments(parser)

	main(parser.parse_args())


