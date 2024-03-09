from scipy.stats import norm, t
import numpy as np
import math


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





