import pandas as pd
import numpy as np
import os
import pickle
import argparse
import syn_ctrl



VARS_IGNORE=["calc_score-ufugm_cat",

'demo_screen-center', 'demo_screen-severity',
'demo_screen-from_onset', 'demo_screen-strata',
'demo_screen-source', 'demo_screen-concordance',
'demo_screen-Language', 'demo_screen-prerandOT',

'demo_screen-CSgender', 'demo_screen-CSstrkloc',
'demo_screen-Cssidehemi', 'demo_screen-CScancer',
'demo_screen-CSchemo', 'demo_screen-CSradi', 'demo_screen-CSrenal',
'demo_screen-Cspace', 'demo_screen-Csoxygen',
'demo_screen-CSaccuracy', 'demo_screen-DEMOEthnicity',
'demo_screen-DEMORace', 'demo_screen-DEMOhand',
'demo_screen-DEMOmarital', 'demo_screen-DEMOLiveWith',
'demo_screen-DEMOrelation', 'demo_screen-DEMOoccupt',
'demo_screen-DEMOsmoke', 'demo_screen-DEMOedu',
'demo_screen-DEMOPrimVer', 'demo_screen-DEMOPrimRead',
'demo_screen-DEMOFluentVer', 'demo_screen-DEMOFluentRead',
'demo_screen-DEMOtransVer', 'demo_screen-DEMOtransWritten',

'eval-SISSTRARM', 'eval-SISSTRGRIP', 'eval-SISSTRLEG',
'eval-SISSTRFOOT', 'eval-SISRTOLD', 'eval-SISRDAYBEF',
'eval-SISRTHINGS', 'eval-SISRDWEEK', 'eval-SISTHINKQIK',
'eval-SISSLVPROB', 'eval-SISFSAD', 'eval-SISFNCLOSE',
'eval-SISFBURDEN', 'eval-SISFLOOKFWD', 'eval-SISBLAME',
'eval-SISENJOY', 'eval-SISFNERVOUS', 'eval-SISFWLIVING',
'eval-SISSMILE', 'eval-SISSAYNAME', 'eval-SISUNDSAID',
'eval-SISREPLYQ', 'eval-SISNAMEOBJ', 'eval-SISCVGROUP',
'eval-SISCVTELE', 'eval-SISPHONEDI', 'eval-SISDCUTFOOD',
'eval-SISDDRESS', 'eval-SISDBATHE', 'eval-SISDTOILET',
'eval-SISDBLADDER', 'eval-SISDBOWELS', 'eval-SISDLCHORES',
'eval-SISDHCHORES', 'eval-SISDSITBAL', 'eval-SISDSTDBAL',
'eval-SISDWKBAL', 'eval-SISDMOVEBC', 'eval-SISDWKBLOCK',
'eval-SISDWKFAST', 'eval-SISD1FSTAIR', 'eval-SISD2FSTAIR',
'eval-SISDIOCAR', 'eval-SISSCARRYHV', 'eval-SISSTURN',
'eval-SISSOPENJAR', 'eval-SISSTIESHOE', 'eval-SISSPICKD',
'eval-SISLWORK', 'eval-SISLSOCIAL', 'eval-SISLQREC',
'eval-SISLAREC', 'eval-SISLROLEFAM', 'eval-SISLRELIG',
'eval-SISLCTRLLIF', 'eval-SISLHELP', 'eval-SISRECOVERY',


'eval-FM_BICEP', 'eval-FM_TRICEP',
'eval-FM_FS_RET', 'eval-FM_FS_ELV', 'eval-FM_FS_ABD',
'eval-FM_FS_EXT', 'eval-FM_FS_ELF', 'eval-FM_FS_SUP',
'eval-FM_ES_SHAD', 'eval-FM_ES_EXT', 'eval-FM_ES_FPR',
'eval-FM_MS_HAND', 'eval-FM_MS_SHF', 'eval-FM_MS_PSUP',
'eval-FM_MOS_SAB', 'eval-FM_MOS_SFL', 'eval-FM_MOS_PRO',
'eval-FM_NR', 'eval-FM_W_SE9', 'eval-FM_W_FE9', 'eval-FM_W_SE3',
'eval-FM_W_FE3', 'eval-FM_W_CIR', 'eval-FM_H_FMF', 'eval-FM_H_FME',
'eval-FM_H_GRASP1', 'eval-FM_H_GRASP2', 'eval-FM_H_GRASP3',
'eval-FM_H_GRASP4', 'eval-FM_H_GRASP5', 'eval-FM_CS_TRE',
'eval-FM_CS_DYS', 'eval-FM_CS_SPE', 'eval-fm_totalscore19',
'eval-FM_totalscore58', 'eval-LIFEideal', 'eval-LIFEconditions',
'eval-LIFEsatisfied', 'eval-LIFEimportant', 'eval-LIFEchange',
'eval-AMTpatsize', 'eval-AMTwfLess1', 'eval-AMTwfLess2',
'eval-AMTwfMore1', 'eval-AMTweMore1', 'eval-AMTefMore1',
'eval-AMTefMore2', 'eval-ATfirst', 

'eval-EQ5Dmobility', 'eval-EQ5Dselfcare',
'eval-EQ5Dactivity', 'eval-EQ5Dpain', 'eval-EQ5Danxiety',
'eval-EQ5Dscale', 'eval-RNLIquarters', 'eval-RNLItrip',
'eval-RNLIselfcare', 'eval-RNLIwork', 'eval-RNLIrecreational',
'eval-RNLIsocial', 'eval-RNLIrole', 'eval-RNLIrelation',
'eval-RNLIcompany', 'eval-RNLIlifeevents'

]




VARS_NUMERICAL_IGNORE=[


# sfMore

# 'calc_score-FD4Q6_ME',
# 'calc_score-FD4Q7_EM', 'calc_score-FD4Q8_CO',
# 'calc_score-FD4Q9_AC', 'calc_score-FD4Q10_MO',
# 'calc_score-FD4Q11_HA', 'calc_score-FD4Q12_HC',
# 'calc_score-FD4_SIS16', 'calc_score-LIFEscore',
# 'calc_score-wmft_mean_time_MA_PA',
# 'calc_score-wmft_mean_time_LA_PA',
# 'calc_score-log_mean_time_MA_PA', 'calc_score-log_mean_time_LA_PA',
# 'calc_score-grip_MA',

'eval-WMFTtime_weightbox1L',
'eval-WMFTtime_grip1L', 'eval-WMFTtime_grip2L',
'eval-WMFTtime_grip3L', 'eval-AMTwfLess3', 'eval-AMTwfMore2',
'eval-AMTwfMore3', 'eval-AMTweMore2', 'eval-AMTweMore3',
'eval-AMTefMore3', 'eval-AMTsfMore1', 'eval-AMTseMore1',


'demo_screen-FM_FS_ELV', 'demo_screen-FM_ES_EXT',
'demo_screen-FM_MS_PSUP', 'demo_screen-FM_W_FE9',
'demo_screen-FM_H_FME', 'demo_screen-FM_H_GRASP2',
'demo_screen-FMFS_totalscore', 'demo_screen-FMFS_totalscorevs58',
'eval-WMFTtime_pencilL', 'eval-CHAMweakhand', 'eval-CHAMdominant',
'eval-CHAMsignature', 'eval-CHAMcutfood', 'eval-CHAMpapertowel',
'eval-CHAMpractice', 'eval-CHAMcarryfood', 'eval-CHAMsmalljar',
'eval-CHAMpressbotton', 'eval-CHAMchallenge',
'eval-CHAMpapermatch', 'eval-CHAMtelebook', 'eval-CHAMquick',
'eval-CHAMdelicate', 'eval-CHAMchinaplatter', 'eval-CHAMcoffee',
'eval-CHAMpotholders', 'eval-CHAMsoftsofa', 'eval-CHAMwrapping',
'eval-CHAMcereal',

'demo_screen-FCtotalOT', 'demo_screen-onset_to_rand',
'demo_screen-age', 'eval-WMFT_armtested',
'eval-WMFTtime_foretableR', 'eval-WMFTtime_foreboxR',
'eval-WMFTtime_elbowsideR', 'eval-WMFTtime_elbowwtR',
'eval-WMFTtime_handtableR', 'eval-WMFTtime_handboxR',
'eval-WMFTtime_reachR', 'eval-WMFTtime_canR',
'eval-WMFTtime_pencilR', 'eval-WMFTtime_paperclipR',
'eval-WMFTtime_checkersR', 'eval-WMFTtime_flipcardsR',
'eval-WMFTtime_keyR', 'eval-WMFTtime_towelR',
'eval-WMFTtime_foretableL', 'eval-WMFTtime_foreboxL',
'eval-WMFTtime_elbowsideL', 'eval-WMFTtime_elbowwtL',
'eval-WMFTtime_handtableL', 'eval-WMFTtime_handboxL',
'eval-WMFTtime_reachL', 'eval-WMFTtime_canL',
'eval-WMFTtime_paperclipL', 'eval-WMFTtime_checkersL',
'eval-WMFTtime_flipcardsL', 'eval-WMFTtime_keyL',
'eval-WMFTtime_towelL', 'eval-WMFTtime_basketL', 'eval-AT1R',
'eval-AT2R', 'eval-AT3R'
]

def get_cache_dir():
	path_prefix="./datasets"
	if os.path.exists('/scratch/gpfs/slala/datasets'):
		path_prefix='/scratch/gpfs/slala/datasets'
	elif os.path.exists('/scratch/network/slala/datasets'):
		path_prefix='/scratch/network/slala/datasets'
	return os.path.join(path_prefix,"ICARE")


def get_complete_dataset(outcome_metric="FD4Q8_CO"):
	path_prefix="./datasets"
	if os.path.exists('/scratch/gpfs/slala/datasets'):
		path_prefix='/scratch/gpfs/slala/datasets'
	elif os.path.exists('/scratch/network/slala/datasets'):
		path_prefix='/scratch/network/slala/datasets'

	dfs=pd.read_excel(os.path.join(path_prefix,"ICARE/ICARE dataset_07132016.xlsx"),[0,1,2])
	calc_score_df,demo_df,eval_df=dfs[0],dfs[1],dfs[2]

	trt_df=demo_df[demo_df.trt_group.isin(['A','D'])] 
	keep_pids=[]
	for pid in trt_df.id:
		pid_df=calc_score_df[calc_score_df.id==pid].sort_values(by="visitnum")

		if not (1 in pid_df.visitnum.tolist() and 4 in pid_df.visitnum.tolist()): continue

		metric_df=pid_df.loc[:,["visitnum",outcome_metric]]

		# if metric_df.loc[:,metric1].isna().any(): continue
		if np.isnan(metric_df[metric_df.visitnum==1].loc[:,outcome_metric].iloc[0]) or \
			np.isnan(metric_df[metric_df.visitnum==4].loc[:,outcome_metric].iloc[0]):
			continue

		keep_pids.append(pid)
	keep_pids=np.array(keep_pids)


	complete_metrics=[outcome_metric+"-vid-1",outcome_metric+"-vid-4"]
	# prefix due to identical measures across sheets i.e., demo vs eval (screening data)
	calc_score_metrics=["calc_score-%s"%m for m in calc_score_df.columns[2:] if m != outcome_metric] # skip id/visit; excludes outcome metric
	demo_metrics=["demo_screen-%s"%m for m in demo_df.columns[1:]] 
	eval_metrics=["eval-%s"%m for m in eval_df.columns[2:]]
	complete_metrics.extend(calc_score_metrics)
	complete_metrics.extend(demo_metrics)
	complete_metrics.extend(eval_metrics)
	complete_metrics=np.array(complete_metrics)


	baseline_calc_score_df,baseline_demo_df,baseline_eval_df=calc_score_df[calc_score_df.visitnum==1],demo_df,eval_df[eval_df.visitnum==1]

	data_by_pid={}
	for pid in keep_pids:
		data=[]
		
		outcome_df=calc_score_df[calc_score_df.id==pid].sort_values(by="visitnum").loc[:,["visitnum",outcome_metric]]
		data.append(outcome_df[outcome_df.visitnum==1].loc[:,outcome_metric].iloc[0])
		data.append(outcome_df[outcome_df.visitnum==4].loc[:,outcome_metric].iloc[0])
		
		for m in calc_score_metrics:
			data.append(baseline_calc_score_df[baseline_calc_score_df.id==pid].loc[:,m.split("-")[-1]].iloc[0])
		for m in demo_metrics:
			data.append(baseline_demo_df[baseline_demo_df.id==pid].loc[:,m.split("-")[-1]].iloc[0])
		for m in eval_metrics:
			data.append(baseline_eval_df[baseline_eval_df.id==pid].loc[:,m.split("-")[-1]].iloc[0])
		data_by_pid[pid]=data


	complete_df=pd.DataFrame.from_dict(data_by_pid,'index',columns=complete_metrics).replace(".",np.NaN)

	return complete_df



def simple_impute(s):
	if not s.isna().any():
		return s

	s=s.copy()

	is_cat=True
	s_sans_na=s[s.notna()]
	for e in s_sans_na.tolist():
		if e%1 != 0: is_cat=False
	
	# print(is_cat)

	if is_cat:
		val=s.mode().iloc[0] # choose the first if multiple modes
	else:
		val=s.median()

	s[s.isna()]=val

	return s


def rsc_impute_df(df):
	pass



def get_nonmissing_df(complete_df, missing_method="drop", missing_threshold=0.1):
	mrs=[]
	for ci,c in enumerate(complete_df.columns):
		mrs.append(complete_df.iloc[:,ci].isna().sum()/len(complete_df))
	mrs=np.array(mrs)

	if missing_method == "drop":
		return complete_df.iloc[:,np.where(mrs==0)[0]]

	# impute

	complete_df=complete_df.iloc[:,np.where(mrs<=missing_threshold)]

	if missing_method == "simple":
		return complete_df.apply(simple_impute)

	if missing_method == "rsc":
		return rsc_impute_df(complete_df)


def convert_float(complete_df):
	# categorical data conversion
			
	def get_label_from_onset(r):
		if r.iloc[0]=="E" and r.iloc[1] == "H":
			return 1
		if r.iloc[0]=="L" and r.iloc[1]=="H":
			return 2
		if r.iloc[0]=="E" and r.iloc[1]=="L":
			return 3
		return 4
		
	complete_df.loc[:,'demo_screen-trt_group']=complete_df.loc[:,'demo_screen-trt_group'].apply(lambda e: e=="A").astype(int)
	complete_df.loc[:,'demo_screen-from_onset']=complete_df.loc[:,['demo_screen-from_onset','demo_screen-severity']].apply(lambda r: get_label_from_onset(r),1).astype(int)
	complete_df.loc[:,'demo_screen-severity']=complete_df.loc[:,'demo_screen-severity'].apply(lambda e: e=="H").astype(int)

	return complete_df.astype(float)

def get_dataset_array(complete_df, outcome_metric):
	features=[outcome_metric]
	for c in complete_df.columns:
		if c in ["demo_screen-trt_group","%s-vid-1"%outcome_metric,"%s-vid-4"%outcome_metric]:
			continue
		features.append(c)
	features=np.array(features)
	ind_outcome_ft=0

	full_data,full_labels=[],[]         
	full_sids=np.arange(len(complete_df))
	for pid in full_sids:
		data=[]
		pid_df=complete_df.iloc[pid]
		
		data.append([pid_df.loc["%s-vid-1"%outcome_metric],pid_df.loc["%s-vid-4"%outcome_metric]])
		
		for c in complete_df.columns:
			if c in ["demo_screen-trt_group","%s-vid-1"%outcome_metric,"%s-vid-4"%outcome_metric]:
				continue
			data.append([pid_df.loc[c],0])
		
		full_data.append(data)
		full_labels.append(int(pid_df.loc["demo_screen-trt_group"]))
		
	full_data,full_labels=np.array(full_data).transpose(0,2,1),np.array(full_labels)

	return [full_data,full_labels,full_sids], features, ind_outcome_ft

		

def is_categ(c,s):
	return c in VARS_IGNORE

def handle_categorical(complete_df, drop_categ=True, drop_extra_numerical=True):
	categ_cols=[c for c in complete_df.columns if (is_categ(c, complete_df.loc[:,c]))]
	if drop_extra_numerical: 
		categ_cols.extend([c for c in complete_df.columns if c in VARS_NUMERICAL_IGNORE])
	return complete_df.drop(columns=categ_cols)



def get_dataset(outcome_metric="FD4Q8_CO",
				missing_method="drop",missing_threshold=None,
				include_extra=0,feature_indices=None,
				drop_categ=True, drop_extra_numerical=True):

	# print(outcome_metric)
	cache_dir=get_cache_dir()
	if os.path.exists(os.path.join(cache_dir,'%s_complete_df.pickle'%outcome_metric)):
		# print("loaded")
		with open(os.path.join(cache_dir,'%s_complete_df.pickle'%outcome_metric),'rb') as f:
			complete_df=pickle.load(f)
	else:
		# print("computing")
		complete_df=get_complete_dataset(outcome_metric)
		with open(os.path.join(cache_dir,'%s_complete_df.pickle'%outcome_metric),'wb') as f:
			pickle.dump(complete_df,f)

	complete_df=convert_float(complete_df)
	complete_df=get_nonmissing_df(complete_df, missing_method, missing_threshold)

	assert not complete_df.isna().any().any()

	if drop_categ:
		complete_df=handle_categorical(complete_df,drop_categ, drop_extra_numerical)

	[full_data,full_labels,full_sids],features,iof=get_dataset_array(complete_df, outcome_metric)

	if feature_indices is None: 
		if include_extra:
			feature_indices=np.arange(len(features))
		else:
			feature_indices=np.array([iof])
	else:
		if iof not in feature_indices: feature_indices.insert(0,iof)
		feature_indices=np.array(feature_indices)

	return [full_data[:,:,feature_indices],full_labels,full_sids],features[feature_indices],iof





def add_arguments(parser):
	parser.add_argument('--drop_categ',type=int,default=1)
	parser.add_argument('--drop_extra_numerical',type=int,default=1)
	parser.add_argument('--outcome_metric',type=str,default="FD4Q8_CO")

	syn_ctrl.add_arguments(parser)
	return parser

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser=add_arguments(parser)
	args=parser.parse_args()

	# if args.topk != -1: # choose topk
	# 	with open(os.path.join(args.path_feature_scores),"rb") as f:
	# 		feature_scores=pickle.load(f)['scores_by_fi'][:,0]
	# 	ind_sort=np.argsort(feature_scores)[::-1]
	# 	feature_indices=ind_sort[:args.topk]
	# 	args.feature_indices=",".join(feature_indices.astype(str))
	# 	print(args.feature_indices)

	if args.feature_indices is not None:
		feature_indices=[int(fi) for fi in args.feature_indices.split(",")]
	else:
		feature_indices=None
		
	full_ds,features,ind_outcome_ft=get_dataset(args.outcome_metric,include_extra=args.include_extra,feature_indices=feature_indices, 
									drop_categ=args.drop_categ, drop_extra_numerical=args.drop_extra_numerical)

	if args.tune_cv_per_data_seed:
		syn_ctrl.main_with_data_seeds(args, full_ds, features, ind_outcome_ft)
	else:
		syn_ctrl.main(args, full_ds, features, ind_outcome_ft)



