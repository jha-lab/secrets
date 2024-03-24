import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import time
import os
from collections import OrderedDict
import argparse
import syn_ctrl
import pickle



VARS_NUMERICAL=[
				'QMSubSumT',
				'QMVision','QMPtosis','QMFacial','QMSwallow','QMSubSum1',
		 		'QMSubSum2', 'QMHead', 'QMLLeg', 'QMSubSum3', 'ADTalk', 'ADChew', 'ADSwallow',
		 		'ADBreath', 'ADSubSum1', 'ADBrush', 'ADChair', 'ADEye', 'ADSubSum2', 'ADScoreT',
		 		'ADSubSumT', 'ADTotal',
		 		"HTWgtKg"
				]



def get_cache_dir():
	path_prefix="./datasets"
	if os.path.exists('/scratch/gpfs/slala/datasets'):
		path_prefix='/scratch/gpfs/slala/datasets'
	elif os.path.exists('/scratch/network/slala/datasets'):
		path_prefix='/scratch/network/slala/datasets'
	return os.path.join(path_prefix,"MGTX")

def parse_helper(inp_str):
		if inp_str == "None":
			return None
		return float(inp_str)

def parse_str_helper(inp_str):
	if inp_str == "None":
		return None
	return inp_str


def is_invalid_str(inp_str):
	inp_str=inp_str.lower()

	invalid_strs=["date", "month", "year",]

	for invalid_str in invalid_strs:
		if invalid_str in inp_str:
			return True

	return False

def is_invalid_data(data):
	def convert(e):
		try:
			return float(e.decode('utf-8'))
		except Exception:
			return np.NaN # nan/strings

	# not a single numerical entry
	if data.apply(convert).isna().all():
		return True

	# check if date type?

	return False


def is_invalid(q_str=None,var_str=None,data=None):

	if q_str:
		return is_invalid_str(q_str)

	if var_str:
		return is_invalid_str(var_str)

	return is_invalid_data(data)

def get_metrics_for_data_type(data_info, data):
	metrics_for_data_type=[]

	q_num_col=6
	q_text_col=7
	v_col=8
	for ri in np.arange(len(data_info)):
		try:
			q_num=data_info.iloc[ri,q_num_col].decode('utf-8')
			q_str=data_info.iloc[ri,q_text_col].decode('utf-8')
			var_str=data_info.iloc[ri,v_col].decode('utf-8')

			if data_info.TableName.iloc[0] == b'ClinicalAssessment': 
				var_str+="M" # similar keys but we use the MC table
				# print(var_str)

			# print(q_num,q_str,var_str)
			if is_invalid(q_str=q_str) or is_invalid(var_str=var_str) or is_invalid(data=data.loc[:,var_str]):
				# print("invalid")
				continue
			# print("valid")
			metrics_for_data_type.append(var_str)
		except Exception as e:
			# print(e)
			pass # this is not a numbered question

	assert len(set(metrics_for_data_type))==len(metrics_for_data_type)

	return metrics_for_data_type


"""
merged: full dataframe
regex: str feature name
features: None/comma delimted string
			Commas separate distinct features

keep_any_with_baseline: bool
"""
def get_metric_dfs(merged, regex, features=None, keep_any_with_baseline=False):
	metric_dfs=[]
	if features is None:
		cns=merged.filter(regex=regex).columns 
		qm_missingness_rate_by_col=[]
		for ic,cn in enumerate(cns):
			qm_missingness_rate_by_col.append(merged.loc[:,cn].isna().sum()/len(merged))
		qm_missingness_rate_by_col=np.array(qm_missingness_rate_by_col)

		missingness_rate_by_col=qm_missingness_rate_by_col

		# note if ANY of the visits have missing data (relevant for omitting)
		all_features=set()
		for f in cns:
			all_features.add(f.split("-")[0])
		all_features=np.array(list(all_features))

		# print(all_features)
		missing_features=set()
		for f in cns[qm_missingness_rate_by_col>0]:
			if keep_any_with_baseline:
				if 'vid-0' in f:
					missing_features.add(f.split("-")[0])
			else:
				missing_features.add(f.split("-")[0])
		missing_features=np.array(list(missing_features))
		
		# print(missing_features)
		features=list(np.setdiff1d(all_features,missing_features))
		
		if 'QMSubSumT' in features: features.remove('QMSubSumT')
	else:
		features=features.split(",")
		
	# print(len(features))
	# "- ensures exact match rather than common substring i.e., QMVision vs QMVisionRpt"
	metric_dfs.extend([merged.filter(regex=f+"-") for f in features])

	return metric_dfs




"""
data: 3d array [# samples, # timepoints, # features]



QMG
['QMSubSumT': outcome
 'QMVision', 
 'QMVisionRpt',  # most likely a repeat not logged anyway
 'QMPtosis', 'QMFacial', 'QMSwallow','QMSubSum1', 
 'QMRArm', 'QMLArm', # not logged
 'QMVital', # not logged
 'QMVitalRpt', # most likely repeat not logged any
 'QMRGrip', 'QMLGrip', # not logged
 'QMSubSum2', logged?
 'QMHead','QMRleg', 'QMLLeg', 'QMSubSum3', 
 'QMScoreT', # identical to QMSubSumT

columns with missing data (after removing those instances with missing wrt QMSubSumT over the target visits):
	['QMRGrip' 'QMLArm' 'QMLGrip' 'QMVisionRpt' 'QMRArm' 'QMVital'
	 'QMVitalRpt']

MG-ADL (activities daily living)
['ADTalk','ADChew','ADSwallow','ADBreath','ADSubSum1','ADBrush',
'ADChair','ADVision','ADEye','ADSubSum2',
'ADScoreT', 
'ADSubSumT', # identical to ADScoreT 
'ADTotal' # identical ^
]

missing:
['ADVision']


*_features: string, comma sep

"""

def get_complete_mgtx_dataset():
	BASELINE_VID=3
	LAST_VID=18
	VID_TO_MONTH=np.concatenate((np.arange(0,5),np.arange(6,39,3)))
	MG_PER_PC=10
	DAYS_PER_PC_VISIT=np.arange(1,167,2)
	TARGET_VIDS=np.array([3,6,7,8,9,10,11,12,13,14,15,16,17,18])


	def load_data(data_type):
		try:
			return pd.read_sas(os.path.join("datasets/MGTX",data_type+".sas7bdat"))
		except Exception:
			if os.path.exists('/scratch/network'):
				return pd.read_sas(os.path.join("/scratch/network/slala/datasets/MGTX",data_type+".sas7bdat"))
			else:
				return pd.read_sas(os.path.join("/scratch/gpfs/slala/datasets/MGTX",data_type+".sas7bdat"))

	rand=load_data("randomization")
	rand_table=load_data("randomization_table")
	data_dict=load_data("datadictionary")
	datasets={}
	data_types=[
				'mgqmg',
				'mgadl'
				# 'pillcount'
			   ]

	static_data_types=[
			'ScreeningEligibility',
			'Randomization',
			'ClinicalMedicationHistory',
			'GeneralHistoryPhysical',
			'TAC_M0',
			'TAS_M0',
			'SF36',
			'SF36Supplementary',
			'ClinicalAssessment',
			'ps1_patient_dose_diary',
			'dbo_sdmt',
				  ]

	for data_type in data_types:
		datasets[data_type]=load_data(data_type)
		
	for data_type in static_data_types:
		# we need to load the MC as this contains the M0 data
		datasets[data_type]=load_data("clinicalassessment_mc") if data_type == "ClinicalAssessment" else load_data(data_type.lower())
		
		# only keep the baseline for now
		if data_type == "ps1_patient_dose_diary": 
			datasets[data_type]=datasets['ps1_patient_dose_diary'][datasets['ps1_patient_dose_diary'].visit_id==b'3']

		
	metrics_by_data_type=OrderedDict()
	metrics_by_data_type['mgqmg']=['QMSubSumT', 
									 'QMVision', 
									 'QMVisionRpt',  
									 'QMPtosis', 'QMFacial', 'QMSwallow',
									'QMSubSum1', 'QMRArm', 'QMLArm', 'QMVital','QMVitalRpt', 'QMRGrip', 'QMLGrip',
									'QMSubSum2', 'QMHead','QMRleg', 'QMLLeg', 'QMSubSum3',
									]
	metrics_by_data_type['mgadl']=['ADTalk','ADChew','ADSwallow','ADBreath','ADSubSum1','ADBrush',
									'ADChair','ADVision','ADEye','ADSubSum2','ADScoreT','ADSubSumT', 'ADTotal']
						 
	metrics_by_static_data_type=OrderedDict()
	for dt in static_data_types:
		if dt in ['dbo_sdmt','ps1_patient_dose_diary','Randomization']: continue
		metrics_by_static_data_type[dt]=get_metrics_for_data_type(data_dict[data_dict.TableName == bytes(dt,'utf-8')],
																  datasets[dt])

	# post correct automated feature extractor

	# not all listed in data dict table
	metrics_by_static_data_type['ScreeningEligibility'].extend(['SCOnset','SCAge','SCAntibRge','SCAntibTst',
															   'SCMGClass','SCPreThym','SCPregnant',
															   'SCDisorder','SCMObesity','SCRemission',
															   'SCDose','SCPreRituxi','SCUnfit'])
	metrics_by_static_data_type['Randomization']=['RDEthty','RDImunMed']
	metrics_by_static_data_type['ClinicalMedicationHistory'].extend(['HTBioMG','HTExeComite'])
	metrics_by_static_data_type['TAC_M0'].remove("AEHyperCM")
	metrics_by_static_data_type['SF36'].append("SFYRAGO")
	metrics_by_static_data_type['ClinicalAssessment'].append("CAPEvltM")
	metrics_by_static_data_type['dbo_sdmt']=['NumCorrect','NumAnswered','M0Status','M0Memo']

	ps1_patient_dose_diary_metrics=['PSPyrids','PSAzap','PSAzapDs','PSCycp','PSCycpDs']
	ps1_patient_dose_diary_metrics.extend(['PS%d'%i for i in np.arange(1,31,2)])
	ps1_patient_dose_diary_metrics.extend(['PS%dD'%i for i in np.arange(1,31,2)])
	ps1_patient_dose_diary_metrics.extend(['PS%dP'%i for i in np.arange(1,31,2)])
	ps1_patient_dose_diary_metrics.extend(['PS%dV'%i for i in np.arange(1,31,2)])
	metrics_by_static_data_type['ps1_patient_dose_diary']=ps1_patient_dose_diary_metrics

	num_features=0
	for m,mf in metrics_by_data_type.items():
		num_features+=len(mf)
	for m,mf in metrics_by_static_data_type.items():
		num_features+=len(mf)
	# print("# features=:%d, # features*len(TARGET_VIDS)=%d"%(num_features, num_features*len(TARGET_VIDS)))

	merged={} # keys: patient ids, values: list 
	missing_ids=[] # ids not assigned treatment
	columns=['scheme']
	columns.extend(['%s-vid-%d'%(mf,vid) for m,mfs in metrics_by_data_type.items() for mf in mfs for vid in TARGET_VIDS])
	columns.extend(['%s-vid-%d'%(mf,vid) for m,mfs in metrics_by_static_data_type.items() for mf in mfs for vid in TARGET_VIDS])


	# build the data row for this subject
	for rid in rand['RDTrmtID'].unique():
		try:
			rs=rand_table[rand_table['Randomized_id']==rid].randomization_scheme.iloc[0].decode('utf-8')
			if rs not in ['TPP','PA']: 
				continue
			
			cd=[rs] 
			
			# metrics recorded across visits
			for m,mfs in metrics_by_data_type.items():
				ds=datasets[m]
				# per metric, aggregate by visit
				for mf in mfs:
	#                 for vid in np.arange(BASELINE_VID,LAST_VID+1):
					for vid in TARGET_VIDS:
						vid=bytes(str(vid),'utf-8')
						try:
							mb=float(ds[((ds.randomized_id==rid) & (ds.visit_id==vid))].loc[:,mf].iloc[0].decode('utf-8'))
						except Exception:
							mb=np.NaN
						cd.append(mb)
				
			# baseline only metrics
			for m,mfs in metrics_by_static_data_type.items():
				ds=datasets[m]
				# per metric, aggregate by visit
				for mf in mfs:
					for vid in TARGET_VIDS[:1]:
						vid=bytes(str(vid),'utf-8')
						try:
							mb=float(ds[((ds.randomized_id==rid) & (ds.visit_id==vid))].loc[:,mf].iloc[0].decode('utf-8'))
						except Exception:
							mb=np.NaN
					cd.extend([mb]*len(TARGET_VIDS)) # replicate value across visits to avoid inflating rank of data (mRSC SVD denoising step)
						
			merged[rid]=cd
			
		except Exception as e:
	#         print(e)
			missing_ids.append(rid)


	merged_complete=pd.DataFrame.from_dict(merged,orient='index',columns=columns)
	merged=merged_complete[merged_complete.filter(regex="QMSubSumT-").isna().sum(axis=1)==0] # drop any subject with any missing visit data from the outcome metric
	# merged=merged.fillna(-1) # other metrics  
	assert merged.filter(regex="QMSubSumT-").isna().sum().sum()==0
	# print(len(merged),(merged.scheme=="PA").sum())

	all_features=[] # not efficient vs set but to remove randomneess
	for f in merged.iloc[:,1:].columns: # skip scheme
		if f.split("-")[0] not in all_features:
			all_features.append(f.split("-")[0])

	keep_features=[]
	for f in all_features:
		# if the feature is missing data across any visit across subjects, skip over for now
		if merged.filter(regex=f+"-").apply(lambda s: s.isna().sum()/len(s)).sum()>0: continue
		keep_features.append(f)
		
	# print("len(keep_features): %d"%len(keep_features))
	merged=merged.iloc[:,:1].join([merged.filter(regex=f+"-") for f in keep_features])         
	assert merged.notna().all(None) 

	return merged, np.array(keep_features)


def is_numeric(fn):
	return fn in VARS_NUMERICAL

def handle_categorical(data, feature_names, drop_categ=True):
	num_feature_names=[c for c in feature_names if is_numeric(c)]
	ind_num_features=np.array([np.where(feature_names==nf)[0][0] for nf in num_feature_names])
	return data[:,:,ind_num_features], num_feature_names

"""
feature_indices: rel. to keep_features
keep_features: excludes the outcome feature
"""
def get_dataset(include_extra=0, feature_indices=None, merged=None, keep_features=None, drop_categ=True):
	TARGET_VIDS=np.array([3,6,7,8,9,10,11,12,13,14,15,16,17,18])

	if os.path.exists(os.path.join(get_cache_dir(),"merged.pickle")):
		with open(os.path.join(get_cache_dir(),"merged.pickle"),"rb") as f:
			res=pickle.load(f)
			merged,keep_features=res['merged'],res['keep_features']
	else: 
		merged, keep_features=get_complete_mgtx_dataset()
		with open(os.path.join(get_cache_dir(),"merged.pickle"),"wb") as f: 
			pickle.dump({'merged':merged,'keep_features':keep_features},f)

	iof=np.where(keep_features=='QMSubSumT')[0]
	keep_features=np.delete(keep_features,iof)

	# select the metrics
	metric_dfs=[merged.filter(regex="QMSubSumT")]
	feature_names=['QMSubSumT']
	ind_outcome_ft=0

	if include_extra:
		feature_indices=feature_indices if feature_indices is not None else np.arange(len(keep_features))
		for ikf in feature_indices:
			metric_dfs.extend(get_metric_dfs(merged, keep_features[ikf]))
			feature_names.append(keep_features[ikf])

	merged=merged.iloc[:,:1].join(metric_dfs)
	assert merged.notna().all(None)

	labels,data=list(merged.iloc[:,0].values),merged.iloc[:,1:].values
	labels=np.array([label=="TPP" for label in labels])
	sids=np.arange(len(data))

	data=data.reshape(len(data),len(TARGET_VIDS),-1,order='F')

	feature_names=np.array(feature_names)
	if drop_categ:
		data,feature_names=handle_categorical(data, feature_names, drop_categ)

	
	return [data,labels,sids], feature_names, ind_outcome_ft




def add_arguments(parser):
	parser.add_argument('--drop_categ',type=int,default=1)
	syn_ctrl.add_arguments(parser)
	return parser

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser=add_arguments(parser)
	args=parser.parse_args()
		
	syn_ctrl.main_with_data_seeds(args, *get_dataset(args.include_extra, args.feature_indices, drop_categ=args.drop_categ))
	


