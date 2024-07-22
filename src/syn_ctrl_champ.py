import pandas as pd
import numpy as np
import os
import pickle
import argparse
import sys 
sys.path.append("..")
import src
from src import syn_ctrl
from collections import OrderedDict
from dataset import parse_str_helper




# categorical vars with >1 level

VARS_IGNORE=[
					'saeevaluation-Serious', 
					'saeevaluation-Related',
					'saeevaluation-Expected', 
					'ecgoverread-ECGResults',
					'medicalhistory-BirthCtrl',
					'demographics-GENDER', 'demographics-ETHNICITY',
					'demographics-EDUMOTHER', 'demographics-EDUFATHER',
					'demographics-HOUSEINCOME', 'demographics-RECRUITMETHOD',
					'acuteheadacheconmed-MedName', 'acuteheadacheconmed-EndDTDays',
					'historymigrainebaseline-ActivityLevel',
					'historymigrainebaseline-ActivityWorse',
					'historymigrainebaseline-Ponytail', 'historymigrainebaseline-Hat',
					'historymigrainebaseline-Glasses',
					'historymigrainebaseline-HeadacheSide',
					'historymigrainebaseline-Severity',
					'historymigrainebaseline-TightClothing',
				]


VARS_NUMERICAL=[
				"brief_di-brief_t",
				"cdi_di-cdi_tscore",
				"demographics-BIRTHDTDays",
				"fdichild-FDItot",
				"fdiparent-FDItot",
				"hads-anxiety",
				"hads-depression",
				"historymigrainebaseline-AverageHeadache",
				"historymigrainebaseline-BadHeadacheDays",
				"historymigrainebaseline-HeadacheAge",
				"historymigrainebaseline-HeadacheDays",
				"historymigrainebaseline-LongHeadache",
				"historymigrainebaseline-ShortHeadache",
				"hit6-hit6tot",
				"medicalhistory-AgeStartedPeriods",
				"pedsqlchildteen-pedstot",
				"pedsqlparent-pedstot",
				"physicalexam_cont-Diastolic",
				"physicalexam_cont-Height",
				"physicalexam_cont-Pulse",
				"physicalexam_cont-Systolic",
				"physicalexam_cont-Weight",
				]




def load_data(data_type):
	data_dir='datasets/CHAMP/CHAMP Study Shared Data Sets and Data Dictionary/CHAMP Study Shared Data Sets'
	prefix_data_dir="../"
	if os.path.exists(os.path.join("/scratch/network/slala")):
		prefix_data_dir=os.path.join("/scratch/network/slala")
	elif os.path.exists(os.path.join("/scratch/gpfs/slala")):
		prefix_data_dir=os.path.join("/scratch/gpfs/slala")

	return pd.read_sas(os.path.join(prefix_data_dir,data_dir,data_type+".sas7bdat"))


def get_dataset_dir():
	path_prefix="../datasets"
	if os.path.exists(path_prefix):
		pass
	elif os.path.exists('/scratch/gpfs/slala/datasets'):
		path_prefix='/scratch/gpfs/slala/datasets'
	elif os.path.exists('/scratch/network/slala/datasets'):
		path_prefix='/scratch/network/slala/datasets'

	return os.path.join(path_prefix,"CHAMP/CHAMP Study Shared Data Sets and Data Dictionary/CHAMP Study Shared Data Sets")

def get_cache_dir():
	path_prefix="../datasets"
	if os.path.exists(path_prefix):
		pass
	elif os.path.exists('/scratch/gpfs/slala/datasets'):
		path_prefix='/scratch/gpfs/slala/datasets'
	elif os.path.exists('/scratch/network/slala/datasets'):
		path_prefix='/scratch/network/slala/datasets'
	return os.path.join(path_prefix,"CHAMP/CHAMP Study Shared Data Sets and Data Dictionary/CHAMP Study Shared Data Sets")


def is_invalid(col):
	col=col.lower()

	if col in ['subjectid']: 
		return True

	if 'visit' in col:
		return True

	if 'dtdays' in col and col not in ["birthdtdays","enddtdays"]:
		return True

	return False


def get_feature_names():
	table_to_features=OrderedDict()

	table_to_features['subject_status']=['age_12','age_13','headache_14','headache_15']

	for table in os.listdir(os.path.join(get_dataset_dir())):

		table=os.path.splitext(table)[0]
		if table in ['subject_status', 'sae_summary', 'ae_summary', 
					'drugacct', 'neuroexam_postrand', 'physicalexam_postrand', 'pedmidas']:
			continue

		table_cols=load_data(table)
		table_to_features[table]=[col for col in table_cols if not is_invalid(col)]

	return table_to_features

def get_valid_sids(treatment_df,outcome_df):
	keep_sids=[]	
	for sid in treatment_df.SubjectID:
		try:
			sid_df=outcome_df[(outcome_df.SubjectID==sid)].sort_values(by="VisitDTDays")
			assert len(sid_df)>=2 and 0 in sid_df.VisitDTDays.tolist() and sid_df.VisitDTDays.tolist()[-1]>=160
			keep_sids.append(sid)
		except:
			continue
	return np.array(keep_sids)

def has_dtdays_col(columns):
	for col in columns:
		col=col.lower()
		if "dtdays" in col and col not in ["birthdtdays", "enddtdays"]:
			return True
	return False

"""
Assumes only 1 date column present
"""
def get_dtdays_col(columns):
	for col in columns:
		col_orig=col[:]
		col=col.lower()
		if "dtdays" in col and col not in ["birthdtdays", "enddtdays"]:
			return col_orig

def get_sid_table_data(sid, tn, features, table):

	cols=table.columns.tolist()
	try:
		if "Visit" in cols:
			if tn == "physicalexam_cont": # multiple recordings
				sid_data=table[(table.SubjectID==sid) & (table.Visit==2)].loc[:,features].iloc[0].tolist()
			else: # other tables only have 1
				sid_data=table[(table.SubjectID==sid) & ((table.Visit==1) | (table.Visit==2))].loc[:,features].iloc[0].tolist()
		elif has_dtdays_col(cols):
			dtdays_col=get_dtdays_col(cols)
			# print(dtdays_col)
			# if subject has multiple eligible dates/recording, choose the latest baseline
			sid_data=table[(table.SubjectID==sid) & (table.loc[:,dtdays_col]<=0)].sort_values(by=dtdays_col).loc[:,features].iloc[-1].tolist()
		else:
			sid_data=table[table.SubjectID==sid].loc[:,features].iloc[0].tolist()
	except Exception as e:
		# print(e)
		sid_data=[np.NaN]*len(features) # subject data not present in table
	return sid_data

def get_sid_data(sid, treatment_df, outcome_df, table_dfs, table_to_features):
	sid_data=[]

	sid_data.append(treatment_df[treatment_df.SubjectID==sid].trt.iloc[0] == 2)

	sid_outcome_df=outcome_df[(outcome_df.SubjectID==sid)].sort_values(by="VisitDTDays")
	assert len(sid_outcome_df)>=2 and 0 in sid_outcome_df.VisitDTDays.tolist() and sid_outcome_df.VisitDTDays.tolist()[-1]>=160
	sid_data.extend([sid_outcome_df.pedmidas_score.iloc[0], sid_outcome_df.pedmidas_score.iloc[-1]])

	for tn,tnfs in table_to_features.items():
		sid_data.extend(get_sid_table_data(sid, tn, tnfs, table_dfs[tn]))

	return sid_data


def convert_med_cat(e):
	e=e.decode('utf-8')
	med_names=[b'Acetaminophen (Tylenol)', b'Aspirin', b'Dihydroergotamine (DHE)',
					   b'Eletriptan (Relpax)', b'Excedrin', b'Ibuprofen (Advil/Motrin)',
					   b'Ketorolac (Toradol)', b'Metoclopramide (Reglan)',
					   b'Naproxen/Naprosyn (Aleve)', b'Other',
					   b'Prochlorperazine (Compazine)', b'Rizatriptan (Maxalt)',
					   b'Sumatriptan (Imitrex)', b'Valproate Sodium (Depacon)',
					   b'Zolmitriptan (Zomig)',b'Almotriptan (Axert)',b'Frovatriptan (Frova)',b'Naratriptan (Amerge)',b'Treximet',
					   b'Chlorpromazine (Thorazine)',b'Dexamethasone (Decadron)', b'Methylprednisone (Solu-Medrol)']
	med_names=np.array([mn.decode('utf-8') for mn in med_names])

	if e in med_names:
		return np.where(med_names==e)[0][0]

	return np.NaN

def get_complete_dataset():
	outcome_df=load_data('pedmidas')
	treatment_df=load_data('subject_status')
	treatment_df=treatment_df[( (treatment_df.trt==1) | (treatment_df.trt==2)) \
								& (treatment_df.final_analysis_pop==1) \
								& (treatment_df.completion_status==1)]
	
	keep_sids=get_valid_sids(treatment_df, outcome_df)

	table_to_features=get_feature_names()
	table_dfs={tn:load_data(tn) for tn in list(table_to_features.keys())}
	
	table_dfs['acuteheadacheconmed'].MedName=table_dfs['acuteheadacheconmed'].MedName.apply(lambda e: convert_med_cat(e))

	columns=['trt','pedmidas-pre','pedmidas-post']
	columns.extend("%s-%s"%(tn,tnf) for tn,tnfs in table_to_features.items() for tnf in tnfs)

	data_by_pid={}
	for sid in keep_sids:
		data_by_pid[sid]=get_sid_data(sid, treatment_df, outcome_df, table_dfs, table_to_features)

	return pd.DataFrame.from_dict(data_by_pid,'index',columns=columns)

def get_nonmissing_df(complete_df, missing_method="drop", missing_threshold=None):
	mrs=[]
	for ci,c in enumerate(complete_df.columns):
		mrs.append(complete_df.iloc[:,ci].isna().sum()/len(complete_df))
	mrs=np.array(mrs)

	if missing_method == "drop":
		return complete_df.iloc[:,np.where(mrs==0)[0]]

	# impute


def get_dataset_array(complete_df, outcome_metric):
	ignore_cols=["trt", "%s-pre"%outcome_metric, "%s-post"%outcome_metric]

	features=[outcome_metric]
	for c in complete_df.columns:
		if c in ignore_cols:
			continue
		features.append(c)
	features=np.array(features)
	ind_outcome_ft=0

	full_data,full_labels=[],[]         
	full_sids=np.arange(len(complete_df))
	for pid in full_sids:
		data=[]
		pid_df=complete_df.iloc[pid]
		
		data.append([pid_df.loc["%s-pre"%outcome_metric],pid_df.loc["%s-post"%outcome_metric]])
		
		for c in complete_df.columns:
			if c in ignore_cols:
				continue
			data.append([pid_df.loc[c],0])
		
		full_data.append(data)
		full_labels.append(int(pid_df.loc["trt"]))
		
	full_data,full_labels=np.array(full_data).transpose(0,2,1),np.array(full_labels)

	return [full_data,full_labels,full_sids], features, ind_outcome_ft


def is_categ(c, s):
	return c not in VARS_NUMERICAL



"""

incl:
categorical vars
any info. collected post baseline visit
"""
def get_indices_to_ignore(features):
	indices=[]
	for fi,f in enumerate(features):
		if f in VARS_IGNORE:
			indices.append(fi)

	return np.array(indices)

def handle_categorical(complete_df, drop_categ=True):
	categ_cols=[c for c in complete_df.columns if (is_categ(c, complete_df.loc[:,c]) and c not in ['trt', 'pedmidas-pre','pedmidas-post'])]
	return complete_df.drop(columns=categ_cols)

def get_dataset(missing_method="drop",missing_threshold=None,
				include_extra=0,feature_indices=None, drop_categ=True):

	cache_dir=get_cache_dir()
	if os.path.exists(os.path.join(cache_dir,'complete_df.pickle')):
		with open(os.path.join(cache_dir,'complete_df.pickle'),'rb') as f:
			complete_df=pickle.load(f)
	else:
		complete_df=get_complete_dataset()
		with open(os.path.join(cache_dir,'complete_df.pickle'),'wb') as f:
			pickle.dump(complete_df,f)

	complete_df=get_nonmissing_df(complete_df, missing_method, missing_threshold)
	complete_df=complete_df.astype(float)
	assert not complete_df.isna().any().any()

	if drop_categ:
		complete_df=handle_categorical(complete_df, drop_categ=drop_categ)

	# complete_df=convert_float(complete_df)

	[full_data,full_labels,full_sids],features,iof=get_dataset_array(complete_df, "pedmidas")

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
	parser.add_argument('--indices_to_ignore',type=parse_str_helper,default=None)
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
	
	full_ds,features,ind_outcome_ft=get_dataset(include_extra=args.include_extra,feature_indices=feature_indices,drop_categ=args.drop_categ)


	if args.tune_cv_per_data_seed:
		syn_ctrl.main_with_data_seeds(args, full_ds, features, ind_outcome_ft)
	else:
		syn_ctrl.main(args, full_ds, features, ind_outcome_ft)


		






