# "The aging human brain: A causal analysis of the effect of sex and age on brain volume"
# Gomez-Ramirez, J et al.
# Jan/24/2021

import os, sys
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)
#%config Inline.figure_format = 'retina'
az.style.use('arviz-darkgrid')

np.random.seed(0)
# figures dir
if sys.platform == 'linux':
	figures_dir = ""
	figures_dir = ""
else:
	figures_dir = ""
#  


def plot_bars_with_erros():
	"""
	"""
	xts = ["Female", "Male", "Age"]
	m_mean = -0.266; m_std = 0.052; f_mean = 0.138 ; f_std= 0.037; a_mean = -0.336; a_std = 0.031;
	x_pos = np.arange(len(xts))
	means = [f_mean, m_mean,a_mean]
	errors = [f_std,m_std,a_std]
	fig, ax = plt.subplots()
	bars=ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
	bars[-1].set_color('r');bars[-2].set_color('r');
	ax.set_ylabel("Brain preservation (Brain2ICV)")
	ax.set_xticks(x_pos)
	ax.set_xticklabels(xts)
	ax.set_title("Coefficients of Sex + Age -> Brain2ICV")
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, 'bars_ageandsex_brain.png'))

def standardize(series):
	"""Standardize a pandas series"""
	std_series = (series - series.mean()) / series.std()
	return std_series


def translateENG(listinesp):
	""" tanslate list features from esp to eng 
	""" 
	listeng = list(listinesp)
	for n, el in enumerate(listinesp):
		if 'bus_' in el or 'fcs' in el or 'scd_' in el:
			# change vista por vist
			el2include= 'cog test'
			listeng[n]= el2include #.append(el2include)
		elif 'sexo' in el:
			el2include=  el[:-1]
			listeng[n]=el2include #.append(el2include)
		elif 'dx_corto' in el:
			el2include=  'dx_y' + el[-1]
			listeng[n]=el2include #.append(el2include)
		elif 'anos_' in el:
			el2include=  ' years_school'
			listeng[n]=el2include #.append(el2include)
		elif 'edad' in el:
			el2include=  'age'
			listeng[n]=el2include #.append(el2include)
		elif 'depre' in el:
			listeng[n]=el #.append(el)
		elif 'nivel' in el:
			listeng[n]='school_level' #.append(el)
		elif 'BrainSeg' in el:
			listeng[n]='BrainSeg2eITV' #.append(el)
	return  listeng

def plots_and_stuff(df):
	"""
	"""
	fig = plt.figure()
	ax = sns.violinplot(x="sexo", y="fr_BrainSegVol_to_eTIV_y1", data=df)
	ax.set(xlabel='Gender', ylabel='Brain2ICV')
	ax.set_xticklabels(['M','F'])
	fig_file = os.path.join(figures_dir, 'violin_sex_b2icv.png')
	plt.savefig(fig_file)
	fig = plt.figure()
	ax = sns.violinplot(x="sexo", y="edad_visita1", data=df)
	ax.set(xlabel='Gender', ylabel='Age')
	ax.set_xticklabels(['M','F'])
	fig_file = os.path.join(figures_dir, 'violin_sex_age.png')
	plt.savefig(fig_file)
	# ttest based on gender groupby
	b0 = df['fr_BrainSegVol_to_eTIV_y1'].loc[df['sexo']==0]
	b1 = df['fr_BrainSegVol_to_eTIV_y1'].loc[df['sexo']==1]
	stats.ttest_ind(b0,b1)
	e0 = df['edad_visita1'].loc[df['sexo']==0]
	e1 = df['edad_visita1'].loc[df['sexo']==1]
	stats.ttest_ind(e0,e1)


def causality_test():
	""" Load csv file to build EDA plots and PyMC models
	"""
	#Load Data https://github.com/grjd/causalityagingbrain/blob/main/dataset_gh.csv
	csv_path = ""
	dataframe = pd.read_csv(csv_path, sep=';')
	dataframe_orig = dataframe.copy()
	plots_and_stuff(df)

	corrmatrix =  df.corr(method='pearson')
	mask = np.zeros_like(corrmatrix)
	mask[np.triu_indices_from(mask)] = True
	plt.figure(figsize=(7,7))
	heatmap = sns.heatmap(corrmatrix,mask=mask,annot=True, center=0,square=True, linewidths=.5)
	#heatmap = sns.heatmap(atrophy_corr,annot=True, center=0,square=True, linewidths=.5)
	heatmap.set_xticklabels(colsofinterest_Eng, rotation=45, fontsize='small', horizontalalignment='right')
	heatmap.set_yticklabels(colsofinterest_Eng, rotation=0, fontsize='small', horizontalalignment='right')
	fig_file = os.path.join(figures_dir, 'heat_CorrChapter.png')
	plt.savefig(fig_file)
	
	# Standardize regressors and target
	df["brain_std"] = standardize(df["fr_BrainSegVol_to_eTIV_y1"])
	df["age_std"] = standardize(df["edad_visita1"])
	df["cog_std"] = standardize(df["fcsrtlibdem_visita1"])
	# Encode Categorical Variables
	df["school_id"] = pd.Categorical(df["nivel_educativo"]).codes
	df["sex_id"] = pd.Categorical(df["sexo"]).codes
	
	################################################################
	################## SEX (0M, 1F) -> BRAIN #######################
	#################################################################
	with pm.Model() as mXB:
		#sigma = pm.Uniform("sigma", 0, 1)
		sigma = pm.HalfNormal("sigma", sd=1)
		#mu_x = pm.Normal("mu_x", 0.7, 0.3, shape=2)
		mu_x = pm.Normal("mu_x", 0.0, 1.0, shape=2)
		#brain_remained = pm.Normal("brain_remained", mu_x[df["sex_id"]], sigma, observed=df["fr_BrainSegVol_to_eTIV_y1"])
		brain_remained = pm.Normal("brain_remained", mu_x[df["sex_id"]], sigma, observed=df["brain_std"])
	    # men - women
	    # mu[0]  0.695,  mu[1]  0.709 Women came at late age with less atrophy, bigger brains
		diff_fm = pm.Deterministic("diff_fm", mu_x[0] - mu_x[1])
		mXB_trace = pm.sample(1000)
	print(az.summary(mXB_trace))
	az.plot_trace(mXB_trace, var_names=["mu_x", "sigma"])
	plt.savefig(os.path.join(figures_dir, 'pm_trace_sex_brain-hn.png'))
	az.plot_forest(mXB_trace, combined=True, model_names=["X~B"],var_names=["mu_x"], hdi_prob=0.95)
	plt.savefig(os.path.join(figures_dir, 'pm_forest_sex_brain-hn.png'))
	# Posterior Predictive checks
	y_pred_g = pm.sample_posterior_predictive(mXB_trace, 100, mXB)
	data_ppc = az.from_pymc3(trace=mXB_trace, posterior_predictive=y_pred_g)
	ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
	ax[0].legend(fontsize=15)
	plt.savefig(os.path.join(figures_dir, 'ppc_xXB-hn.png'))
	
	################################################################
	################## AGE -> BRAIN ################################
	#################################################################
	print('Calling to PyMC3 Model Age - > Brain...\n')
	with pm.Model() as m_AB:
		alpha = pm.Normal("alpha", 0, 1) #0.2
		betaA = pm.Normal("betaA", 0, 1) #0.5
		#sigma = pm.Exponential("sigma", 1)
		sigma = pm.HalfNormal("sigma", sd=1)
		mu = pm.Deterministic("mu", alpha + betaA * df["age_std"])
		brain_std = pm.Normal("brain_std", mu=mu, sigma=sigma, observed=df["brain_std"].values)
		prior_samples = pm.sample_prior_predictive()
		m_AB_trace = pm.sample(1000)
	print(az.summary(m_AB_trace, var_names=["alpha", "betaA", "sigma"]))
	az.plot_trace(m_AB_trace, var_names=["alpha", "betaA","sigma"])
	plt.savefig(os.path.join(figures_dir, 'pm_trace_age_brain.png'))
	az.plot_forest([m_AB_trace,],model_names=["A~B"],var_names=["betaA"],combined=True,hdi_prob=0.95);
	plt.savefig(os.path.join(figures_dir, 'pm_forest_AtoB.png'))
	# Posterior Predictive checks
	y_pred_g = pm.sample_posterior_predictive(m_AB_trace, 100, m_AB)
	data_ppc = az.from_pymc3(trace=m_AB_trace, posterior_predictive=y_pred_g)
	ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
	ax[0].legend(fontsize=15)
	plt.savefig(os.path.join(figures_dir, 'ppc_AB-hn.png'))
	################################################################
	################## SEX+AGE -> BRAIN #######################
	#################################################################
	print('Calling to PyMC3 Model Age + Sex - > Brain...\n')
	sexco = pd.Categorical(df.loc[:, "sexo"].astype(int))
	with pm.Model() as m_XAB:
		alphax = pm.Normal("alphax", 0, 1, shape=2)
		betaA = pm.Normal("betaA", 0, 1)
		mu = alphax[sexco] + betaA*df["age_std"]
		sigma = pm.Exponential("sigma", 1)
		#mu = pm.Deterministic("mu", alpha + betaA * df["age_std"] + betaB * df["brain_std"])
		brain_std = pm.Normal("brain_std", mu=mu, sigma=sigma, observed=df["brain_std"].values)
		prior_samples = pm.sample_prior_predictive()
		m_XAB_trace = pm.sample()
	print(az.summary(m_XAB_trace, var_names=["alphax", "betaA", "sigma"]))
	az.plot_trace(m_XAB_trace, var_names=["alphax", "betaA"])
	plt.savefig(os.path.join(figures_dir, 'pm_trace_ageandsex_brain.png'))
	az.plot_forest([m_XAB_trace, mXB_trace, m_AB_trace,],model_names=["XA~B", "X~B", "A~B"], var_names=["alphax","mu_x","betaA"], combined=True,hdi_prob=0.95);
	plt.savefig(os.path.join(figures_dir, 'pm_forest_mXAtoB.png'))
	# Posterior Predictive checks
	y_pred_g = pm.sample_posterior_predictive(m_XAB_trace, 100, m_XAB)
	data_ppc = az.from_pymc3(trace=m_XAB_trace, posterior_predictive=y_pred_g)
	ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
	ax[0].legend(fontsize=15)
	plt.savefig(os.path.join(figures_dir, 'ppc_XAB-hn.png'))

	print('Calling to PyMC3 Model Brain - > Memory...\n')
	with pm.Model() as m_BC:
		alpha = pm.Normal("alpha", 0, 1) #0.2
		betaB = pm.Normal("betaB", 0, 1) #0.5
		sigma = pm.Exponential("sigma", 1)
		mu = pm.Deterministic("mu", alpha + betaB * df["brain_std"])
		cognition_std = pm.Normal("cognition_std", mu=mu, sigma=sigma, observed=df["cog_std"].values)
		prior_samples = pm.sample_prior_predictive()
		m_BC_trace = pm.sample()
	az.plot_trace(m_BC_trace, var_names=["alpha", "betaB"])
	plt.savefig(os.path.join(figures_dir, 'pm_trace_brain_cog.png'))
	print(az.summary(m_BC_trace, var_names=["alpha", "betaB", "sigma"]))
	# Scatter plot x = Brain atrophy Y= Memory test
	mu_mean = m_BC_trace['mu']
	mu_hpd = pm.hpd(mu_mean)
	plt.figure(figsize=(9, 9))
	df.plot('brain_std', 'cog_std', kind='scatter') #, xlim = (-2, 2)
	plt.plot(df.brain_std, mu_mean.mean(0), 'C2')
	plt.savefig(os.path.join(figures_dir, 'scatter_hpd_B2M.png'))
	print('Saved Figure scatter_hpd_B2M.png \n')

	print('Calling to PyMC3 Model School - > Memory...\n')
	# School -> Memory method2  m5_9
	with pm.Model() as mSM2:
		#sigma = pm.Uniform("sigma", 0, 1)
		sigma = pm.Exponential("sigma", 1)
		mu = pm.Normal("mu", 0.0, 0.5, shape=df["school_id"].max() + 1)
		memory = pm.Normal("memory", mu[df["school_id"]], sigma, observed=df["cog_std"])
		mSM2_trace = pm.sample()
	print(az.summary(mSM2_trace))
	az.plot_trace(mSM2_trace, var_names=["mu", "sigma"])
	plt.savefig(os.path.join(figures_dir, 'pm_trace2_school_memory.png'))
	az.plot_forest(mSM2_trace, combined=True, var_names=["mu"], hdi_prob=0.95)
	plt.savefig(os.path.join(figures_dir, 'pm_forest2_school_memory.png'))
	pdb.set_trace()

	print('Calling to PyMC3 Model Age - > Memory...\n')
	with pm.Model() as m_AC:
		alpha = pm.Normal("alpha", 0, 1)
		betaA = pm.Normal("betaA", 0, 1)
		sigma = pm.Exponential("sigma", 1)
		mu = pm.Deterministic("mu", alpha + betaA * df["age_std"])
		cognition_std = pm.Normal("cognition_std", mu=mu, sigma=sigma, observed=df["cog_std"].values)
		prior_samples = pm.sample_prior_predictive()
		m_AC_trace = pm.sample()
	az.plot_trace(m_AC_trace, var_names=["alpha", "betaA"])
	plt.savefig(os.path.join(figures_dir, 'pm_trace_age_cog.png'))
	print(az.summary(m_AC_trace, var_names=["alpha", "betaA", "sigma"]))
	# Scatter A2M
	mu_mean = m_AC_trace['mu']
	mu_hpd = pm.hpd(mu_mean)
	plt.figure(figsize=(9, 9))
	df.plot('age_std', 'cog_std', kind='scatter') #, xlim = (-2, 2)
	plt.plot(df.age_std, mu_mean.mean(0), 'C2')
	plt.savefig(os.path.join(figures_dir, 'scatter_hpd_A2M.png'))
	print('Saved Figure scatter_hpd_A2M.png \n')
	 
	print('Calling to PyMC3 Model Age + Brain - > Memory...\n')
	with pm.Model() as m_BAC:
		alpha = pm.Normal("alpha", 0, 1)
		betaA = pm.Normal("betaA", 0, 1)
		betaB = pm.Normal("betaB", 0, 1)
		sigma = pm.Exponential("sigma", 1)
		mu = pm.Deterministic("mu", alpha + betaA * df["age_std"] + betaB * df["brain_std"])
		cognition_std = pm.Normal("cognition_std", mu=mu, sigma=sigma, observed=df["cog_std"].values)
		prior_samples = pm.sample_prior_predictive()
		m_BAC_trace = pm.sample()
	print(az.summary(m_BAC_trace, var_names=["alpha", "betaB", "betaA", "sigma"]))
	az.plot_forest([m_BAC_trace, m_AC_trace, m_BC_trace,],model_names=["BA~C", "A~C", "B~C"],var_names=["betaA", "betaB"],combined=True,hdi_prob=0.95);
	plt.savefig(os.path.join(figures_dir, 'pm_forest_mBAC_AB2M.png'))