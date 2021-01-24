# Code and Dataset supplement "The aging human brain: A causal analysis of the effect of sex and age on brain volume"

This repository contains code and data used in the following publication:

**Citation**

Jaime Gómez-Ramírez, Miguel A. Fernández-Blázquez, Javier González-Rosa, "The aging human brain: A causal analysis of the effect of sex and age on brain volume" (pre-print on BioRxiv: https://doi.org/10.1101/2020.11.20.391623)

**Abstract**

The goal of this work is to study how brain volume loss at old age is affected by factors such as age, APOE gene, sex, and school level. The study of brain volume loss at old age relative to young age requires at least in principle two MRI scans performed at both young and old age. There is, however, a way to address the problem by having only one MRI scan at old age. We compute the total brain loss of elderly subjects as the ratio between the estimated brain volume and the estimated total intracranial volume. Magnetic resonance imaging (MRI) scans of 890 healthy subjects aged 70 to 85 were assessed. The causal analysis of factors affecting brain atrophy was performed using Probabilistic Bayesian Modeling and the Mathematics of Causal Inference.
We find that healthy subjects get into their seventies with an average brain volume loss of $30\%$ from their maximum brain volume at a young age. Both age and sex are causally related to brain atrophy, with women getting to elderly age with $1\% $ larger brain volume relative to intracranial volume than men. 
How the brain ages and what are the reasons for sex differences in adult lifespan are causal questions that need to be addressed with causal inference and empirical data. The graphical causal modeling presented here can be instrumental in understanding a puzzling scientific inquiry -the biological age of the brain.

**Dataset description**

The dataset contains a csv file, it can be opened as a Pandas dataframe containing the results of the automated segmentation performed with FSL. The dataset includes the columns:

- _edad_visita1_: Age of the participant in the moment of performing the MRI scan 
- _sexo_: Sex of the participant encoded as 0 Male and 1 Female
- _nivel_educativo_: Schooling level encoded as 0 _no formal education_, 1 _primary education_, 2 _middle or high school degree_ and 3 _university degree_. 
- _apoe_: APOE genotype was studied with total DNA isolated from peripheral blood following standard procedures. The APOE variable was coded 1 for the e4-carriers, and 0 for non-carriers. 
- _familial_ad_: Family history of AD was coded as 0 for subjects with no parents or siblings diagnosed with dementia and 1 for those with at least one parent or sibling diagnosed with dementia.
- _fcsrtlibdem_visita1_: Cognitive status was determined with the Mini-Mental Status Examination (MMSE), Free and Cued Selective Reminding Test (FCSRT), Semantic fluency, Digit-Symbol Test and Functional Activities Questionnaire (FAQ). 
- _fr_BrainSegVol_to_eTIV_y1_: brain2icv or brain volume estimate to intracraneal volume estimate ratio. 

The dimensionality of the dataset is 890x7 (subjects x variables)
```
df.shape
(890, 7)
df.columns
Index(['edad_visita1', 'sexo', 'apoe', 'nivel_educativo', 'familial_ad',
       'fcsrtlibdem_visita1', 'fr_BrainSegVol_to_eTIV_y1'],
      dtype='object')
```   
**MRI Data collection**

The imaging data were acquired in the sagittal plane on a 3T General Electric scanner (GE Milwaukee, WI) utilizing T1-weighted inversion recovery, supine position, flip angle $12\circ$, 3-D pulse sequence: echo time \textit{Min. full}, time inversion 600 ms., Receiver Bandwidth $19.23$ kHz, field of view $= 24.0$ cm, slice thickness $1$ mm and Freq $\times$ Phase $288 \times 288$. The brain volume loss at the moment of having the MRI compared to the maximum brain volume is computed as the Brain Segmentation Volume to estimated Total Intracranial Volume (eTIV) \cite{eTIV} ratio (ICV and eTIV the FreeSurfer term for intracranial volume are used equivalently). The postprocessing was performed with FreeSurfer \cite{fischl2012freesurfer}, version freesurfer-darwin-OSX-ElCapitan-dev-20190328-6241d26 running under a Mac OS X, product version 10.14.5. For the sake of illustration, Figure \ref{fig:brains} shows the result produced of the intracranial volume segmentation for two subjects in the study. 

_[FreeSurfer, 2017] FreeSurfer cortical reconstruction and parcellation process. (2017).Anatomical processing script:recon-all.
https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all._

