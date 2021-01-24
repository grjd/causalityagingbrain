# Code and Dataset supplement "The aging human brain: A causal analysis of the effect of sex and age on brain volume"

This repository contains code and data used in the following publication:

J. Gomez-Ramirez et al, "The aging human brain: A causal analysis of the effect of sex and age on brain volume" (pre-print on BioRxiv: https://doi.org/10.1101/2020.11.20.391623)

**Abstract**
The goal of this work is to study how brain volume loss at old age is affected by factors such as age, APOE gene, sex, and school level. The study of brain volume loss at old age relative to young age requires at least in principle two MRI scans performed at both young and old age. There is, however, a way to address the problem by having only one MRI scan at old age. We compute the total brain loss of elderly subjects as the ratio between the estimated brain volume and the estimated total intracranial volume. Magnetic resonance imaging (MRI) scans of 890 healthy subjects aged 70 to 85 were assessed. The causal analysis of factors affecting brain atrophy was performed using Probabilistic Bayesian Modeling and the Mathematics of Causal Inference.
We find that healthy subjects get into their seventies with an average brain volume loss of $30\%$ from their maximum brain volume at a young age. Both age and sex are causally related to brain atrophy, with women getting to elderly age with $1\% $ larger brain volume relative to intracranial volume than men. 
How the brain ages and what are the reasons for sex differences in adult lifespan are causal questions that need to be addressed with causal inference and empirical data. The graphical causal modeling presented here can be instrumental in understanding a puzzling scientific inquiry -the biological age of the brain.

**Dataset description**

The dataset contains two csv files: 
- *df_fsl_lon.csv* is the Pandas dataframe containing the results of the automated segmentation performed with FSL 
- *df_free_lon.csv* contains the  Pandas dataframe containing the results of the automated segmentation performed with FreeSurfer. 

The fields include in the dataset are as follows:
- _Age_ the age of the participant in the moemnt of performing the MRI scan (%.2f)
- _Sex_ encoded as 0 Male and 1 Female
- Subcortical Volume estimates use the nomenclature: [fsl|free] [R|L] [structure] where structure can be Thalamus, Accumbens, Pallidum, Hippocampus, Amygdala, Caudate and Putamen. The volume is expressed in mm^3.

```

```   
**MRI Data collection**
A total of 4028 MRIs were collected in 5 years, 990 in the first visit, 768 in the second, 723 in the third, 634 in the fourth, 542 in the fifth, and 371 in the sixth year. The imaging data were acquired on a 3T General Electric scanner (GE Milwaukee) utilizing the following T1-weighted inversion recovery, flip angle 12°, 3-D pulse sequence: echo time _Min. full_, time inversion 600 ms, Receiver Bandwidth = 19.23 kHz, field of view = 24.0 cm, slice thickness = 1 mm, _Freq. x Phase = 288 x 288_.
The preprocessing of MRI 3 Tesla images in this study consisted of generating an isotropic brain image with non-brain tissue removed. We used the initial, preprocessing step in the two computational segmentation tool used in this study: FSL pipeline _(fsl-anat)_ and the FreeSurfer pipeline _(recon-all)_. 
We run both pipelines in an identical computational setting: Operating System Mac OS X, product version 10.14.5 and build version 18F132. The version of FreeSurfer is FreeSurfer-darwin-OSX-ElCapitan-dev-20190328-6241d26. The version of the BET tool for FSL is v2.1 - FMRIB Analysis Group, Oxford and the FIRST tool version is 6.0.

_[FreeSurfer, 2017] FreeSurfer cortical reconstruction and parcellation process. (2017).Anatomical processing script:recon-all.
https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all._

_[FSL, 2017] FSL (2017). Anatomical processing script: fsl_anat. https://fsl.fmrib.ox.ac.uk/
fsl/fslwiki/fsl_anat._ 
