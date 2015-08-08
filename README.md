### NIOSH Lifting Equation using Kinect and error-correction models
This repository contains code for using data from the Kinect skeleton model to estimate parameters for the 
NIOSH Lifting Equation, including parameter estimate error correction using machine learning methods. This code
(and other code not yet included) was used [here](http://www.ncbi.nlm.nih.gov/pubmed/24987523) to produce 
error-correction models for the parameters of the NIOSH Lifting Equation.

#### The idea
The Kinect generates noisy skeleton data, especially for non-gaming positions. We simultaneously collected
data using the Kinect and a gold-standard marker-based system (Qualysis) and used them to build regression models for the
Kinect-vs-Qualysis error values of individual parameters, using the Kinect skeleton (suitably normalized) as an input. 
The best performance came from gradient boosted regression tree models (with a similar performance to random forest 
regression, at a lower computational cost).

#### Current state of the repository

**Code**: Not all code written for the project is included, only the parts that demonstrate how we calculated all of 
the individual parameters and how we built error-correction models for those parameters using our master database.
The code is not meant to be used by anyone without careful reading. With the exception of removing comments,
it has not been modified since the experiments described above were done.

This is not engineering-quality code. It is not production-ready code. It is very poorly-written code that got the 
job done so we could test the idea. Documentation is sparse and mostly non-existent. 
No guarantees are made about the future usefulness of this code. Use it at your own risk.

The main reason we have put it here is so that anyone can see exactly how we calculated the various 
quantities involved, especially quantities like the 
[Asymmetry Angle](http://wonder.cdc.gov/wonder/prevguid/p0000427/p0000427.asp#head005003004000000), 
which is difficult to derive from the Kinect skeleton model due to the subtlety of robustly describing the 
mid-saggital plane from Kinect joint positions.

**Data**: We have also included 

- the raw data files (processed into csv files) from both the Qualysis and Kinect systems (in `Python/DataFiles/qual` and `Python/DataFiles/kin`, respectively)
- the master data file (in `Python/DataFiles/master`), which is a `json` file containing an object with two fields: `qual` and `kin`
- two further "master" files: one file that includes data derived (as described below) from all but subject number 3, and the other containing only data from subject 3.

The master files contain the result of filtering, smoothing, aligning, and resampling each per-subject file and concatenating all (or some, in the case of `allbut3.json`) of the resulting data into a list of timestamped poses (aligned so that the rows in the Qualysis and Kinect data correspond to appropriately resampled simultaneous measurements). All master files are rather large (in small data terms); running the scripts that symmetrize them result in even larger files, but helped with model-building.

All identifying information has been removed with the exception of collection date-times. In building our models, attempted to avoid overfitting due to autocorrelation by separating training and testing sets according to subject number and not random choices from the concatenated timeseries. The master data file provided here does not have subject numbers built in; such "all-but-one" files were created separately from the raw data and can be done using the scripts provided here. We have included `allbut3.json` and `only3.json` for anyone who wants one such split for testing ideas.

#### Not presently included
We do not currently include the `C#` code for ingesting raw Kinect data or the code for the WebGL-based web application used to process and clean the data in this repository. 
