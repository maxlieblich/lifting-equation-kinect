### NIOSH Lifting Equation using Kinect and error-correction models
This repository contains code for using data from the Kinect skeleton model to estimate parameters for the 
NIOSH Lifting Equation, including parameter estimate error correction using machine learning methods. This code
(and other code not yet included) was used [here](http://www.ncbi.nlm.nih.gov/pubmed/24987523) to produce 
error-correction models for the parameters of the NIOSH Lifting Equation.

#### The idea
The Kinect generates noisy skeleton data, especially for non-gaming positions. We simultaneously collected
data using the Kinect and a gold-standard marker-based system and used them to build regression models for the
Kinect-vs-true error values of individual parameters, using the Kinect skeleton (suitably normalized) as an input. 
The best performance came from gradient boosted regression tree models (with a similar performance to random forest 
regression, at a lower computational cost).

#### Current state of the repository
Not all code written for the project is included, only the parts that demonstrate how we calculated all of 
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

#### Not presently included
We do not currently include 

- the `C#` code for ingesting Kinect data,
- the code for the WebGL-based web application used to process and clean the data, or
- the resulting master data files

in this repository. We may do so at a later date.
