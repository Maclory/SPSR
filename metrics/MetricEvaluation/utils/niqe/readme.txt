NIQE Software release.

=======================================================================
-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2011 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
and Center for Perceptual Systems (CPS, http://www.cps.utexas.edu) at the University of Texas at Austin (UT Austin, 
http://www.utexas.edu), is acknowledged in any publication that reports research using this code. The research
is to be cited in the bibliography as:

1) A. Mittal, R. Soundararajan and A. C. Bovik, "NIQE Software Release", 
URL: http://live.ece.utexas.edu/research/quality/niqe.zip, 2012.

2) A. Mittal, R. Soundararajan and A. C. Bovik, "Making a Completely Blind Image Quality Analyzer", submitted to IEEE Signal Processing Letters, 2012.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

Author  : Anish Mittal 
Version : 1.0

The authors are with the Laboratory for Image and Video Engineering
(LIVE), Department of Electrical and Computer Engineering, The
University of Texas at Austin, Austin, TX.

Kindly report any suggestions or corrections to mittal.anish@gmail.com

=======================================================================

This is a demonstration of the Naturalness Image Quality Evaluator(NIQE) index. The algorithm is described in:

A. Mittal, R. Soundararajan and A. C. Bovik, "Making a Completely Blind Image Quality Analyzer", submitted to IEEE Signal Processing Letters, 2012.

You can change this program as you like and use it anywhere, but please
refer to its original source (cite our paper and our web page at
http://live.ece.utexas.edu/research/quality/niqe_release.zip).

=======================================================================
Running on Matlab 

Input : A test image loaded in an array

Output: A quality score of the image. Higher value represents a lower quality.
  
Usage:

1. Load the image, for example

   image     = imread('testimage1.bmp'); 

2. Load the parameters of pristine multivariate Gaussian model. 


 load modelparameters.mat;


The images used for making the current model may be viewed at http://live.ece.utexas.edu/research/quality/pristinedata.zip
 
 
3. Initialize different parameters

 Height of the block
 blocksizerow = 96;
 Width of the block
 blocksizecol = 96;
 Verical overlap between blocks
 blocksizerow = 0;
 Horizontal overlap between blocks
 blocksizecol = 0;

   For good performance, it is advisable to divide the distorted image in to same size patched as used for the construction of multivariate Gaussian model.

3. Call this function to calculate the quality score:

 
 qualityscore = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam)

Sample execution is also shown through example.m


=======================================================================

MATLAB files: (provided with release): example.m, computefeature.m, computemean.m, computequality.m, estimateaggdparam.m and estimatemodelparam.m 

Image Files: image1.bmp, image2.bmp, image3.bmp and image4.bmp

Dependencies: Mat file: modelparameters.mat provided with release

=======================================================================

Note on training: 
This release version of NIQE was trained on 125 pristine images with patch size set to 96X96 and sharpness threshold of 0.75.

Training the model

If the user wants to retrain the model using different set of pristine image or set the patch sizes to different values, he/she can do so
use the following function. The images used for making the current model may be viewed at http://live.ece.utexas.edu/research/quality/pristinedata.zip

Folder containing the pristine images 
folderpath   = 'pristine'
Height of the block
blocksizerow = 96;
Width of the block
blocksizecol = 96;
Verical overlap between blocks
blocksizerow = 0;
Horizontal overlap between blocks
blocksizecol = 0;
The sharpness threshold level
sh_th        = 0.75;


[mu_prisparam cov_prisparam]  = estimatemodelparam(folderpath,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,sh_th)
=======================================================================
 