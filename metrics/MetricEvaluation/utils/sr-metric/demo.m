
addpath('external\matlabPyrTools','external\randomforest-matlab\RF_Reg_C');

im=imread('peppers.png');

score=quality_predict(im);