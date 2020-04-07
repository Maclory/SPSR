function f = dct_freq_bands(In)

I=dct2(In);
eps=0.00000001;

%% 5x5 freq band 1
var_band1=var([I(1,2) I(2,1) I(1,3) I(3,1) I(2,2)]);


%% 5x5 freq band 2
var_band2=var([I(4,1) I(5,1) I(3,2) I(4,2) I(5,2) I(2,3) I(3,3) I(4,3) I(1,4) I(2,4) I(3,4) I(1,5) I(2,5)]);

%% 5x5 freq band 3
var_band3=var([I(3,5) I(4,4) I(5,3) I(4,5) I(5,4) I(5,5)]);

r1 = abs(var_band3 - mean([var_band1 var_band2]))/(var_band3 + mean([var_band1 var_band2])+eps);
r2 = abs(var_band2 - var_band1)/(var_band3 + var_band1+eps);

f = (r1+r2)/2;