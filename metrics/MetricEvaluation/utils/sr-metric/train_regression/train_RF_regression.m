

clc
clear
close all;

rng('default');

% load extracted features

load ../FeatureSet/feature_proposed.mat

% feat{i,j} is a extracted feature vector, where i is the index of image 
% and j (=3) is the type of features (dct, gsm, pca)


load ../SubjectData/SubjectScore_Final_RightOrderPSNR.mat



ss=mean40(:);
    
id_all=randperm(180*8);

sshat=zeros(180*8,3);

rf=cell(3,1);

for jj=1:3

    ft=feature(:,jj);
    ft=[ft{:}]';

    X_trn=ft;
    Y_trn=ss;

    rf{jj} = regRF_train(X_trn,Y_trn, 2000);
    sshat(:,jj) = regRF_predict(X_trn,rf{jj});

end

B=robustfit(sshat,ss);

model.rf=rf;

model.linear=B;

save('model.mat', 'model');
