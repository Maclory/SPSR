function [ score ] = quality_predict( im )

load model.mat

rf=model.rf;

[f1,f2,f3]=feature_all(im);

s1=regRF_predict(reshape(f1, 1,[]),rf{1});
s2=regRF_predict(reshape(f2, 1,[]),rf{2});
s3=regRF_predict(reshape(f3, 1,[]),rf{3});

score=[1 s1 s2 s3]*model.linear;

end

