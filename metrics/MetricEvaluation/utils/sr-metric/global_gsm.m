function [ feature ] = global_gsm( img )
%   Blobal_GSM used to extract global statiscs information of SR image 


if(ndims(img)==3)
    img=rgb2gray(img);
end

im=double(img);

%% Constants
num_or = 6;
num_scales = 2;

f=[];

[pyr, pind] = buildSFpyr(im,num_scales,num_or-1);
[subband, size_band] = norm_sender_normalized(pyr,pind,num_scales,num_or,1,1,3,3,50);

% gama of each subband
gama_horz = zeros(1,length(subband));
for ii = 1:length(subband)
    t = subband{ii}; 
    gama_horz(ii) = gama_gen_gauss(t);    
end
f = [f, gama_horz];

% gama across scales
gama_scale=zeros(1,length(subband)/2);
for ii = 1:length(subband)/2
    t = [subband{ii}; subband{ii+num_or}];
    gama_scale(ii) = gama_gen_gauss(t);
end
f = [f, gama_scale];

% % global gama across scales 
% t = cell2mat(subband');
% gama_global = gama_gen_gauss(t);
% 
% f = [f, gama_global];

% structural correlation between scales
hp_band = pyrBand(pyr,pind,1);
cs_val=zeros(1,length(subband));
for ii = 1:length(subband)
    curr_band = pyrBand(pyr,pind,ii+1);
    [~, ~, cs_val(ii)] = ssim_index_new(imresize(curr_band,size(hp_band)),hp_band);
end
f = [f, cs_val];

% strctural correlation between orientations
clear cs_val;
nn = 1; 
for i = 1:length(subband)/2
    for j = i+1:length(subband)/2
      [~, ~, cs_val(nn)] = ssim_index_new(reshape(subband{i},size_band(i,:)),reshape(subband{j},size_band(j,:)));  
      nn = nn + 1;
    end
end

f = [f, cs_val];
feature=f;

end