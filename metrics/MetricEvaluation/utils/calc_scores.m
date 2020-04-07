function scores = calc_scores(input_dir,GT_dir,shave_width,verbose)

addpath(genpath(fullfile(pwd,'utils')));

%% Loading model
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%% Reading file list
file_list = dir([input_dir,'/*.png']);
GT_list = dir([GT_dir, '/*.png']);
im_num = length(file_list);
scale = 4;
%% Calculating scores
scores = struct([]);

for ii=1:im_num
    if verbose
        fprintf(['Calculating scores for image ',num2str(ii),' / ',num2str(im_num), '\n']);
    end
    
    % Reading and converting images
    input_image_path = fullfile(input_dir,file_list(ii).name);
    input_image = convert_shave_image(imread(input_image_path),shave_width);
    GD_image_path = fullfile(GT_dir,GT_list(ii).name);
    GD_image = modcrop(imread(GD_image_path), scale);
    GD_image = convert_shave_image(GD_image,shave_width);
    
    if size(input_image) ~= size(GD_image)
      display(size(input_image), input_image_path)
      display(size(GD_image), GD_image_path)
    end
    % Calculating scores
    scores(ii).name = file_list(ii).name;
    scores(ii).MSE = immse(input_image,GD_image);
    scores(ii).Ma = quality_predict(input_image);
    scores(ii).NIQE = computequality(input_image,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    scores(ii).PSNR = psnr(input_image, GD_image);
    [scores(ii).SSIM,scores(ii).SSIM_map] = ssim(input_image, GD_image);
end

end

function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
