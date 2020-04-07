%% Parameters
% Directory with your results
%%% Make sure the file names are as exactly %%%
%%% as the original ground truth images %%%
function evaluate_PI(input_dir, test_name)
% Set verbose option
verbose = true;

%% Calculate scores and save
addpath('utils')
scores = calc_PI(input_dir, verbose);
% Saving
save_path = fullfile([input_dir, '/', strcat(test_name,'.mat')])
save(save_path, 'scores');

%% Printing results
perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;
fprintf(['\n\nYour perceptual score is: ',num2str(perceptual_score)]);
%fprintf(['\nYour RMSE is: ',num2str(sqrt(mean([scores.MSE]))),'\n']);
%fprintf(['Your PSNR is: ',num2str(mean_psnr), '\n']);
end
