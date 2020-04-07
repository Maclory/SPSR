function [ feature ] = block_dct( im)
%   
%   I: INPUT IMAGE
%   nSc: number of scale
%   feature: Returned feature on Block_DCT statistics

% shape parameter estimation on blocks and pool them together, 
% select the first 10th,last 10th and mean as featurs

gama_L1 = blkproc(im,[3,3],[2,2],@gama_dct);
gama_sorted_temp = sort(gama_L1(:),'ascend');
gama_count = length(gama_sorted_temp);
p10_gama_L1=mean(gama_sorted_temp(1:ceil(gama_count*0.1)));
% p10_last_gama_L1=mean(gama_sorted_temp(fix(gama_count*0.9):end));
p100_gama_L1=mean(gama_sorted_temp(:));
clear gama_sorted_temp gama_count

feature=[p10_gama_L1 p100_gama_L1];

% coefficient variation estimation on blocks and pool them together, 
% select the first 10th,last 10th and mean as featurs
coeff_var_L1 = blkproc(im,[3,3],[2,2],@coeff_var_dct);
cv_sorted_temp = sort(coeff_var_L1(:),'ascend');
cv_count = length(cv_sorted_temp);
% p10_cv_L1=mean(cv_sorted_temp(1:ceil(cv_count*0.1)));
p10_last_cv_L1=mean(cv_sorted_temp(fix(cv_count*0.9):end));
p100_cv_L1=mean(cv_sorted_temp(:));
clear cv_sorted_temp cv_count

feature=[feature p10_last_cv_L1 p100_cv_L1];

ori1_rho_L1 = blkproc(im,[3,3],[2,2],@oriented1_dct_rho_config3);
ori2_rho_L1 = blkproc(im,[3,3],[2,2],@oriented2_dct_rho_config3);
ori3_rho_L1 = blkproc(im,[3,3],[2,2],@oriented3_dct_rho_config3);
temp_size=size(ori1_rho_L1);
var_temp=zeros(temp_size);
    
for i=1:temp_size(1)
    for j=1:temp_size(2)
        var_temp(i,j)=var([ori1_rho_L1(i,j) ori2_rho_L1(i,j) ori3_rho_L1(i,j)]);
   end
end
ori_rho_L1=var_temp;

ori_sorted_temp = sort(ori_rho_L1(:),'ascend');
ori_count = length(ori_sorted_temp);
% p10_orientation_L1=mean(ori_sorted_temp(1:ceil(ori_count*0.1)));
p10_last_orientation_L1=mean(ori_sorted_temp(fix(ori_count*0.9):end));
p100_orientation_L1=mean(ori_sorted_temp(:));
clear var_ori_sorted_temp rho_count

feature=[feature p10_last_orientation_L1 p100_orientation_L1];

end

