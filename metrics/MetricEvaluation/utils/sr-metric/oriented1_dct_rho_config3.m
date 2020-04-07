function g1 = oriented1_dct_rho_config3(I)

%H=fspecial('gaussian',[7 7]);

temp=dct2(I);
nn=size(I,1);
eps=0.00000001;

%% 3x3
% temp1=[temp(1,3) temp(2,3)];
%% 5x5
% F = [0 1 1 1 1;0 0 1 1 1; 0 0 0 0 1; 0 0 0 0 0;0 0 0 0 0];
% temp1=temp(F~=0);

temp1=[];
if nn==5
    temp1=[temp(1,2:end) temp(2,3:end) temp(3,end)];
elseif nn==7
    temp1=[temp(1,2:end) temp(2,3:end) temp(3,5:end) temp(4,6:end)];
elseif nn==9
    temp1=[temp(1,2:end) temp(2,3:end) temp(3,5:end) temp(4,6:end) temp(5,8:end) temp(6,9:end)];
end

std_gauss=std(abs(temp1(:)));
mean_abs=mean(abs(temp1(:)));
g1=std_gauss/(mean_abs+eps);


