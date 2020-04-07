function g3 = oriented3_dct_rho_config3(I)

%H=fspecial('gaussian',[7 7]);

nn=size(I,1);

eps=0.00000001;
temp=dct2(I);


temp3=[];
if nn==5
    temp3=[temp(2:end,1); temp(3:end,2); temp(end,3)];
elseif nn==7
    temp3=[temp(2:end,1); temp(3:end,2); temp(5:end,3); temp(6:end,4)];
elseif nn==9
    temp3=[temp(2:end,1); temp(3:end,2); temp(5:end,3); temp(6:end,4);...
        temp(8:end,5); temp(9:end,6)];
end
std_gauss=std(abs(temp3));
mean_abs=mean(abs(temp3));
g3=std_gauss/(mean_abs+eps);
