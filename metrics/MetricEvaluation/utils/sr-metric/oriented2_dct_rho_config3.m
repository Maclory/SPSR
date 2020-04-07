function g2 = oriented2_dct_rho_config3(I)

%H=fspecial('gaussian',[7 7]);

nn=size(I,1);

temp=dct2(I);
eps=0.00000001;

temp2=[];
if nn==5
    temp2=[temp(2,2),temp(3,3:4),temp(4,3:end),temp(5,4:end)];
elseif nn==7
    temp2=[temp(2,2) temp(3,3:4) temp(4,3:5)... 
        temp(5,4:end) temp(6,5:end) temp(7,5:end)];
elseif nn==9
    temp2=[temp(2,2) temp(3,3:4) temp(4,3:5) temp(5,4:7)...
        temp(6,5:8) temp(7,5:end) temp(8,6:end) temp(9,7:end)]; 
end

std_gauss=std(abs(temp2(:)));
mean_abs=mean(abs(temp2(:)));
g2=std_gauss/(mean_abs+eps);
