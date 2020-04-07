function gama =gama_gen_gauss(I)

mean_gauss=mean(I(:));
var_gauss=var(I(:));
mean_abs=mean(abs(I(:)-mean_gauss))^2;
rho=var_gauss/(mean_abs+0.0000001);

g=0.03:0.001:10;
r=gamma(1./g).*gamma(3./g)./(gamma(2./g).^2);
    
% gamma_gauss=11;
% for j=1:length(g)-1
%     if rho<=r(j) && rho>r(j+1)
%         gamma_gauss=g(j);
%         break
%     end
% end
% gama=gamma_gauss;

[~,idx]=min(abs(r-rho));
gama=g(idx);
