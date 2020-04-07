function rho =coeff_var_gen_gauss(I)

std_gauss=std(abs(I(:)));
mean_abs=mean(abs(I(:)));
rho=std_gauss/(mean_abs+0.0000001);