function yhat = fitting(beta,x)

b1 = beta(1);
b2 = beta(2);
b3 = beta(3);
b4 = beta(4);
b5 = beta(5);

yhat = b1*(0.5 - 1./(1+exp(b2*(x-b3)))) + b4.*x + b5;
