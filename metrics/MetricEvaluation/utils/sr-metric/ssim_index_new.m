function [mssim, ssim_map, mcs, cs_map,sigma1_sq,mu1] = ssim_index_new(img1, img2, K, wind)

if (nargin < 2 | nargin > 4)
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) | (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   wind = fspecial('gaussian', 11, 1.5);	%
   K(1) = 0.01;										% default settings
   K(2) = 0.03;										%
end

if (nargin == 3)
   if ((M < 11) | (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   wind = fspecial('gaussian', 11, 1.5);
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 4)
   [H W] = size(wind);
   if ((H*W) < 4 | (H > M) | (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

C1 = (K(1)*255)^2;
C2 = (K(2)*255)^2;
wind = wind/sum(sum(wind));
% all 'same' s were 'valid' --anush
mu1   = filter2(wind, img1, 'valid');
mu2   = filter2(wind, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(wind, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(wind, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(wind, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
   cs_map = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
	denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
   
   cs_map = ones(size(mu1));
   index = denominator2 > 0;
   cs_map(index) = numerator2(index)./denominator2(index);
end

mssim = mean2(ssim_map);
mcs = mean2(cs_map);

return