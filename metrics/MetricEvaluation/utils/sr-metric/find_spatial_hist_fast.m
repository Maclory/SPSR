function [b rho ypred matrix xval] = find_spatial_hist_fast(x)
% Function to find the spatial histogram for IMAGE and then compute Corr
% and fit 3-parameter logistic. returns logistic fitting value.
[nrows ncolms] = size(x);
 im = x;
 nbins = 100;
 max_x = 25;
 ss_fact = 4; 
 [hval bins] = hist(x(:),nbins);
 x = map_matrix_to_closest_vec(im,bins);
for j = 1:ss_fact:max_x
    matrix = zeros(nbins);
    
    for i = 1:nbins
        
        [r c] = find(x==bins(i));
        
        if(isempty(r)~=1)
           h  = zeros(1,nbins);

            ctemp = c-j;
            rtemp = r;rtemp(ctemp<1) = [];
            ctemp(ctemp<1) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
            if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            ctemp = c+j; rtemp = r;rtemp(ctemp>ncolms) = [];
            ctemp(ctemp>ncolms) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
             if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            rtemp = r-j;ctemp = c;ctemp(rtemp<1) = [];
            rtemp(rtemp<1) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
            if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            rtemp = r+j;
            ctemp = c;ctemp(rtemp>nrows) = [];
            rtemp(rtemp>nrows) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
            if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            rtemp = r+j;ctemp = c;ctemp(rtemp>nrows) = [];rtemp(rtemp>nrows) = [];
            ctemp = ctemp+j;rtemp(ctemp>ncolms) = []; ctemp(ctemp>ncolms) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
            if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            rtemp = r+j;ctemp = c;ctemp(rtemp>nrows) = [];rtemp(rtemp>nrows) = [];
            ctemp = ctemp-j; rtemp(ctemp<1) = [];ctemp(ctemp<1) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
            if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            rtemp = r-j;ctemp = c;ctemp(rtemp<1) = [];rtemp(rtemp<1) = [];
            ctemp = ctemp+j;rtemp(ctemp>ncolms) = []; ctemp(ctemp>ncolms) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
             if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            rtemp = r-j;ctemp = c; ctemp(rtemp<1) = []; rtemp(rtemp<1) = [];
            ctemp = ctemp-j;rtemp(ctemp<1) = []; ctemp(ctemp<1) = [];
            ind = sub2ind(size(im),rtemp,ctemp);
             if(~isempty(ind))
                h = h+hist(im(ind),bins);
            end

            matrix(i,:) = h;
        end

    end
    X = [bins]'; Y = X; matrix = matrix/sum(sum(matrix));
    px = sum(matrix,2); py = sum(matrix,1)';
    mu_x = sum(X.*px); mu_y = sum(Y.*py);
    sigma2_x = sum((X-mu_x).^2.*px);    sigma2_y = sum((Y-mu_x).^2.*py);
    [xx yy] = meshgrid(X,Y);
    rho(j) = (sum(sum(xx.*yy.*matrix))-mu_x.*mu_y)/(sqrt(sigma2_x)*sqrt(sigma2_y));

end

rho = [rho(1:ss_fact:end)];
xval = [1:ss_fact:max_x];
% b = nlinfit([0 1:4:100],rho,@fit_spatial_corr,ones(1,5));
pp = polyfit([1:ss_fact:max_x],rho,3);
%  ypred = fit_spatial_corr(b,1:4:100);
ypred = polyval(pp,[1:ss_fact:max_x]);
err = sum((ypred-rho).^2);
b = [pp err];
% 
% figure
% plot(rho,'r','LineWidth',3); hold on
% plot(ypred,'k'); title(['Error:',num2str( sum((ypred-rho).^2))]);
% toc
% subplot(1,3,1)
% imagesc(log(matrix+1));colormap(gray);axis xy
% subplot(1,3,2)
% bar(bins,sum(matrix,1)); axis([1 255 0 max(sum(matrix,1))])
% subplot(1,3,3)
% bar(bins,sum(matrix,2));axis([1 255 0 max(sum(matrix,2))])
