function [ f1, f2, f3 ] = feature_all( img )

if(ndims(img)==3)
    img=rgb2gray(img);
end

im1=im2double(img);

% Scale factor is 3
h=fspecial('gaussian',3);
im_f=double(imfilter(im1,h));
im2 = im_f(2:2:end,2:2:end);

h=fspecial('gaussian',3);
im_f=double(imfilter(im2,h));
im3 = im_f(2:2:end,2:2:end);

t1=block_dct(im1);
t2=block_dct(im2);
t3=block_dct(im3);

f1=[t1 t2 t3];

f2=global_gsm(img);

col=im2col(im1,[5 5],'distinct');
t1=svd(col);
col=im2col(im2,[5 5],'distinct');
t2=svd(col);
col=im2col(im3,[5 5],'distinct');
t3=svd(col);
f3=[t1 t2 t3];

end