function x = map_matrix_to_closest_vec(matrix,vec)
% Function to map the values in the matrix to a value from the vector so
% that the values in x are the same as those in vec
% vec is a 1xm vector and x is a pxq matrix
[nrows ncolms ] = size(matrix);
x = zeros(size(matrix));
if(nrows<=ncolms)
    for i = 1:nrows
        curr = matrix(i,:)';
        [minval ind] = min(abs(repmat(curr,[1,length(vec)])-repmat(vec,([ncolms 1]))),[],2);
        x(i,:) = vec(ind);
    end
elseif(nrows>ncolms)
    for i = 1:ncolms
        curr = matrix(:,i);
        [minval ind] = min(abs(repmat(curr,[1,length(vec)])-repmat(vec,([nrows 1]))),[],2);
        x(:,i) = vec(ind);
    end
end