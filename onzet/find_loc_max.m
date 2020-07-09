function [idx idx0] = find_loc_max(x)
%
%
%
%

N = length(x);
dx = diff(x);
dx1 = dx(2:N-1);
dx2 = dx(1:N-2);
prod = dx1.*dx2;
idx1 = find(prod<0);
idx2 = find(dx1(idx1)<0);
idx = idx1(idx2)+1;

%zeros in dx? maxima with 2 identical values
idx3 = find(dx==0);
idx4 = find(x(idx3)>0);
idx0 = idx3(idx4);

%positions of double maxima, same values at idx3(idx4)+1
if nargout==1
    idx = sort([idx;idx0]);
end