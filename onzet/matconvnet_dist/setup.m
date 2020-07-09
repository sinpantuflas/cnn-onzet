function setup(varargin)

%<<< In HEAD
%run matconvnet-master/matlab/vl_setupnn ;
%addpath matconvnet-master/examples ;
%>>> 

run vlfeat/toolbox/vl_setup ;

%run ../ant-cnn/matconvnet19/matlab/vl_setupnn;
%addpath ../ant-cnn/matconvnet19/examples ;

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath matconvnet/matlab ;

opts.useGpu = false;
opts.verbose = true ;
opts = vl_argparse(opts, varargin) ;



if opts.useGpu
  try
    vl_nnconv(gpuArray(single(1)),gpuArray(single(1)),[]) ;
  catch
    vl_compilenn('enableGpu', opts.useGpu, 'verbose', opts.verbose) ;
    warning('GPU support does not seem to be compiled in MatConvNet. Trying to compile it now') ;
  end
else
    try
  vl_nnconv(single(1),single(1),[]) ;
catch
  warning('VL_NNCONV() does not seem to be compiled. Trying to compile it now.') ;
  vl_compilenn('enableGpu', opts.useGpu, 'verbose', opts.verbose) ;
end
end
