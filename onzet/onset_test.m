function onset_test
% ONSET_TEST retrieves an example from the Prosemus Onset Database and
% evaluates a previously trained CNN model for Onset detection in audio
% signals.
%
%
% Author: C. de Obaldia.

% Set path to DB
pathToDB = '/media/carlos/INTENSO/obaldia/Proyectos/Sounds/Prosemus-Onset/';

% Setup matconvnet and compile MEX files. If GPU support is available, set
% opts.useGPU in matconvnet_dist/setup.m to true.
run matconvnet_dist/setup;


inDir = pwd;
if ~strcmp(inDir(end-4:end),'onzet'),
    if strcmp(inDir(end-2:end),'one'),
        cd(strcat(pwd,'/cnn_onzet/'));
    end
else
    disp('Please change your path..');
end
addpath('../');

testDir = strcat(pathToDB,'audio/');
lablDir = strcat(pathToDB,'ground_truth/');

%File name for the test file. Check available files in audio folder
testName = '6-three.wav';

% Flag for indicating if test file has annotations
has_gt = 1;
% Flag for plotting test results
plotit = 1;

[a fs] = audioread(strcat(testDir, testName));
a = a(:,1); %Take only left channel.
a = (a - mean(a))./std(a);
a = a./max(abs(a));

% Resample to a constant sample rate, for consistency accross training and
% test sets.
nfs=44100;
a = resample(a,nfs,fs);

% The input to the neural network contains several time-frequency
% representations which are introduced as input to the neural network.
% These (multi-modal) representations contain different time resolutions,
% so that the network can be aware of the temporal structure of the audio
% file.

% Calculate the spectrum
nfft_s = 4096;
win_s1 = ceil(0.023.*nfs);
win_s2 = ceil(0.046.*nfs);
win_s3 = ceil(0.093.*nfs);
olap_s1 = floor(win_s1 - ceil(0.010.*nfs));
olap_s2 = floor(win_s2 - ceil(0.010.*nfs));
olap_s3 = floor(win_s3 - ceil(0.010.*nfs));

%noiseTh = -80;

[sp1 fp1 tp1] = spectrogram(a,win_s1,olap_s1,nfft_s, nfs);
[sp2 fp2 tp2] = spectrogram(a,win_s2,olap_s2,nfft_s, nfs);
[sp3 fp3 tp3] = spectrogram(a,win_s3,olap_s3,nfft_s, nfs);

if plotit,
    figure(2),clf;
    ax1=subplot(3,1,1);
    title('Original spectrograms');
    imagesc(tp1,(fp1./1000),20*log10(abs(sp1)));
    axis xy; xlabel('t (s) \rightarrow'); ylabel('f (kHz) \rightarrow');
    %title('23ms Hop size');
    ax2=subplot(3,1,2);
    imagesc(tp1,(fp1./1000),20*log10(abs(sp2)));
    axis xy; xlabel('t (s) \rightarrow'); ylabel('f (kHz) \rightarrow');
    %title('46ms Hop size');
    ax3=subplot(3,1,3);
    imagesc(tp1,(fp1./1000),20*log10(abs(sp3)));
    axis xy; xlabel('t (s) \rightarrow'); ylabel('f (kHz) \rightarrow');
    %title('93ms Hop size');
end

mfl = 27.5; mfh = 16e3; nfb=80;
%mflh =  ceil([mfl mfh].*nfft_s/2./(nfs/2));
[mfb mc mn mx] = melbankm(nfb,nfft_s,nfs,mfl/nfs,mfh./nfs,'f');

until = min([size(sp1,2) size(sp2,2) size(sp3,2)]);
z1 = zeros(nfb,size(sp1,2)); z2=z1; z3=z1;
for n3=1:until,
    z1(:,n3) = log10(mfb*((sp1(mn:mx,n3)).*conj(sp1(mn:mx,n3))));
    z2(:,n3) = log10(mfb*((sp2(mn:mx,n3)).*conj(sp2(mn:mx,n3))));
    z3(:,n3) = log10(mfb*((sp3(mn:mx,n3)).*conj(sp3(mn:mx,n3))));
end

im = zeros(size(z1,2),size(z1,1),3);
im(:,:,1) = z1(1:size(z1,1),1:size(z1,2))';
im(:,:,2) = z2(1:size(z1,1),1:size(z1,2))';
im(:,:,3) = z3(1:size(z1,1),1:size(z1,2))';

im=single(im);

if plotit,
    figure(3),clf;
    ax4=subplot(3,1,1);
    title('Mel-filtered spectra');
    imagesc(tp1,1:80,abs(z1));
    axis xy; xlabel('t \rightarrow'); ylabel('Mel bins');
    ax5=subplot(3,1,2);
    imagesc(tp1,1:80,abs(z2));
    axis xy; xlabel('t \rightarrow'); ylabel('Mel bins');
    ax6=subplot(3,1,3);
    imagesc(tp1,1:80,abs(z3));
    axis xy; xlabel('t \rightarrow'); ylabel('Mel bins');
    linkaxes([ax1,ax2,ax3,ax4,ax5,ax6],'x');
end

if plotit
    imR=imrotate(im,-90,'bilinear');
    figure(4);
    imagesc(tp1,(fp1./1000),imR);
    axis xy;
    title('Input image to the CNN')
    xlabel('t (s) \rightarrow'); ylabel('f (kHz) \rightarrow');
    ax7 = gca;
    linkaxes([ax1,ax4,ax7],'x');
end

netDir = 'learned_models/';

%thisNet = 'net-onzet-10-05-16-54.mat';  %trained on Softmax
%thisNet = 'net-onzet-10-05-18-51.mat';  %Sigmoid max, wo relu
%thisNet = 'net-onzet-10-06-13-04.mat';  % '' w relu (not on fc layers)
%thisNet = 'net-onzet-10-06-14-20.mat';  %Just one relu on the fc layers%

thisNet = 'net-onzet-10-06-16-28.mat';  %ReLu after each conv layer. Solved a problem with smeared trainsients
%thisNet = 'net-onzet-10-07-15-34.mat'; %relu everywhere, no trailing on frames. Peks appear not so prominent, but a probabiliy of 0.5 could make the onsets appear. On the smeared parts, the network tries to identify onsets at the end of the transient.
%thisNet = 'net-onzet-10-07-17-04.mat'; %no relus, no trailing. Higher "probability floor" (could be handles by log - post processing)
%thisNet = 'net-onzet-10-07-18-31.mat'; %no relus, 500 ephocs; no improvements, higher sensitivity to smear drums
%thisNet = 'net-onzet-10-10-11-02.mat'; %Some improvement in prominence of peaks
%thisNet = 'net-onzet-10-10-12-06.mat';
%thisNet = 'net-onzet-10-10-13-46.mat'; %just one class

thisNet = strcat(netDir,thisNet);

net = load(thisNet);
net = vl_simplenn_tidy(net);
net.layers(end) = [];
%net.layers{end+1} =  struct('type', 'softmax') ;
vl_simplenn_display(net)
[res] = vl_simplenn(net, im, [],[],'mode','test');

A = res(end).x(:,1,1);

%% Post-processing

% The output of the CNN is smoothed with a smoothing filter defined by a
% hamming window
sl = 7;
win = hamming(sl);
Aa = conv(A,win,'same');
Aa = Aa(1:end-1);
AA = zeros(length(Aa),1);
AA(sl+1:end)=Aa(1:end-sl);
Aa=AA;

%And then we normalize the output vector
Aa = (Aa-min(abs(Aa(10:end-10))))./((max(abs(Aa(10:end-10))))-min(abs((Aa(10:end-10)))));

%Create a vector with the time instances, for visualization and evaluation
%purposes.
t_vec = linspace(0, (length(a)-1)./nfs, length(Aa));

if has_gt,
    % Load ground truth
    gt = load(strcat(lablDir,testName(1:end-4),'.txt'));
    % Find nearest numbers in the vector
    gt_ix=[];
    for n4=1:length(gt)
        [val ix] = min(abs(t_vec - gt(n4)));
        gt_ix=[gt_ix;ix];
    end
end
[ixs ixs0] = find_loc_max(Aa);

%Analyze results
%for each onset candidate, find if they are a true positive, or a false
%positive.
if has_gt,
    tol = 0.1;
    vals = t_vec(ixs);
    tps = []; fps = []; cor=[]; fal=[];
    for n6=1:length(vals),
        [va ti] = min(abs([gt - vals(n6)]));
        va = gt(ti);
        if abs(va - vals(n6)) <= tol,
            tps = [tps;n6];
            cor = [cor;ti];
        else
            fps = [fps;n6];
            fal = [fal;ti];
        end
    end
    %For the remaining onsets, count false negatives
    stay = sort([cor;fal]);
    remain = gt; remain(stay)=[];
end

Aa(Aa<0)=0; %Just to be sure
if plotit,
    figure(4), clf
    title('Onset detection')
    plot(t_vec,Aa,'DisplayName',strcat('Detection function')); hold on
    if has_gt, stem(t_vec(gt_ix),0.8.*ones(length(gt),1),'^','DisplayName', 'Ground truth'); end
    axis([0 t_vec(end) 0 1.2]);
    ylim([0 1]);
    xlabel('t (s) \rightarrow');
    legend show
end


cd('../')
