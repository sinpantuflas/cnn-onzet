function onset_train()
% Training script. It loads raw audio files, process them and arranges them
% in a structure wihch is then fed to a CNN for training. The trained model
% is then saved. Run ONSET_TEST for evaluation

cleanDir = 'path/To/Leveau-Onset/Leveau/sounds/';
labelDir = 'path/To/Leveau-Onset/Leveau/labelsPL/';


try
    load('audio_database.m');
catch
    %Extract input data and labels
    [im1 lb1] = makedata(cleanDir, labelDir);
    
    %Input data is composed of spectrograms at different time resolutions
    imdb.images.data = []; n00=1; imdb.images.label = [];
    for n0=1:size(im1,2),
        tmpcell = im1{n0};
        tmplab = lb1{n0};
        for n1=1:size(tmpcell,4),
            imdb.images.data(:,:,:,n00)=tmpcell(:,:,:,n1);
            imdb.images.id(n00) = n00;
            imdb.images.label(n00) = tmplab(n1);
            imdb.images.set(n00) = 1;
            n00 = n00 + 1;
        end
    end
    
    valset = randi(length(imdb.images.label),1,floor(length(imdb.images.label).*.20));
    imdb.images.set(valset)=2;
    save('audio_database.m','imdb');
end

thisNet = initializeOnset(0); %Define network architecture.
thisNet = vl_simplenn_tidy(thisNet); %Make sure is in the appropiate format
vl_simplenn_display(thisNet) %Show network


tic
%% Training and options
trainOpts.batchSize = 250;
trainOpts.numEpochs = 200;
trainOpts.continue = true ;
%trainOpts.residualnet = 1;
%trainOpts.useGpu = true ;

lr=0.05; %The learning rate
for n=1:(trainOpts.numEpochs-1)
    lr=[lr lr(end)/1.00004]; %Reduce learning rate after each epoch
end

[net,info] = cnn_train_normal(thisNet, imdb, @getBatch, trainOpts);

disp('Saving the net'); timeStamp = datestr(now,'mm-dd-HH-MM');
save(strcat('learned_models/net-onzet-',timeStamp,'.mat'), '-struct', 'net') ;


end

function [ims lbs]= makedata(currDirName, currLabelDir)
%%
% Current data
trailing = 1;

currDir = dir(currDirName);
list=dir(currLabelDir); lists=cell(1); n000=0;
for n00=1:size(list,1),
    if ~list(n00).isdir,
        n000=n000+1;
        lists{n000}=list(n00).name(1:end-4);
    end
end
ims = cell(1); n2=1; nn=1; lbs = cell(1);
for n0 = 1:size(currDir,1),
    im = [];
    lb = [];
    if ~currDir(n0).isdir,
        % Read input signal
        [a fs] = audioread(strcat(currDirName, currDir(n0).name));
        a = a(:,1); %Take only left channel.
        a = (a - mean(a))./std(a);
        a = a./max(abs(a));
        
        % Calculate the spectrum
        nfft_s = 4096;
        win_s1 = ceil(0.023.*fs);
        win_s2 = ceil(0.046.*fs);
        win_s3 = ceil(0.093.*fs);
        olap_s1 = floor(win_s1 - ceil(0.010.*fs));
        olap_s2 = floor(win_s2 - ceil(0.010.*fs));
        olap_s3 = floor(win_s3 - ceil(0.010.*fs));
        
        %noiseTh = -80;
        
        [sp1 fp1 tp1] = spectrogram(a,win_s1,olap_s1,nfft_s, fs);
        [sp2 fp2 tp2] = spectrogram(a,win_s2,olap_s2,nfft_s, fs);
        [sp3 fp3 tp3] = spectrogram(a,win_s3,olap_s3,nfft_s, fs);
        
        %Optionally, the spectral flux can be calculated insted of the mel
        %bank representation.
        fluxed=0; difff2=1;
        if fluxed,
            sp1 = diff([sp1(:,1),sp1],1,2);
            sp2 = diff([sp2(:,1),sp2],1,2);
            sp3 = diff([sp3(:,1),sp3],1,2);
            if difff2,
                sp1 = diff([sp1(:,1),sp1],1,2);
                sp2 = diff([sp2(:,1),sp2],1,2);
                sp3 = diff([sp3(:,1),sp3],1,2);
            end
        end
        
        mfl = 27.5; mfh = 16e3; nfb=80;
        %mflh =  ceil([mfl mfh].*nfft_s/2./(fs/2));
        [mfb mc mn mx] = melbankm(nfb,nfft_s,fs,mfl/fs,mfh./fs,'f');
        
        until = min([size(sp1,2) size(sp2,2) size(sp3,2)]);
        z1=zeros(nfb,size(sp1,2)); z2=z1; z3=z1;
        for n3=1:until,
            z1(:,n3) = log10(mfb*((sp1(mn:mx,n3)).*conj(sp1(mn:mx,n3))));
            z2(:,n3) = log10(mfb*((sp2(mn:mx,n3)).*conj(sp2(mn:mx,n3))));
            z3(:,n3) = log10(mfb*((sp3(mn:mx,n3)).*conj(sp3(mn:mx,n3))));
        end
        
        % Get labels
        ixl=find(strcmpi(lists,currDir(n0).name(1:end-4)));
        if (isempty(ixl)),
            disp('Error');
            break;
        end
        timePos = load(strcat(currLabelDir,lists{ixl},'.mat'));
        timePos.labels_sample=timePos.labels_time.*fs;
        
        %Positive examples
        tixpos=[];
        for n4=1:size(timePos.labels_sample,1),
            [tf tix] = min(abs([tp1 - timePos.labels_time(n4)]));
            if tix <= 7,
                tixx = [1:15];
            elseif tix >= until-7,
                tixx = [until-14:until];
            else
                tixx = [tix-7:tix+7];
            end
            im(:,:,1,n4) = z1(:,tixx)';
            im(:,:,2,n4) = z2(:,tixx)';
            im(:,:,3,n4) = z3(:,tixx)';
            lb(n4)=1;
            tixpos=[tixpos; tix];
        end
        
        %False examples
        ftix= 1:size(tp1,2);
        if trailing,
            tixpos = [tixpos; tixpos-1; tixpos+1;...
                tixpos-2; tixpos+2;...
                tixpos-3; tixpos+3;...
                tixpos-4; tixpos+4;...
                tixpos-5; tixpos+5;...
                tixpos-6; tixpos+6;...
                tixpos-7; tixpos+7];
        else
            tixpos = [tixpos; tixpos-1; tixpos+1;...
                tixpos-2; tixpos+2;...
                tixpos-3; tixpos+3];
        end
        tixpos(tixpos<1)=[];
        tixpos(tixpos>size(tp1,2))=[];
        ftix(tixpos)=[];
        
        %Get just an equilibrated dataset.
        equildata = false;
        poscount = n4;
        if equildata,
            getpos=randi([1 length(ftix)],poscount,1); ftixx=ftix(getpos);
        else
            ftixx=ftix;
        end
        ftixx=sort(ftixx);
        ftixx=ftixx(:);
        
        for n5=1:size(ftixx,1),
            ptix=ftixx(n5);
            if ptix <= 7,
                ptixx = [1:15];
            elseif ptix >= until-7,
                ptixx = [until-14:until];
            else
                ptixx = [ptix-7:ptix+7];
            end
            im(:,:,1,poscount+n5) = z1(:,ptixx)';
            im(:,:,2,poscount+n5) = z2(:,ptixx)';
            im(:,:,3,poscount+n5) = z3(:,ptixx)';
            lb(poscount+n5)=0;
        end
        ims{nn} = im; lbs{nn}=lb; nn=nn+1;
        clear tixx ptixx until tixpos ftixx ftix ptix
    end %if isdir
end %all dir files

end

function [im, labels] = getBatch(imdb, batch)
%GETBATCH is called on every iteration in the current epoch.
isRGB = false;

im = single(imdb.images.data(:,:,:,batch));

labels = imdb.images.label(batch);
end
