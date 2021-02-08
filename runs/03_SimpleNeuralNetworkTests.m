% Let's get a bunch of training data to do the most basic test --
% predict energy of a single conformation+pairing.

load 02_GenerateTrainingData.mat all_x all_p all_sequence_onehot params sequences all_bpp;
N = size( all_x,2);
Nres = size( all_x,1);

% base pair matrices
P = zeros(Nres,Nres,N);
for n = 1:N
    p = all_p(:,n);
    for i = find(p~=0)'
        P(i,p(i),n) = 1;
    end
end

is_chainbreak = zeros(Nres,1); is_chainbreak(Nres) = 1;
E =  get_energy(all_x,all_p,is_chainbreak,params);


%%
%% Fully Connected -- fails?
layers = [ 
    featureInputLayer([28])
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:500];
test_idx = [9001:10000];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{ x_p(train_idx,:), -E(train_idx)'}, ...
     'ValidationFrequency',30 );

% Tried negation below in case of any funny business with signs, but didn't help
train_idx = [1:1000];
net = trainNetwork( x_p(train_idx,:), -E(train_idx)', layers,options );

%%
clf
Epred = -1 * predict( net, x_p );
plot( Epred, E,'.' );
  

% predict energy E from features x and p.
% argh no unsqueeze? prepare feature vector by hand.
X = zeros(Nres,2,n);
for n = 1:N
    X(:,1,n) = all_x(:,n);
    X(:,2,n) = all_p(:,n);
end

%%
%filename = 'x_p.csv';
%csvwrite( filename, [all_x; all_p]  );

%% Convnet? Try 1D convnet with two input layers.
layers = [ 
    imageInputLayer([14 1 2])
    convolution2dLayer([3 1],10)
    reluLayer
    convolution2dLayer([3 1],10)
    reluLayer
    convolution2dLayer([3 1],10)
    reluLayer
    globalAveragePooling2dLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:500];
test_idx = [9001:10000];
X_reshape = reshape( X, [14 1 2 10000]);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{X_reshape(:,:,:,test_idx), E(test_idx)'}, ...
     'ValidationFrequency',30 );

net = trainNetwork( X_reshape(:,:,:,train_idx), E(train_idx)', layers,options );
%%
clf
Epred = predict( net, X_reshape );
plot( Epred(train_idx), E(train_idx),'.' ); hold on
plot( Epred(test_idx), E(test_idx),'.' ); hold on
%clf; plot( [E', Epred])
xlabel( 'E-pred');
ylabel( 'E-actual');
h=legend('train','test');set(h,'Location','NorthWest');
set(gcf, 'PaperPositionMode','auto','color','white');


%% hmm. Real 2D convnet
D2 = [];
D2(:,:,3,:) = reshape(P,[14 14 1 10000]);
for n = 1:N
    D2(:,:,1,n) = repmat(all_x(:,n),[1 14]);
    D2(:,:,2,n) = repmat(all_x(:,n)',[14 1]);
end

layers = [ 
    imageInputLayer([14 14 3])
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), E(test_idx)'}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), E(train_idx)', layers,options );


%%
clf
Epred = predict( net, D2 );
plot( Epred(train_idx), E(train_idx),'.' ); hold on
plot( Epred(test_idx), E(test_idx),'.' ); hold on
xlabel( 'E-pred');
ylabel( 'E-actual');
h=legend('train','test');set(h,'Location','NorthWest');
set(gcf, 'PaperPositionMode','auto','color','white');


%% What if only supply conformation (i.e., can NN infer pairing? Or does it also need sequence)
D2 = [];
for n = 1:N
    D2(:,:,1,n) = repmat(all_x(:,n),[1 14]);
    D2(:,:,2,n) = repmat(all_x(:,n)',[14 1]);
end

layers = [ 
    imageInputLayer([14 14 2])
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), E(test_idx)'}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), E(train_idx)', layers,options );


%% What if only supply pairing (i.e., can NN infer conformation?)
D2 = [];
D2(:,:,1,:) = reshape(P,[14 14 1 10000]);

layers = [ 
    imageInputLayer([14 14 1])
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), E(test_idx)'}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), E(train_idx)', layers,options );


%%
clf
Epred = predict( net, D2 );
plot( Epred(train_idx), E(train_idx),'.' ); hold on
plot( Epred(test_idx), E(test_idx),'.' ); hold on
xlabel( 'E-pred');
ylabel( 'E-actual');
h=legend('train','test');set(h,'Location','NorthWest');
set(gcf, 'PaperPositionMode','auto','color','white');


%% Revisit prediction from conformation alone (no pairings) -- put in sequence.
%% and now try to jitter
D2 = [];
for n = 1:N
    D2(:,:,1,n) = repmat(all_x(:,n),[1 14]);
    D2(:,:,2,n) = repmat(all_x(:,n)',[14 1]);
    for m = 1:4
        D2(:,:,2*m+1,n) = repmat(all_sequence_onehot(:,m,n) ,[1 14]);
        D2(:,:,2*m+2,n) = repmat(all_sequence_onehot(:,m,n)',[14 1]);
    end
end

D2 = D2 + 0.1* randn(size(D2));  % jitter

layers = [ 
    imageInputLayer([14 14 10])
    convolution2dLayer([3 3],10)
    reluLayer
    convolution2dLayer([3 3],10)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), E(test_idx)'}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), E(train_idx)', layers,options );


%% Try to directly predict pairing
D2 = [];
for n = 1:N
    D2(:,:,1,n) = repmat(all_x(:,n),[1 14]);
    D2(:,:,2,n) = repmat(all_x(:,n)',[14 1]);
    for m = 1:4
        D2(:,:,2*m+1,n) = repmat(all_sequence_onehot(:,m,n),[1 14]);
        D2(:,:,2*m+2,n) = repmat(all_sequence_onehot(:,m,n)',[14 1]);
    end
end

P = zeros(Nres,Nres,1,N);
for n = 1:N
    p = all_p(:,n);
    for i = find(p~=0)'
        P(i,p(i),1,n) = 1;
    end
end

D2 = D2 + 0.1* randn(size(D2));

% Just give it the answer for x1=x2? See tests below.
D2(:,:,11,:) =  abs(D2(:,:,1,:)-D2(:,:,2,:)); 
layers = [ 
    imageInputLayer([14 14 11])
    convolution2dLayer([3 3],10,'Padding','Same')
    reluLayer
    convolution2dLayer([1 1],10,'Padding','Same')
    reluLayer
    convolution2dLayer([1 1],1,'Padding','Same')
    reluLayer
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), P(:,:,:,test_idx)}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), P(:,:,:,train_idx), layers,options );
%%
P_pred = predict( net, D2 );
sum_err = squeeze(sum(sum(abs(P_pred-P),1),2));
clf; 
plot(sum_err)
[~,idx] = max(sum_err )

%%
subplot(1,2,1);
idx = 9447; %randi(10000);
imagesc(P(:,:,1,idx) )
title( 'P')
subplot(1,2,2);
imagesc(P_pred(:,:,1,idx));
title( 'P_{pred}')
  
%% Here's the answer (by hand)...
subplot(1,2,1);
imagesc(D2(:,:,1,10)==D2(:,:,2,10) & ...
     ( (D2(:,:,3,10)==1 & D2(:,:,10,10)==1) | ...
       (D2(:,:,5,10)==1 & D2(:,:,8,10)==1) | ...
       (D2(:,:,7,10)==1 & D2(:,:,6,10)==1) | ...
       (D2(:,:,9,10)==1 & D2(:,:,4,10)==1) ) );
  title( 'D2 conv by hand')
subplot(1,2,2);
imagesc(P(:,:,1,10) )
  title( 'P')
  
 %% Silly test -- just get a net that can predict equality!
 D2 = [];
for n = 1:N
    D2(:,:,1,n) = repmat(all_x(:,n),[1 14]);
    D2(:,:,2,n) = repmat(all_x(:,n)',[14 1]);
%     for m = 1:4
%         D2(:,:,2*m+1,n) = repmat(all_sequence_onehot(:,m,n) ,[1 14]);
%         D2(:,:,2*m+2,n) = repmat(all_sequence_onehot(:,m,n)',[14 1]);
%     end
end
Xequal = double(D2(:,:,1,:)==D2(:,:,2,:));

%D2 = D2 + 0.1*randn(14,14,2,10000);
D2(:,:,3,:) =  abs(D2(:,:,1,:)-D2(:,:,2,:)); % Just give it the answer!
%D2(:,:,3,:) =  Xequal + 0.1*randn(14,14,1,10000); % Just give it the answer with jitter
 
layers = [ 
    imageInputLayer([14 14 3])
    convolution2dLayer([3 3],1,'Padding','Same')
    reluLayer
    convolution2dLayer([3 3],1,'Padding','Same')
    reluLayer
    regressionLayer
    ];

train_idx = [1:500];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), Xequal(:,:,:,test_idx)}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), Xequal(:,:,:,train_idx), layers,options );

%%
Xequal_pred = predict( net, D2 );

subplot(1,2,1);
imagesc(Xequal(:,:,1,10) )
title( 'Xequal')
subplot(1,2,2);

imagesc(Xequal_pred(:,:,1,10) )
title( 'prediction of net');


%% Revisit energy estimate from conformation alone
% but now encode conformation as x1-x2.
D2 = [];
for n = 1:N
    x1 = repmat(all_x(:,n),[1 14]);
    x2 = repmat(all_x(:,n)',[14 1]);
    D2(:,:,1,n) = abs(x1-x2);
    for m = 1:4
        D2(:,:,2*m,n) = repmat(all_sequence_onehot(:,m,n) ,[1 14]);
        D2(:,:,2*m+1,n) = repmat(all_sequence_onehot(:,m,n)',[14 1]);
    end
end

%D2 = D2 + 0.1* randn(size(D2));  % jitter

layers = [ 
    imageInputLayer([14 14 9])
    convolution2dLayer([3 3],10)
    reluLayer
    convolution2dLayer([3 3],10)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    convolution2dLayer([3 3],2)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), E(test_idx)'}, ...
     'ValidationFrequency',30);

net = trainNetwork( D2(:,:,:,train_idx), E(train_idx)', layers,options );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Let's just try bpp matrices from sequence
D2 = [];
for n = 1:N
    for m = 1:4
        D2(:,:,2*m-1,n) = repmat(all_sequence_onehot(:,m,n) ,[1 14]);
        D2(:,:,2*m,  n) = repmat(all_sequence_onehot(:,m,n)',[14 1]);
    end
%     D2(:,:,9,n) = ...
%         (D2(:,:,1,n)==1 & D2(:,:,8,n)==1) | ...
%         (D2(:,:,3,n)==1 & D2(:,:,6,n)==1) | ...
%         (D2(:,:,5,n)==1 & D2(:,:,4,n)==1) | ...
%         (D2(:,:,7,n)==1 & D2(:,:,2,n)==1);
end

for n = 1:N
    BPP(:,:,1,n) = all_bpp(:,:,n);
end

%D2 = D2 + 0.1* randn(size(D2));  % jitter

% layers = [ 
%     imageInputLayer([14 14 9])
%     convolution2dLayer([3 3],64,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],64,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],64,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],1,'Padding','Same')
%     reluLayer
%     regressionLayer
%     ];

% layers = [ 
%     imageInputLayer([14 14 9])
%     convolution2dLayer([3 3],10,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],10,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],10,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],10,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],10,'Padding','Same')
%     reluLayer
%     convolution2dLayer([3 3],1,'Padding','Same')
%     reluLayer
%     regressionLayer
%     ];

layers = [ 
    imageInputLayer([14 14 8])
    convolution2dLayer([3 3],16,'Padding','Same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],16,'Padding','Same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],16,'Padding','Same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],16,'Padding','Same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],16,'Padding','Same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],1,'Padding','Same')
    batchNormalizationLayer
    reluLayer
    regressionLayer
    ];

train_idx = [1:5000];
test_idx = [9001:10000];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',200, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
     'ValidationData',{D2(:,:,:,test_idx), BPP(:,:,1,test_idx)}, ...
     'ValidationFrequency',30);

BPPnet = trainNetwork( D2(:,:,:,train_idx), BPP(:,:,1,train_idx), layers,options );

% tried to start with previously trained net 
%BPPnet = trainNetwork( D2(:,:,:,train_idx), BPP(:,:,1,train_idx), saveBPPnet.Layers,options );
%BPPnet = trainNetwork( D2(:,:,:,train_idx), BPP(:,:,1,train_idx), BPPnet_12layer.Layers,options );

%%
BPP_pred = predict( BPPnet, D2 );
sum_err = squeeze(sum(sum(abs(BPP_pred-BPP),1),2));
clf; 
plot(sum_err)
[~,idx] = max(sum_err )

%% Image the predictions
%idx = randi(10000);
subplot(1,2,1);
%idx = 9447; 
imagesc(BPP(:,:,1,idx) )
title( sprintf('BPP %d',idx))
subplot(1,2,2);
imagesc(BPP_pred(:,:,1,idx));
title(sprintf('BPP_{pred} %d',idx))
  
