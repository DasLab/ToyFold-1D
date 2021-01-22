% Let's get a bunch of training data to do the most basic test --
% predict energy of a single conformation+pairing.

load 02_GenerateTrainingData.mat all_x all_p all_sequence_onehot params;
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


     
   
