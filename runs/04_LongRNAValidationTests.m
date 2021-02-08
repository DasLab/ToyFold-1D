%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test on some different-length RNA's... looks OK:
figure(1)
params.epsilon = -2; params.delta = 5; 
nts = 'ACGU';
N = 17;
sequences_long = {};
all_sequence_onehot_long = []; all_x_long = []; all_p_long = []; all_bpp_long = [];
Ndata = 8;
for n = 1:Ndata
    rand_seq_by_numbers = randi(4,1,N);
    rand_sequence = nts( rand_seq_by_numbers );
    sequences_long{n} = rand_sequence;

    fprintf( 'Doing %d out of %d...\n',n,Ndata );
    tic; [x,p,~,E,bpp] = analyze_sequence( rand_sequence, params, 0); toc
    
    % for MFE, pick at random if multiple structures have MFE energy.
    idx = find( E == min(E) );
    idx = idx( randi( length(idx) ) );
    all_x_long(:,n)        = x(:,idx);
    all_p_long(:,n)        = p(:,idx);
    all_bpp_long(:,:,n)    = bpp;
    sequence_onehot = zeros(N,4);
    for i = 1:N; sequence_onehot( i,rand_seq_by_numbers(i) ) = 1; end;
    all_sequence_onehot_long(:,:,n) = sequence_onehot;
end

%%


%%
D2_long = zeros(14,14,1,1);
for n = 1:length(sequences_long)
    for m = 1:4
        D2_long(1:N,1:N,2*m-1,n) = repmat(all_sequence_onehot_long(:,m,n) ,[1 N]);
        D2_long(1:N,1:N,2*m,  n) = repmat(all_sequence_onehot_long(:,m,n)',[N 1]);
    end
end
BPP_pred_long = predict( BPPnet, D2_long );

%%
figure(2);
for i = 1:Ndata
    subplot(Ndata,2,2*(i-1)+1)
    imagesc( all_bpp_long(:,:,i));
    %subplot(Ndata,2,2*(i-1)+2)
    %imagesc( BPP_pred_long(:,:,1,i));
end