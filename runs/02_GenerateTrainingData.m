% Higher delta, lower epsilon results in fewer pseudoknots, as expected.
nts = 'ACGU'; N = 14; % roughly 1 sec run time.
rand_sequence = nts(randi(4,1,N));
params.epsilon = -2; params.delta = 5; % try to get more cooperativity
tic; [x,p] = analyze_sequence( rand_sequence, params, 0); toc

%%
params.epsilon = -2; params.delta = 5; % try to get more cooperativity
%sequence = 'CAGAAGGGCUGAACCC'; % now need 3-bp to get stable stems
sequence = 'CAGAAGGGCUGAACCC'; % this looks good! 16 nts though. Can we get it shorter?
%sequence = 'CAGAAGGGCUGAACCC'; % now need 3-bp to get stable stems
tic; [x,p,~,E] = analyze_sequence( sequence, params, 0 ); toc;


%%
% try to get more cooperativity.
% no pseudoknots in MFE possible in 14mer, but
%  may contribute to BPP.
params.epsilon = -2; params.delta = 5; 
N = 14;
sequences = {};
all_sequence_onehot = []; all_x = []; all_p = []; all_bpp = [];
Ndata = 10000;
for n = 1:Ndata
    rand_seq_by_numbers = randi(4,1,N);
    rand_sequence = nts( rand_seq_by_numbers );
    sequences{n} = rand_sequence;

    fprintf( 'Doing %d out of %d...\n',n,Ndata );
    tic; [x,p,~,E,bpp] = analyze_sequence( rand_sequence, params, 0); toc
    
    % for MFE, pick at random if multiple structures have MFE energy.
    idx = find( E == min(E) );
    idx = idx( randi( length(idx) ) );
    all_x(:,n)        = x(:,idx);
    all_p(:,n)        = p(:,idx);
    all_bpp(:,:,n)    = bpp;
    sequence_onehot = zeros(N,4);
    for i = 1:N; sequence_onehot( i,rand_seq_by_numbers(i) ) = 1; end;
    all_sequence_onehot(:,:,n) = sequence_onehot;
end

%% show base pair plots
for n = 1:25
    subplot(5,5,n);
    imagesc(all_bpp(:,:,n),[0 1] );
    p = all_p(:,n);
    for i = 1:length(p)
        if ( p(i)>0) rectangle('pos',[i-0.5,p(i)-0.5,1,1],'edgecolor','red'); end;
    end
    title( sequences{n} );
end
