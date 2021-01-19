function [x,p,is_chainbreak,E] = analyze_sequence( sequence, params, calc_m2 );
% analyze_sequence( sequence );
% analyze_sequence( sequence, params, calc_m2 );
% 
% Analyze and plot lowest energy conformations, base pair probability
%  matrix, and simulated M2seq data for sequence
%
% INPUT
%  sequence = sequence like 'AAACCCGGA'
%  params = Energy parameter values for delta, epsilon, etc. [MATLAB struct]
%  calc_m2 = do mutate-and-map simulation [default 1]
%
% (C) R. Das, Stanford University

if ~exist( 'params','var') params = get_default_energy_parameters(); end;
if ~exist( 'calc_m2','var') calc_m2 = 1; end;
N = length( sequence );

[x,p,is_chainbreak,E] = get_conformations('',sequence,params);

clf; 
subplot(1,2,1); 
draw_conformations(x,p,8,sequence,E);

subplot(1,2,2);
if calc_m2; subplot(2,2,2); end
bpp = get_bpp(x,p,is_chainbreak,params);
imagesc(bpp,[0 1]); axis image
hold on; plot([0.5:N+0.5],[0.5:N+0.5]); title( ['BPP for ',sequence] );
colormap( 1 - gray(100));

if calc_m2
    subplot(2,2,4);
    [M2seq, profile] = get_M2seq( sequence, params );
    imagesc(M2seq,[0 1]); axis image
    hold on; plot([0.5:N+0.5],[0.5:N+0.5]); title( ['M2-seq for ',sequence] );
end
