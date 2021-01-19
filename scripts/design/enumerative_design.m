function [sequences,p_target,x,d,p] = enumerative_design( secstruct, pattern, params );
% [sequences,p_target,x,d,p] = enumerative_design( secstruct [, pattern, params] );
%
%   Emnumerate over all sequences that could form target secondary
%    structure (and conform to optional design pattern) and see which ones
%    fold best.
%
% Inputs
% secstruct = Target secondary structure in dot-parens notation, e.g.
%                 '((.))'. Give [] or '' if you want
%                  the secondary structures to be enumerated
% pattern = [Optional] sequence like AANUU, where characters like 
%                N,R,Y,W, and S are wild cards, and A,U,C, and G are preserved.
%                If not specified, code will use ANNNNNN... (note that
%                first base can be set to A without loss of generality in
%                current energy model)
%  params = Energy parameter values for delta, epsilon, etc. [MATLAB struct]
%
%
% Outputs
% sequences = all sequences that match secstruct and pattern, ordered
%              with 'best' design (by p_target) first
% p_target  = For each sequence, fraction of conformations with target 
%                secondary structure.
% x = For each sequence, all sets of conformations.
%        If there are no base pairs specified, should get
%        2^(Nbeads-1). First position is always 0.   
% d = For each sequence, input directions (array of +/-1's)
% p = For each sequence, partners  (0 if bead is unpaired,
%        otherwise index of partner from 1,... Nbeads )
%
%
% (C) R. Das, Stanford University, 2020
N = length( secstruct );
if ~exist( 'pattern', 'var' ) pattern = ['A',repmat( 'N', 1, N-1 )]; end;
if ~exist( 'params','var') params = get_default_energy_parameters(); end;

assert( length( pattern ) == length( secstruct ) );

sequences = get_sequences_for_pattern( pattern );
sequences = filter_by_secstruct( sequences, secstruct );
%[p_target,x,d,p] = test_designs( sequences, secstruct );
fprintf( 'Will test %d sequences...\n', length(sequences) );

%% are some sequences better designs than others?
x = {}; d = {}; p = {};
x_target = {}; d_target = {}; p_target = [];
Z = []; Z_target = [];
for q = 1:length( sequences )
    sequence = sequences{q};
    [p_target(q),x{q},p{q},is_chainbreak] = test_design(sequence,secstruct,params);
end
[p_sort,idx] = sort( -p_target );
p_target = p_target( idx );
x = x( idx );
p = p( idx );
sequences = sequences(idx);

%% calculate base pair probability maps to visualize alternatives.
colormap( 1 - gray(100));
clf
subplot(1,2,1);
set(gcf,'pos',[94   618   444   217]);
q = 1;
imagesc( get_bpp(x{q},p{q},is_chainbreak,params),[0 1] );
title( ['Best design\newline',sequences{q},'\newline',num2str(p_target(q))] );

subplot(1,2,2);
q = length(x);
imagesc(get_bpp(x{end},p{end},is_chainbreak,params),[0 1] );
title( ['Worst design\newline',sequences{end},'\newline',num2str(p_target(q))] );



