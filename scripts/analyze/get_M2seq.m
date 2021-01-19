function [M2seq, profile] = get_M2seq(sequence,params);
% [M2seq, profile] = get_M2seq(sequence,params);
%
% INPUT
%  sequence = sequence like 'AAACCCGGA'
%    params = Energy parameter values for delta, epsilon, etc. [MATLAB struct]
% (C) R. Das, Stanford University

if ~exist( 'params','var') params = get_default_energy_parameters(); end;

N = length( sequence );

[x,p,is_chainbreak] = get_conformations('',sequence);
bpp = get_bpp(x,p,is_chainbreak,params);
profile = 1 - sum(bpp);

M2seq = [];
for n = 1:N
    sequence_mut = sequence;
    sequence_mut(n) = reverse_complement_TOYFOLD( sequence(n) );
    [x,p,is_chainbreak] = get_conformations('',sequence_mut);
    bpp = get_bpp(x,p,is_chainbreak,params);
    M2seq = [M2seq; 1-sum(bpp)];
end
