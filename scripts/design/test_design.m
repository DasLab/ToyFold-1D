function [p_target,x,p,is_chainbreak] = test_design(sequence,secstruct,params);
% [p_target,x,d,p] = test_design(sequence,secstruct,params);
%
% Main script for testing if a sequence folds well into target 
%   secondary structure.
%
% Inputs
%  sequence  = sequence like 'AAACCCGGA'
%  secstruct = target secondary structure in dot parens notation
%  params = Energy parameter values for delta, epsilon, etc. [MATLAB struct]

% Output
%  p_target = Fraction of conformations with target 
%                secondary structure. (Higher is better.)
%  x = [Nbeads x Nconformations] all sets of conformations.
%        If there are no base pairs specified, should get
%        2^(Nbeads-1). First position is always 0.   
%  p = [Nbeads x Nconformations] partners  (0 if bead is unpaired,
%        otherwise index of partner from 1,... Nbeads )
% is_chainbreak = [Nbeads x 1] is the bead at the end of a chain?
%
% (C) Rhiju Das, Stanford University 2020

if ~exist( 'params','var') params = get_default_energy_parameters(); end;

[x,p,is_chainbreak] = get_conformations('',sequence);
Z = get_Z(x,p,is_chainbreak,params);

[x_target,p_target] = get_conformations(secstruct,sequence,params);
Z_target = get_Z(x_target,p_target,is_chainbreak,params);

p_target = Z_target/Z;
