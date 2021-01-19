function [Z,conf_prob] = get_Z(x,p,is_chainbreak,params);
% [Z,conf_prob] = get_Z(x,p,is_chainbreak,params);
%
% Partition function calculation.
%
% INPUT
%  x = [Nbeads x Nconformations] all sets of conformations.
%        If there are no base pairs specified, should get
%        2^(Nbeads-1). First position is always 0.   
%  p = [Nbeads x Nconformations] partners  (0 if bead is unpaired,
%        otherwise index of partner from 1,... Nbeads )
%  is_chainbreak = [Nbeads x 1] is the bead at the end of a chain?
%  params = Energy parameter values for delta, epsilon, etc. [MATLAB struct]
%
% OUTPUT
% secstruct = Secondary structure in dot-parens notation, e.g.
%                 '((.))'. Give [] or '' if you want
%                  the secondary structures to be enumerated
% sequence  = [optional] Input sequence (array of 0's and 1's) with
%                 'colors'. Required if you want secstruct's to be
%                 enumerated.
%
% 
% (C) R. Das, Stanford University, 2020

if ~exist( 'params','var') params = get_default_energy_parameters(); end;

E = get_energy(x,p,is_chainbreak,params);
Z = sum( exp( -E ) );
conf_prob = exp(-E)/Z;

