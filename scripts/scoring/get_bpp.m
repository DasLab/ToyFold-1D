function  bpp = get_bpp(x,p,is_chainbreak,params);
% bpp = get_bpp(x,d,p,epsilon,delta);
%
% Get base pair probability matrix from get_conformations() output of
%   ensemble of conformations (positions,directions,pairings).
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
% Output
% bpp = [Nbeads x Nbeads] matrix of probabilities (from 0 to 1) that  
%                bead i is paired to bead j in the ensemble.
%
% 
% (C) R. Das, Stanford University
if ~exist( 'params','var') params = get_default_energy_parameters(); end;

[Z,conf_prob] = get_Z(x,p,is_chainbreak,params);
N = size(x,1);
bpp = zeros(N,N);
for q = 1:size(x,2)
    partner = p(:,q);
    for m = 1:length(partner)
        if partner(m) > 0;
            bpp(m,partner(m)) = bpp(m,partner(m)) + conf_prob(q);
        end
    end
end
