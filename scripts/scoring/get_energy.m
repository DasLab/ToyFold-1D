function E =  get_energy(x,p,is_chainbreak,params)
% E = get_energy(x,p,is_chainbreak,params)
% 
% Energy in toyfold model.
%
%
% Inputs
%  x = [Nbeads x Nconformations] input positions
%  p = [Nbeads x Nconformations] partners  (0 if bead is unpaired,
%        otherwise index of partner from 1,... Nbeads )
%  is_chainbreak = [Nbeads x 1] is the bead at the end of a chain?
%  params = Energy parameter values for delta, epsilon, etc. [MATLAB struct]
%
% Output
% E = [Nconformations] energies for each conformation 
%
% (C) R. Das, Stanford University 2020-21
%if ~exist( 'params','var') params = get_default_energy_parameters(); end;
assert( is_chainbreak(end) == 1 );

d = get_directions_from_positions(x); 

num_bends = score_bends_from_d(d, is_chainbreak);
num_pairs = score_pairs(p);
E = params.delta * num_bends + params.epsilon * num_pairs;

