function num_bends = score_bends( d, is_chainbreak );
%  num_bends = score_bends( d, is_chainbreak );
%
% Take a bunch of conformations and score the number of bends in each
% conformation. For use in determining bending energy penalty ('Delta' in
% toyfold notes).
%
% INPUT
%
% d = [Nbeads x Nconformations] input directions
% is_chainbreak = [Nbeads x 1] is the bead at the end of a chain?
%
% OUTPUT
%
% num_bends = [1 x num_conformations] number of bends
%
% (C) R. Das, Stanford University, 2020
Nconf = size(d,2);
num_bends = sum( (d(1:end-1,:) ~= d(2:end,:)) .* repmat(~is_chainbreak(2:end),1,Nconf) );


