function num_bends = score_bends( x, is_chainbreak );
%  num_bends = score_bends( x, is_chainbreak );
%
% Take a bunch of conformations and score the number of bends in each
% conformation. For use in determining bending energy penalty ('Delta' in
% toyfold notes).
%
% INPUT
%
% x = [Nbeads x Nconformations] Positions
% is_chainbreak = [Nbeads x 1] is the bead at the end of a chain?
%
% OUTPUT
%
% num_bends = [1 x num_conformations] number of bends
%
% (C) R. Das, Stanford University, 2020
if ~exist('is_chainbreak','var') is_chainbreak = zeros(size(x,1),1); is_chainbreak(end) = 1; end;
Nconf = size(x,2);
d = get_directions_from_positions(x);
num_bends = sum( (d(1:end-1,:) ~= d(2:end,:)) .* repmat(~is_chainbreak(2:end),1,Nconf) );


