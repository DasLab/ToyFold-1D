function d = get_directions_from_positions(x);
% d = get_directions_from_positions(x);
%
% Input
%  x = [Nbeads x Nconformations] input positions
%
% Output
%  d = [Nbeads x Nconformations] output directions (+/-1)
%
% (C) R. Das, Stanford University, 2021

d = 1+0*x;

d(1:end-1,:) = x(2:end,:)-x(1:end-1,:);

%assert( all( abs(d(:)) == 1 ) ); % needs to be relaxed if there are chainbreaks