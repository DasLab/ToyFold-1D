function [x,d,p,E] = get_conformations( secstruct, sequence, epsilon, delta );
% [x,d,p] = get_conformations( secstruct, sequence );
%
% Figure out all the conformations (bead positions)
%  that are consistent with a secondary structure.
%
% Without loss of generality, conformations are assumed to 
%  start out at origin (0) and initially point in positive direction.
%
% Conformations are returned so that first configurations have the most 
%  pairs and least bends. (sort is on num_bends - num_pairs );
% 
%
% INPUT
% secstruct = Secondary structure in dot-parens notation, e.g.
%                 '((.))'. Give [] or '' if you want
%                  the secondary structures to be enumerated
% sequence  = [optional] Input sequence (array of 0's and 1's) with
%                 'colors'. Required if you want secstruct's to be
%                 enumerated.
%  epsilon = [optional] energy bonus for each pair (use negative number for bonus), 
%               units of kT. [Default -2]
%  delta = [optional] energy penalty for each bend (use positive number for penalty), 

% OUTPUT
%  x = [Nbeads x Nconformations] all sets of conformations.
%        If there are no base pairs specified, should get
%        2^(Nbeads-1). First position is always 0.   
%  d = [Nbeads x Nconformations] input directions (array of +/-1's)
%  p = [Nbeads x Nconformations] partners  (0 if bead is unpaired,
%        otherwise index of partner from 1,... Nbeads )
%  E = [Nconformations] Energies for each conformation.
% 
% (C) R. Das, Stanford University, 2020

if ~exist( 'epsilon','var') epsilon = -2; end;
if ~exist( 'delta','var') delta = 1; end;

x = [];

% partner is array with same length as chain
%  that records the partner of each bead 
%  (set to 0 if bead has no partner).
[is_chainbreak, secstruct] = parse_out_chainbreak( secstruct );
partner = secstruct_to_partner( secstruct );
N = length( secstruct ); 

% sequence input (asking get_conformations to enumerate secondary
%   structures)
if exist( 'sequence', 'var' ) 
    [is_chainbreak_sequence, sequence ] = parse_out_chainbreak( sequence ); 
end 
if N == 0
    assert( logical(exist( 'sequence', 'var' ) ) );
    N = length( sequence );
    is_chainbreak = is_chainbreak_sequence;
    partner = zeros(1,N);
end

% without loss of generality,
%  place first bead at x = 0,and point in positive direction. 
if N > 0;
    x = [0];
    d = [1];
    p = partner(1);
end

stem_assignment = figure_out_stem_assignment( secstruct );

% How about the rest?
for i = 2:N
    if ~is_chainbreak(i-1)
        if ~isempty( stem_assignment ) && ...
                stem_assignment(i) > 0 && ...
                stem_assignment(i) == stem_assignment(i-1) ...
                % continuing a stem. go in the same direction!
            d(i,:) = d(i-1,:);
        else
            q = size(x,2);
            % two choices for next move -- forward or backward.
            % Forward:
            d(i,1:q) = 1;
            
            % Backward (note that these are 'new' histories).
            newblock = size(x,2) + [1:q];
            x(:, newblock) = x(:, 1:q);
            d(:, newblock) = d(:, 1:q);
            p(:, newblock) = p(:, 1:q);
            d(i, newblock) = -1;
        end
        x(i,:) = x(i-1,:)+d(i,:);
        p(i,:) = partner(i);
    else
        q = size(x,2);
        % new strand! can go anywhere! lots of new options to enumerate
        for xx = -(2*N-1):(2*N-1)
            for dd = [1 -1];
                newblock = size(x,2) + [1:q];
                x(:,newblock) = x(:,1:q);
                d(:,newblock) = d(:,1:q);
                p(:,newblock) = p(:,1:q);
                x(i,newblock) = xx;
                d(i,newblock) = dd;
                p(i,newblock) = partner(i);
            end
        end
    end
    
    if ~isempty( stem_assignment )
        % filter trajectories that obey user-inputted pairs -- positions are at same level,
        %   and going in opposite directions
        if partner(i) == 0; continue; end;
        if partner(i) > i;  continue; end;
        gp = find( x(i,:) == x(partner(i),:) & d(i,:) == -d(partner(i),:) );
        x = x(:,gp);
        d = d(:,gp);
        p = p(:,gp);
    else
        % enumerate secondary structures by figuring out possible partners
        %  of the new bead with any previously positioned beads
        % Just use a for loop now for coding simplicity -- vectorize later
        %  if we need the speed:
        xnew = [];
        dnew = [];
        pnew = [];
        assert( all(p(i,:) == 0 ) );
        for m = 1:size( x, 2 );
            xm = x(:,m);
            dm = d(:,m);
            pm = p(:,m);
            
            % no partner for bead i
            xnew = [xnew, xm];
            dnew = [dnew, dm];
            pnew = [pnew, pm];

            % look for partners based on matching position
            sequence_match =  (sequence(1:i) == 'N' | sequence(i) == 'N' | ...
                (sequence(1:i) == 'A' & sequence(i) == 'U') | ...
                (sequence(1:i) == 'U' & sequence(i) == 'A') | ...
                (sequence(1:i) == 'G' & sequence(i) == 'C') | ...
                (sequence(1:i) == 'C' & sequence(i) == 'G') | ...
                (isstrprop( sequence(1:i) ,'lower' ) & ...
                isstrprop( sequence(i), 'lower' ) & ...
                sequence(1:i) == sequence(i) ) );
            partners = find( xm == xm(i) & ~pm & dm ~= dm(i) & sequence_match' );
            for pp = partners'
                pm = p(:,m);
                pm(i)  = pp;
                pm(pp) = i;
                                
                xnew = [xnew, xm];
                dnew = [dnew, dm];
                pnew = [pnew, pm];
            end
        end
        x = xnew;
        d = dnew;
        p = pnew;
    end
end

% re-order trajectories, sorted by the number of bends
E =  get_energy(d,p,epsilon,delta);
[E,idx] = sort( E );

x = x(:,idx);
d = d(:,idx);
p = p(:,idx);

% 
% 
% function sequence_num = convert_sequence_to_numbers(sequence)
% % may be unnecessary
% sequence_num = [];
% for i = 1:length(sequence)
%     switch upper( sequence(i) )
%         case 'N'
%             sequence_num(i) = -1;
%         case 'A'
%             sequence_num(i) = 0;
%         case 'U'
%             sequence_num(i) = 1;
%     end
% end
% sequence_num = sequence_num;


