%%
[x] = get_conformations('.....');
assert( size(x,1) == 5 );
assert( size(x,2) == 2^3);

[x,p] = get_conformations('((.))');
assert( size(x,1) == 5 );
assert( size(x,2) == 1);
num_bends = score_bends( x );
assert( all( num_bends == [1]) );
draw_conformations( x, p);

[x] = get_conformations('(...)');
assert( size(x,1) == 5 );
assert( size(x,2) == 3 );
num_bends = score_bends( x );
assert( all( num_bends == [1,2,3]) );

[x,p] = get_conformations( '(((...)))' );
num_bends = score_bends( x );
assert( all( num_bends == [1,3,3]) );
draw_conformations( '(((...)))' );

[x] = get_conformations('.((...))' );
assert( size(x,1) == 8 );
assert( size(x,2) == 6 );

%% 
% Chainbreaks
[x] = get_conformations('(( ))' );
assert( size(x,1) == 4 );
assert( size(x,2) == 1 );

[x] = get_conformations('.( )' );
assert( size(x,1) == 3 );
assert( size(x,2) == 1 );

[x] = get_conformations('(. )' );
assert( size(x,1) == 3 );
assert( size(x,2) == 1 );

[x] = get_conformations('( .)' );
assert( size(x,1) == 3 );
assert( size(x,2) == 2 );

%%
% test motif expansion
% 3-way junction
[x] = get_conformations( '(.( )..( )....)' );
assert( size(x,2)  == 126);

% secondary structure harboring 3 way junction and other motifs.
[x,d] = get_conformations( '..((.((..((...)))..)..(((...)))....))' );
%assert( size(x,2)  ==   4*126*12*3*3*3 );
assert( size(x,2)  == 2*2*126*4*3*3*3*3 );

%%
% enumerate conformations for a sequence
[x,p] = get_conformations( '','NNN' );
assert( size(x,1) == 3 );
assert( size(x,2) == 3 );
% look for hairpin as 'best' conformation
assert( all( x(:,1) == [0,1,0]') );
assert( all( p(:,1) == [3,0,1]') );

[x,p] = get_conformations( '','AAA' );
assert( size(x,1) == 3 );
assert( size(x,2) == 2 );
assert( all( p(:) == 0 ) );

[x,p] = get_conformations('','UAUUA');
assert( size(x,1) == 5 );
% look for hairpin as 'best' conformation
assert( all( x(:,1) == [0,1,2,1,0]') );
assert( all( p(:,1) == [5,4,0,2,1]') );
assert( size(x,2) == 23 );


% pseudoknot
[x,p] = get_conformations( '((..[)).]' );
assert( all( x == [0,1,2,1,0,1,0,-1,0]' ) );
assert( all( p(:,1) == [7,6,0,0,9,2,1,0,5]' ) );


