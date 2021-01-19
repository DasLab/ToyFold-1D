%% get_nts
assert( strcmp(get_nts_for_symbol( 'A' ),'A' ) );
assert( strcmp(get_nts_for_symbol( 'N' ),'ACGU' ) ); 
assert( strcmp(get_nts_for_symbol( 'W' ),'AU' ) );

sequences = get_sequences_for_pattern( 'AANUU' );
assert( all( strcmp(sequences,{'AAAUU','AACUU','AAGUU','AAUUU'} ) ) );

%% enumerative_design
% design by enumeration! Small hairpin
secstruct = '((.))';
[sequences,p_target,x,d,p] = enumerative_design( secstruct );
assert( strcmp(sequences{1},'AUCAU') );
assert( p_target(1) > 0.58 );
assert( p_target(end) < p_target(1) );
[x,p] = analyze_sequence(sequences{1});
assert( all( p(:,1) == secstruct_to_partner(secstruct) ) );

% enumerative_design force A/U only
[sequencesW,p_targetW,x,d,p] = enumerative_design( '((.))','AWWWW' );
assert( strcmp(sequencesW{1},'AUAAU') );
assert( p_targetW(1) > 0.1 );
assert( p_targetW(1) < p_target(1) );
% worst sequence
assert( p_targetW(end) == p_target(end) );
[x,p] = analyze_sequence(sequences{1});
assert( all( p(:,1) == secstruct_to_partner(secstruct) ) );

%% check pseudoknot design
secstruct = '(..[)]';
[sequences,p_target,x,d,p] = enumerative_design( secstruct );
assert( strcmp(sequences{1},'ACCUUA') );
assert( p_target(1) > 0.2 );
[x,p] = analyze_sequence(sequences{1});
assert( all( p(:,1) == secstruct_to_partner(secstruct) ) );

