[is_chainbreak, secstruct_new ] = parse_out_chainbreak('(( ))' );
assert( all( is_chainbreak == [0,1,0,0]' ) )
assert( strcmp(secstruct_new, '(())') );
