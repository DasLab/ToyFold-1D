function [is_chainbreak, secstruct_new ] = parse_out_chainbreak( secstruct );
secstruct_new = '';
is_chainbreak = [];
for i = 1:length( secstruct )
    if any( strcmp({',','+',' ','&'},secstruct(i) ) )
        if length( is_chainbreak ) > 0; is_chainbreak(end) = 1; end;
    else
        secstruct_new = [ secstruct_new, secstruct(i) ];
        is_chainbreak = [ is_chainbreak, 0 ];
    end
end
is_chainbreak = is_chainbreak';