function c2_string = reverse_complement_TOYFOLD( c_string );
% c2_string = reverse_complement_TOYFOLD( c_string );
%
% INPUT:
%  c_string = string of nucleotides;
%
% OUTPUT:
%  c2_string = same string, reversed and with A->U, U->A, C->G, G->C, others to 'X';
%
% (C) R. Das, Stanford University, 2008, 2020

if iscell( c_string )
  c2_string = {};
  for k = 1: length( c_string )
    c2_string{k} = reverse_complement_string( c_string{k} );
  end
else
  c2_string = reverse_complement_string( c_string );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
function c2_string = reverse_complement_string( c_string );

for k = 1:length( c_string )
  c = c_string( length(c_string ) - k + 1);
  c2 = 'X';
  
  switch c
   case 'A'
    c2 = 'U';
   case 'C'
    c2 = 'G';
   case 'G'
    c2 = 'C';
   case 'U'
    c2= 'A';
   case 'T'
    c2 = 'A';    
   case ' '
    c2 = ' ';    
  end
  c2_string( k ) = c2;
end
return;
