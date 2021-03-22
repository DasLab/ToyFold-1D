# Benchmarks

[das_toyfold1d_10k_14mers.csv.gz](benchmarks/das_toyfold1d_10k_14mers.csv.gz)

Contains 10,000 examples of folding 14-mers in the ToyFold 1D model.

* Contains information on minimum free energy (MFE) structure for each sequence, as well as base pair 
     probability matrix computed for entire Boltzmann ensemble via enumeration.
* Data are 
  `sequence` (string of A,C,G,U); 
  `x_1`,`x_2`,... (1-indexed position of beads in MFE structure); 
  `p_1`,`p_2`,... (pairing partner of each bead, 1-indexed; 0 means no partner); 
  `bpp_1_1`, `bpp_1_2`,... (base pair probability of index i & j, averaged over Boltzmann ensemble)
* Generated with parameters: `params.epsilon = -2; params.delta = 5`. 
* The bending penalty of +5 and the base pair bonus of -2 mean that 3 base pairs are needed to stabilize a stem
* A-U and G-C pairs are worth the same
* Due to the length restriction of 14-mers, its not possible to get a pseudoknot as the MFE structure
   (Pseudoknots do contribute to the partition function and can appear in the base pair probability matrix.)


