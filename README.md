# JARFlex
Simulation of Jeff Johnson's Arithmetic (L,m,alpha,beta,gamma)Log 
To build, one modify (if necessary) the file configure_jar.h to specify the five parameter L, m, alpha, beta, gamma
One also specify an upper limit on the size of the exp2 table. In general Max_exp2_ind_bits set to 12 is sufficient
to ensure the entire fractional part of a Johnson logarithmic-domain number is used in looking up an exp2 value.
When one build demo using    make demo
a generator is built and executed, providing the jar_type.h file together with four tables.
The rest of the simulation is then built.
