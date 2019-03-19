
#include <assert.h>
#include "jar_type.h"

#ifndef JAR_UTILS

#define JAR_UTILS

void show_UniJAR( char description[], UniJAR x );
void show_val_LogPSLm( char description[], UniJAR x );
void show_val_LinFP32( char description[], UniJAR x );

UniJAR two_2_k( int k );
UniJAR rnd_2_L_frac( UniJAR x, int L );
UniJAR rnd_2_PSLm( UniJAR x );
float  LogPSLm_2_Lin_val( UniJAR x );

#endif



