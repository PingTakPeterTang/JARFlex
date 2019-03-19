
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "jar_type.h"
#include "jar_utils.h"

#define DEBUG_utils 0

void show_UniJAR( char description[], UniJAR x ) {
   printf("%s Hex representation of val.I = %08X \n", description, x.I);
   return;
}

void show_val_LogPSLm( char description[], UniJAR x ) {
/* 
A LogPSLm number is of the form ((-1)^s, m+f) but encoded
here as IEEE FP32 representation of the value (-1)^s * 2^m * (1+f).
This function shows the numeric values of (-1)^s and  m+f
*/
   int    m;
   UniJAR sign, f;

   sign.I = (x.I & SIGN_MASK) | 0X3F800000;
   f.I    = (x.I & FRAC_MASK) | 0x3F800000;
   f.F   -= 1.0;
   
   m      = ((x.I & BEXP_MASK) >> 23) - 127;
   printf("%s  sign is %f, m is %d, f is %f \n", description, sign.F, m, f.F);
}

void show_val_LinFP32( char description[], UniJAR x ) {
/* 
A LinFP32 number is a linear domain number encoded in IEEE FP32.
So in fact the value is the numerical val of the bit string of x
interpreted as an IEEE encoding. This function is here just for
the sake of symmetry. To add a little bit of value, we show the
bit string in hex.
*/

   printf("%s  linear-space value is %10.6e, IEEE Hex is %08X \n", description, x.F, x.I);
}

UniJAR two_2_k( int k ) {
   UniJAR x;
   assert( k <= 50 & k >= -50 );
   if (k >= 0) {
      x.F = 1.0; x.I += (k << 23); 
   }
   else {
      x.F = 1.0; x.I -= ((-k) << 23);
   }
 
   return x;
}


UniJAR rnd_2_L_frac( UniJAR x, int L ) {
   UniJAR Big, y;
   
   assert (L >= 0 & L <= 10);
   y = x; y.I &= CLEAR_FRAC;
   Big = two_2_k( 23-L );

   if (DEBUG_utils) {
      show_UniJAR("..input ---> ",x);   printf("\n"); 
      show_UniJAR("..BIG   ---> ",Big); printf("\n"); 
   }

   Big.F *= y.F;
   x.F += Big.F;
   x.F -= Big.F;

   if (DEBUG_utils) {
      show_UniJAR("..after rnd  ",x); printf("\n");
   }
   return x;
}

UniJAR rnd_2_PSLm( UniJAR x ) {
   UniJAR Big, sign_x, y;
   UniJAR opMask;
   int    ind;

   sign_x.I = x.I & SIGN_MASK;
   ind = (x.I & BEXP_MASK) >> 23;


   /* (Big + (x & opMask)) - (Big & opMask) */
   opMask = Mask_tbl[ind];
   Big = Big_tbl[ ind ];
   
   if (DEBUG_utils) {
      printf("....inside rnd_2_PSLm \n"); 
      show_UniJAR("      input  ",x);
      show_UniJAR("      Big    ",Big);
   }
   
   y.I = x.I & opMask.I;
   y.F += Big.F;
   if (DEBUG_utils) show_UniJAR("    x+Big    ",y); 
   Big.I &= opMask.I;
   y.F -= Big.F;
   if (DEBUG_utils) show_UniJAR("  undo Big   ",y); 
   y.I |= sign_x.I;
   return y;
}

float LogPSLm_2_Lin_val( UniJAR x ) {
/*
Compute the accurate exp2 value of a LogPSLm input number
*/
   UniJAR y;
   float a, b;

   /* input logarithmic value is ( (-1)^s, m + f ) */
   y.I = (x.I & FRAC_MASK) | 0X3F800000;    
   /* y.F is now 1 + f */
   a = y.F - 1.0; y.F = exp2(a); 
   x.I &= CLEAR_FRAC; y.I &= FRAC_MASK;
   x.I |= y.I;
   b = x.F;
   return b;
}


#include "Big_tbl.h"
#include "Mask_tbl.h"

