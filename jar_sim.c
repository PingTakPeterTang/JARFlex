
#include <stdio.h>
#include <math.h>
#include "jar_sim.h"

#define  DEBUG_sim  0

UniJAR LinFP32_2_LogPSLm( UniJAR x ) {
/*
Input is in the linear (real) domain and encoded in IEEE FP32. 
The input usually corresponds to either 
(1) parameters already trained in a normal way and we wish to perform 
inference in JAR, or
(2) the linear-domain value in an FP32 accumulator. In our JAR emulation,
this value is encoded in IEEE FP32. This can also be an FP32 value rounded
from an FP64 accumulator.
In either case, the value is of the form (-1)^s * 2^m * (1 + f).
The logarithmic value is  ( (-1)^s,  m + log2(1+f_rnd)_rnd ). In Johnson's
paper, a log-domain value of ((-1)^s, m + g) -- m integer, g fraction -- is encoded in 
Posit: the information (-1)^s is encoded in Posit's sign bit, and  m is treated as an exponent 
and thus encoded using the combination of regime/expoent; g is encoded in the fractional part. 
Thus the number of bits for g varys depending on the value of m. 
In our simulation, we encode this value using IEEE FP32: (-1)^s is encoded in the sign bit,
the value m is biased by 127 and the resulting 8-bit (non-negative) value is encoded in the
exponent field, and g is simply put into the explicit mantissa field.

The conversion LinFP32_2_LogPSLm works as follows:
(1) The input value is first rounded to "beta" fractional bits corresponds to 
the log2 lookup table size of 2^beta. This is the "beta" in 
"(L, m, alpha, beta, gamma) log" using Johnson's
notation. So the input value is rounded to (-1)^s * 2^m * (1 + f_rnd). (Note that 
occasionally this "m" can be different from the "m" of the original input by 1 due to
the rounding.)
(2) The "beta" fractional bits in f_rnd is used to fetch from a table log2(1+f_rnd), which
is of the form 0.g where g has "gamma" bits -- the "gamma" in "(L, m, alpha, beta, gamma) log". 
In our emulation, we simply replace the explicit mantissa bits of the input number with "g"
(3) Finally, the "g" fraction of ((-1)^s, m + g) is rounded according to Posit80 based
on the value of m. This is accomplished by the utility function rnd_2_LogPSLm.

In Johnson's software implementation, a special tag IsZero is raised if the linear domain
value is 0. In our emulation, for the sake of speed, we encode zero as 2^(-63) or 0x20000000
if FP32. Any product involving this number will likely be rounded off completely with any
non-zero within range sum. (In Posit(8,0) and Posit(8,1) round off completely is guaranteed. 
*/

   UniJAR y, z;
   int    i, j, k;
   
   if (DEBUG_sim) printf("....inside LinFP32_2_LogPSLm \n");
   
   /* round the fraction to the number of bits used to fetch log2(1+f) */
   y    = rnd_2_L_frac( x, LOG2_IND_BITS );
   if (DEBUG_sim) show_UniJAR("     input LinFP32 rounded to LOG2_IND_BITS ", y);

   /* get index and fetch from table */
   i = (y.I & FRAC_MASK) >> LOG2_IND_SHIFT; 
   z    = log2_tbl[i];
   if (DEBUG_sim) show_UniJAR("     fraction bits of log2(1+f)             ", z);
   
   /* replace fraction of y by the log2(1+f) */
   y.I &= CLEAR_FRAC; y.I |= z.I;
   if (DEBUG_sim) show_UniJAR("     fraction replaced by log2(1+f)         ", y);



   /* Must round y to a number of fractional bits under Posit (L,m) */
   z = rnd_2_PSLm( y );
   if (DEBUG_sim) show_UniJAR("     rounded to PSLm precision              ", z);
   if (DEBUG_sim) printf("----exiting LinFP32_2_LogPSLm \n");

   return z;

}

UniJAR LogPSLm_2_LinFP32( UniJAR x ) {
/*
Input is in the logarithmic domain with range and precision 
that accmmodates the sum of two LogPSLm values. Note however
that the fractional bits never exceed L-3.
The input is usually the sum of two LogPSLm values; the sum
thus corresponds to the product of two linear domain values.
The logarithmic domain value is of the form ((-1)^s, m + f )
where the sign (-1)^s (s is 0 or 1) is the sign of the linear 
domain value, m is an integer (can be positive or negative) and
f is a fraction, 0 <= f < 1.

The linear domain value is therefore (-1)^s * 2^m * 2^(1+f) 
and in FP32 encoding, s is the sign bit, the biased exponent
is 127+m, represented in 8 bits, and the explicit mantissa bits
g is such that 2^(1+f) = 1 + g.

In our simulation the logarithm value ((-1)^s, m+f) is 
encoded as the IEEE FP32 representation of the value 
(-1)^s * 2^m * (1+f). Hence to get the LinFP32 encoding,
one merely needs to replace the explicit mantissa bits f
with g, which is obtained by table lookup indexed by f.

*/

   UniJAR y, z;
   int    i;
   
   if (DEBUG_sim) printf("....inside LogPSLm_2_LinFP32 \n");
   
   /* get index and fetch from table */
   i = (x.I & FRAC_MASK) >> EXP2_IND_SHIFT; 
   z    = exp2_tbl[i];
   if (DEBUG_sim) show_UniJAR("     fraction bits of exp2(1+f)             ", z);
   
   /* replace fraction of x by the exp2(1+f) */
   y.I = x.I & CLEAR_FRAC; y.I |= z.I;
   if (DEBUG_sim) show_UniJAR("     fraction replaced by exp2(1+f)         ", y);

   return y;
}

UniJAR sum2_LogPSLm( UniJAR x, UniJAR y ) {
/* 
This function sums two LogPSLm values exactly. The two
logarithmic value ((-1)^s_x , m_x + f_x ) and
((-1)^s_y, m_y + f_y). The sum is simply
(-1)^(s_x+s_y), m_z +  f_z where m_z and f_z are the integer 
and fractions of m_x+f_x + m_y+f_y.

Since the encoding we use for the logarithm value ((-1)^s, m+f) here is the 
IEEE FP32 representation of the value (-1)^s * 2^m * (1+f),
the sum can be computed as follows

1) take the encodings of x and y and perform a simple unsigned integer
addition.
2) At this point, the bottom 23 bits corresponds to the fraction of
f_z and the 8-bit exponent field is 127+127+m_z mod 128,
which is m_z - 2 mod 128. Thus, adding 127+1 to this will yield 127+m_z
3) Replace the msb (sign bit) with s_x + s_y mod 2 (which is s_x XOR s_y
if using logical operation)
*/

   UniJAR sign_z, z;

   sign_z.I = (x.I & SIGN_MASK) + (y.I & SIGN_MASK);
   z.I = x.I + y.I + 0X40800000; /* adding 128+1 to exponent field of int sum */
   z.I = (z.I & CLEAR_SIGN) | sign_z.I;

   return z;
}

void jar_fma( const UniJAR* a, const UniJAR* b, UniJAR* c ) {
  UniJAR w;
  
  w = LogPSLm_2_LinFP32( sum2_LogPSLm( *a, *b ) );
  
  (*c).F += w.F;
}

double dbl_fma( const UniJAR* a, const UniJAR* b, double* c ) {
  UniJAR w;
  
  w = LogPSLm_2_LinFP32( sum2_LogPSLm( *a, *b ) );
  
  *c += (double) w.F;
}


/*-----------------Commented out even there is a flag __AVX512F__ 
#if defined(__AVX512F__)
__m512i jar_fma_avx512( const __m512i a, const __m512i b, const __m512i c ) {
#if 0
  __m512i y;

  UniJAR tmpa[16];
  UniJAR tmpb[16];
  UniJAR tmpc[16];
  int i;

  _mm512_storeu_epi32( tmpa, a );
  _mm512_storeu_epi32( tmpb, b );
  _mm512_storeu_epi32( tmpc, c );

  for ( i = 0; i < 16; i++ ) {
    jar_fma( tmpa+i, tmpb+i, tmpc+i );
  }

  y = _mm512_loadu_epi32( tmpc );
#else
  __m512i sign_z;
  __m512i z;
  __m512i i;
  __m512i y;
  __m512i x;

  /-* comment: "sum2_LogPSLm function  *-/
  sign_z = _mm512_add_epi32( _mm512_and_epi32( a, _mm512_set1_epi32( SIGN_MASK ) ), _mm512_and_epi32( b, _mm512_set1_epi32( SIGN_MASK ) ) );
  z = _mm512_add_epi32( _mm512_add_epi32( a, b ), _mm512_set1_epi32( 0X40800000 ) );
  z = _mm512_or_epi32( _mm512_and_epi32( z, _mm512_set1_epi32( CLEAR_SIGN ) ), sign_z );

  /-* comment "LogPSLm_2_LinFP32" function *-/
  x = z;
  i = _mm512_srai_epi32( _mm512_and_epi32( x, _mm512_set1_epi32( FRAC_MASK ) ), EXP2_IND_SHIFT );
  z = _mm512_i32gather_epi32( i, exp2_tbl, 4 );
  y = _mm512_and_epi32( x, _mm512_set1_epi32( CLEAR_FRAC ) );
  y = _mm512_or_epi32( y, z );

  /-* comment: let's do fp32 add to c of y *-/
  y = _mm512_castps_si512( _mm512_add_ps( _mm512_castsi512_ps(c), _mm512_castsi512_ps(y) ) );
#endif
  return y; 
}
#endif
---------------------------------------end of commented out code-------*/



UniJAR jar_dotprod( const int n, const UniJAR* x, const UniJAR* y ) {
/* 
compute n-length dotprod in JAR. In particular, inputs x[], y[] and output are LogPSLm 
but accumulation of products are done in linear domain, using the type FP32 as accumulator.
The additions of LogPSLm quantities and also accumulation of LinFP32 numbers can be inexact 
when the summands have very different magnitudes. This occasionaly inexactness does not matter 
much as conversion between the two domains are not necessarily exact in the first place.
*/

#if 0
   UniJAR w;
#endif
   UniJAR z;
   int    i;

   assert (n >= 0);
   z.I = JAR_ZERO;
   for (i=0; i<n; i++) {
#if  1
     jar_fma( x+i, y+i, &z );
#else
     w = LogPSLm_2_LinFP32(
           sum2_LogPSLm( x[i], y[i] ) );
     z.F += w.F;
#endif
   }
   z = LinFP32_2_LogPSLm( z );
   return z;
}



UniJAR jar_dotprod_dbl( const int n, const UniJAR* x, const UniJAR* y ) {
/* 
compute n-length dotprod in JAR. In particular, inputs x[], y[] and output are LogPSLm 
but accumulation of products are done in linear domain, using the FP64 as accumulator.
The additions of LogPSLm quantities and also accumulation of LinFP32 numbers are in general
exact beacuse of FP64 ; but conversion between the two domains are not necessarily exact.
*/

#if 0
   double w;
#endif
   UniJAR result;
   double z;
   int    i;

   assert (n >= 0);
   result.I = JAR_ZERO;
   result = LogPSLm_2_LinFP32( result );
   z = (double) result.F;

   for (i=0; i<n; i++) {
#if  1
     dbl_fma( x+i, y+i, &z );
#else
     result = LogPSLm_2_LinFP32(
           sum2_LogPSLm( x[i], y[i] ) );
     z += (double) result.F;
#endif
   }
   result.F = (float) z;
   result = LinFP32_2_LogPSLm( result );
   return result;
}



void jar_matvecmul( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c ) {
/* 
compute matrix-vector product in JAR. In particular, inputs A[][], b[] and output c[] are LogPSLm 
but accumulation of products are done in linear domain. The additions of LogPSLm quantities
and also accumulation of LinFP32 numbers are exact; but conversion between the two domains
are not necessarily exact. Matrix A is in col-major format. 
*/
#if 0
  UniJAR w;
#endif
  int    m, k;

  assert (M >= 0);
  assert (K >= 0);

  /* let's set result to JAR_ZERO */
  for (m=0; m<M; ++m) {
    c[m].I = JAR_ZERO;
  }

  /* let's perform a matrix vector multiplication */
  for (k=0; k<K; ++k) {
    for ( m=0; m<M ; ++m ) {
      jar_fma( A+(k*M)+m, b+k, c+m );
    }
  }

  /* let convert to LogPSLm after accumulation */
  for (m=0; m<M; ++m) {
    c[m] = LinFP32_2_LogPSLm( c[m] );
  }
}

void jar_matvecmul_dbl( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c ) {
/* 
compute matrix-vector product in JAR. In particular, inputs A[][], b[] and output c[] are LogPSLm 
but accumulation of products are done in linear domain. The additions of LogPSLm quantities
and also accumulation of LinFP32 numbers are exact; but conversion between the two domains
are not necessarily exact. Matrix A is in col-major format. 
*/
#if 0
  UniJAR w;
#endif
  int    m, k;

  

  assert (M >= 0);
  assert (K >= 0);
  double c_dbl[M];

  /* let's set result to JAR_ZERO */
  for (m=0; m<M; ++m) {
    c_dbl[m] = 0.0;
  }

  /* let's perform a matrix vector multiplication */
  for (k=0; k<K; ++k) {
    for ( m=0; m<M ; ++m ) {
      dbl_fma( A+(k*M)+m, b+k, c_dbl+m );
    }
  }

  /* let convert to LogPSLm after accumulation */
  for (m=0; m<M; ++m) {
    c[m].F = (float) c_dbl[m];
    c[m] = LinFP32_2_LogPSLm( c[m] );
  }
}


/*-----------Commenting out this part; let Alex deal with this part 
void jar_matvecmul_avx512( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c ) {
/-* original comments
compute matrix-vector product in JAR. In particular, inputs A[][], b[] and output c[] are LogPSLm 
but accumulation of products are done in linear domain. The additions of LogPSLm quantities
and also accumulation of LinFP32 numbers are exact; but conversion between the two domains
are not necessarily exact. Matrix A is in col-major format. 
*-/ 
#if 0
  UniJAR w;
#endif
  int    m, k;

  assert (M >= 0);
  assert (K >= 0);

  /-* let's set result to JAR_ZERO *-/
  for (m=0; m<M; ++m) {
    c[m].I = JAR_ZERO;
  }

  /-* let's perform a matrix vector multiplication *-/
#if defined(__AVX512F__)
  for (m=0; m<(M/16)*16; m+=16) {
    __m512i vc = _mm512_loadu_epi32( c+m );
    for (k=0; k<K; ++k) {
      __m512i va = _mm512_loadu_epi32( A+(k*M)+m );
      __m512i vb = _mm512_set1_epi32( b[k].I );
      vc = jar_fma_avx512( va, vb, vc );
    }
    _mm512_storeu_epi32( c+m, vc );
  }
  for (   ; m<M        ; ++m ) {
    for (k=0; k<K; ++k) {
      jar_fma( A+(k*M)+m, b+k, c+m );
    }
  }
#else
  for (k=0; k<K; ++k) {
    for ( m=0; m<M ; ++m ) {
      jar_fma( A+(k*M)+m, b+k, c+m );
    }
  }
#endif

  /-* let convert to LogPSLm after accumulation *-/
  for (m=0; m<M; ++m) {
    c[m] = LinFP32_2_LogPSLm( c[m] );
  }
}
------------------------------- end of commenting out  */


void jar_matmul( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C ) {
/* 
compute matrix-vector product in JAR. In particular, inputs A[][], B[][] and output C[][] are LogPSLm 
but accumulation of products are done in linear domain. The additions of LogPSLm quantities
and also accumulation of LinFP32 numbers are exact; but conversion between the two domains
are not necessarily exact. All matrices are in col-major format. 
*/
#if 0
  UniJAR w;
#endif
  int    m, n, k;

  assert (M >= 0);
  assert (N >= 0);
  assert (K >= 0);

  /* let's set result to JAR_ZERO */
  for (m=0; m<M*N; ++m) {
    C[m].I = JAR_ZERO;
  }

  /* let's perform a matrix matrix multiplication */ 
  for (k=0; k<K; ++k) {
    for (n=0; n<N; ++n) {
      for (m=0; m<M; ++m) {
#if 1
        jar_fma( A+(k*M)+m, B+(n*K)+k, C+(n*M)+m );
#else
        w = LogPSLm_2_LinFP32(
                 sum2_LogPSLm( A[(k*M)+m], B[(n*K)+k] ) );
        C[(n*M)+m].F += w.F;
#endif
      }
    }
  }

  /* let convert to LogPSLm after accumulation */
  for (m=0; m<M*N; ++m) {
    C[m] = LinFP32_2_LogPSLm( C[m] );
  }
}



void jar_matmul_dbl( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C ) {
/* 
compute matrix-vector product in JAR. In particular, inputs A[][], B[][] and output C[][] are LogPSLm 
but accumulation of products are done in linear domain, using FP64. The additions of LogPSLm quantities
is exact, and accumulation of FP64 numbers is most likely exact; but conversion between the two domains
are not necessarily exact. All matrices are in col-major format. 
*/
#if 0
  UniJAR w;
#endif
  int    m, n, k;

  assert (M >= 0);
  assert (N >= 0);
  assert (K >= 0);
  double C_dbl[M*N];

  /* let's set result to JAR_ZERO */
  for (m=0; m<M*N; ++m) {
    C_dbl[m] = 0.0;
  }

  /* let's perform a matrix matrix multiplication */ 
  for (k=0; k<K; ++k) {
    for (n=0; n<N; ++n) {
      for (m=0; m<M; ++m) {
#if 1
        dbl_fma( A+(k*M)+m, B+(n*K)+k, C_dbl+(n*M)+m );
#else
        w = LogPSLm_2_LinFP32(
                 sum2_LogPSLm( A[(k*M)+m], B[(n*K)+k] ) );
        C_dbl[(n*M)+m] += (double) w.F;
#endif
      }
    }
  }

  /* let convert to LogPSLm after accumulation */
  for (m=0; m<M*N; ++m) {
    C[m].F = (float) C_dbl[m];
    C[m] = LinFP32_2_LogPSLm( C[m] );
  }
}


/*----------Commenting out this part and let Alex deal with it
void jar_matmul_avx512( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C ) {
/-* original comments 
compute matrix-vector product in JAR. In particular, inputs A[][], B[][] and output C[][] are LogPSLm 
but accumulation of products are done in linear domain. The additions of LogPSLm quantities
and also accumulation of LinFP32 numbers are exact; but conversion between the two domains
are not necessarily exact. All matrices are in col-major format. 
*-/
#if 0
  UniJAR w;
#endif
  int    m, n, k;

  assert (M >= 0);
  assert (M >= 0);
  assert (K >= 0);

  /-* let's set result to JAR_ZERO *-/
  for (m=0; m<M*N; ++m) {
    C[m].I = JAR_ZERO;
  }

#if defined(__AVX512F__)
  /-* let's perform a matrix matrix multiplication *-/
  for ( m=0; m<(M/16)*16; m+=16 ) {
    for ( n=0; n<(N/8)*8; n+=8 ) {
      __m512i vc0 = _mm512_loadu_epi32( C+((n+0)*M)+m );
      __m512i vc1 = _mm512_loadu_epi32( C+((n+1)*M)+m );
      __m512i vc2 = _mm512_loadu_epi32( C+((n+2)*M)+m );
      __m512i vc3 = _mm512_loadu_epi32( C+((n+3)*M)+m );
      __m512i vc4 = _mm512_loadu_epi32( C+((n+4)*M)+m );
      __m512i vc5 = _mm512_loadu_epi32( C+((n+5)*M)+m );
      __m512i vc6 = _mm512_loadu_epi32( C+((n+6)*M)+m );
      __m512i vc7 = _mm512_loadu_epi32( C+((n+7)*M)+m );
      for (k=0; k<K; ++k) {
        __m512i va  = _mm512_loadu_epi32( A+(k*M)+m );
        __m512i vb0 = _mm512_set1_epi32( B[((n+0)*K)+k].I );
        vc0 = jar_fma_avx512( va, vb0, vc0 );
        __m512i vb1 = _mm512_set1_epi32( B[((n+1)*K)+k].I );
        vc1 = jar_fma_avx512( va, vb1, vc1 );
        __m512i vb2 = _mm512_set1_epi32( B[((n+2)*K)+k].I );
        vc2 = jar_fma_avx512( va, vb2, vc2 );
        __m512i vb3 = _mm512_set1_epi32( B[((n+3)*K)+k].I );
        vc3 = jar_fma_avx512( va, vb3, vc3 );
        __m512i vb4 = _mm512_set1_epi32( B[((n+4)*K)+k].I );
        vc4 = jar_fma_avx512( va, vb4, vc4 );
        __m512i vb5 = _mm512_set1_epi32( B[((n+5)*K)+k].I );
        vc5 = jar_fma_avx512( va, vb5, vc5 );
        __m512i vb6 = _mm512_set1_epi32( B[((n+6)*K)+k].I );
        vc6 = jar_fma_avx512( va, vb6, vc6 );
        __m512i vb7 = _mm512_set1_epi32( B[((n+7)*K)+k].I );
        vc7 = jar_fma_avx512( va, vb7, vc7 );
      }
      _mm512_storeu_epi32( C+((n+0)*M)+m, vc0 );
      _mm512_storeu_epi32( C+((n+1)*M)+m, vc1 );
      _mm512_storeu_epi32( C+((n+2)*M)+m, vc2 );
      _mm512_storeu_epi32( C+((n+3)*M)+m, vc3 );
      _mm512_storeu_epi32( C+((n+4)*M)+m, vc4 );
      _mm512_storeu_epi32( C+((n+5)*M)+m, vc5 );
      _mm512_storeu_epi32( C+((n+6)*M)+m, vc6 );
      _mm512_storeu_epi32( C+((n+7)*M)+m, vc7 );
    }
    for (    ; n<N ; ++n ) {
      __m512i vc0 = _mm512_loadu_epi32( C+((n+0)*M)+m );
      for (k=0; k<K; ++k) {
        __m512i va  = _mm512_loadu_epi32( A+(k*M)+m );
        __m512i vb0 = _mm512_set1_epi32( B[((n+0)*K)+k].I );
        vc0 = jar_fma_avx512( va, vb0, vc0 );
      }
      _mm512_storeu_epi32( C+((n+0)*M)+m, vc0 );
    }
  }
  for (k=0; k<K; ++k) {
    for (n=0; n<N; ++n) {
      for ( ; m<M; ++m) {
        jar_fma( A+(k*M)+m, B+(n*K)+k, C+(n*M)+m );
      }
    }
  }
#else
  /-* let's perform a matrix matrix multiplication *-/ 
  for (k=0; k<K; ++k) {
    for (n=0; n<N; ++n) {
      for (m=0; m<M; ++m) {
#if 1
        jar_fma( A+(k*M)+m, B+(n*K)+k, C+(n*M)+m );
#else
        w = LogPSLm_2_LinFP32(
                 sum2_LogPSLm( A[(k*M)+m], B[(n*K)+k] ) );
        C[(n*M)+m].F += w.F;
#endif
      }
    }
  }
#endif

  /-* let convert to LogPSLm after accumulation *-/
  for (m=0; m<M*N; ++m) {
    C[m] = LinFP32_2_LogPSLm( C[m] );
  }
}
---------End of commenting out --------*/



#include "exp2_tbl.h"
#include "log2_tbl.h"

