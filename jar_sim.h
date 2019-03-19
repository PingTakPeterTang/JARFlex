
/****************************************************************************************
 *  JAR is Johnson ARithmetic. There are two domains in question:
 *  Following Johnson's terminology, "Linear" is the usual real domain,
 *  and "Logarithmic", which is taking the based-2 logarithm of a real
 *  numbers. More specifically,
 *        (real domain) x |--> ( sign(x), log2(|x|) )
 *  For x == 0, a special IsZero flag is tagged along.
 *
 *  In a specific implementation, one needs to chose an encoding for the logarithmic
 *  values. Johnson's paper uses Posit(L,m) (e.g. Posit(8,1)) for the logarithmic values
 *  m + f, m integer and f fraction. The regime-and-exponent field of Posit are use
 *  to encode m, and the fractional part, f. A specific implementation requires the computation
 *  of exp2 and log2 functions, which are typically realized using table lookups. Thus, 
 *  a specific instance of Johnson's arithmetic is denoted by (L,m,alpha,beta,gamma)Log.
 *
 *  This emulation accommodates these parameters as follows.
 *  1) The parameters L, m, alpha, beta, gamma, and the longest allowed bit strings for exp2 lookup
 *     are given in configure_jar.h. Explanation for the longest allowed bit strings is this.
 *     The logarithmic values that are Posit representable number usually 
 *     have a short fractional part. Take Posit(8,0) for example. The logarithmic value is in the form
 *     m + f. The smallest m takes up 2 bits making the smallest lsb of 5 to be 2^(-5). Consequently,
 *     one only needs look up a table indexed by 5 bits (32 entries) to obtain all possible values of exp2(f).
 *     For Posit(L,m) of longer L, the number of index bits for exp2 lookup is increased. This generator uses
 *     a limit of the length based on what is configured in configure_jar.h.  
 *     Simply edit it to correspond to one's choice
 *  2) The generator.exe is run (generator.c is the source). This generator produces appropriate
 *     header files and table values.
 *  3) The rest of the emulation program can then be compiled and built.
 *
 *  Computing an inner product sum_j x_j*y_j for real numbers involves
 *    0) x_j and y_j are already represented in the logarithmic domain
 *    1) adding the logarithmic domain values of x_j and y_j
 *    2) immediately converting the logarithmic sum back to the real domain
 *       which involves taking the exp2 (that is 2**(z) function)
 *    3) summation of the resulting values is done inside an accumulator.
 *       In the ideal case, the accumulator has sufficient range and precision 
 *       so that accmulation incurs no rounding errors (note however that neither of the 
 *       conversions back and forth between real and logarithmic domains are exact). For
 *       small values of L and m, an FP32 accumulator sufficies to yield error free accumulation.
 *       For larger L,m configuration, one needs an FP64 accumulator to maintain this error free
 *       property. Note that error-free accumulation is of limited benefit as the linear <--> log
 *       space conversion are inexact. This emulation program provides two accumulation based on
 *       two "fma" functions c <-- a*b + c. They are 
 *           UniJAR jar_fma( UniJAR *a, UniJAR *b, UniJAR *c )   and
 *           double dbl_fma( UniJAR *a, UniJAR *b, double *c )
 *       The first one accumulates in FP32 (the accumulator is a UniJAR type)
 *       The second one accumulates into an FP64 variable. 
 *    4) At the end of all summations, the accumulated value, which is in 
 *       the real domain is converted back to logarithmic domain. In the case of accumulation 
 *       done in FP64, the FP64 is first converted back to FP32 because conversion from Linear space
 *       FP32 to Log Posit is performed.
 *
 *  JAR is emulated here in jar.h and jar.c. These are the key components
 *    1) The type JAR is the main container of the numerical quantities.
 *       UniJAR is a simple union type that overlays an uint32 with FP32.
 *       If Posit(8,0) is used for the the logarithmic domain, FP32 can encode the 
 *       corresponding numerical values without error. Moreover, accumulation of the 
 *       summands of the products x_j*y_j can be done without error in FP32. 
 *       Overlays of uint32 with FP32 facilitates mainpulation as in JAR, (1) one
 *       often needs to replace the fractional bits during conversion between
 *       the two domains, (2) table lookup based on certains bits of the encoding
 *       are often needed, and (3) our method here in adding two logarithmic values
 *       in the range of Posit (L,m) but encoded using FP32 requries some explicit
 *       manipulation of bit patterns. UniJAR facilitates all these quite well.
 *       The "Uni" in UniJAR can be thought of as "union" of "universal".
 *    2) There are two conversion functions 
 *           1) LinFP32_2_LogPSLm: Takes a normal IEEE FP32 number whose value
 *              is (-1)^s 2^m *(1+f) and converts to the form ( s, m + log2(1+f)_rnd )
 *              the number of significant bits in log2(1+f) is rounded according to
 *              the value of m (following Posit (L,m)). Finally, this value is
 *              encoded as the FP32 number (-1)^s * 2^m * (1 + log2(1+f)_rnd).
 *              LinFP32_2_LogPSLm is used to convert a normal FP32 neuralnet model
 *              to be used in JAR and (b) to convert the accumulator value back to the
 *              logarithmic domain. If the accumulator is FP64, the value is first converted
 *              to FP32 before LinFP32_2_LogPSLm is used.
 *           2) LogPSLm_2_FP32: The reverse of the LinFP32_2_LogPSLm. This is used 
 *              primarily after sum of two numbers in the logarithmic domain is computed. 
 *              A logarithmic number is of the form (s, m + f). The exp2 value is 
 *              (-1)^s * 2^m * (1 + (2^f - 1)). The fraction 2^f - 1 is obtained by 
 *              a table lookup from the most significant bits of f. FP32 is sufficient
 *              to hold the exponentiated value exactly (although the exponentiation itself
 *              has error). If these FP32 values need to be accumulated in a longer type such
 *              FP64, a cast after LogPSLm_2_FP32 sufficies.
 *    3) add_LogPSLm_2_LinFP32: Sums two LogPSLm numbers and immediately converts to LinFP32
 *    4) JAR_dotprod dot product computed using FP32 accumulator (using jar_fma)
 *    5) JAR_dotprod_dbl is a dot producted using FP64 accumulator and converted back to UniJAR 
 *       at the end.
 *
 ****************************************************************************************/

#ifndef JAR_SIM

#define JAR_SIM
#include "jar_type.h"
#include "jar_utils.h"

UniJAR LinFP32_2_LogPSLm( UniJAR x );
UniJAR LogPSLm_2_LinFP32( UniJAR x );
UniJAR sum2_LogPSLm( UniJAR x, UniJAR y );
UniJAR jar_dotprod( const int n, const UniJAR* x, const UniJAR* y );
UniJAR jar_dotprod_dbl( const int n, const UniJAR* x, const UniJAR* y ); 
void jar_matvecmul( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c );
void jar_matvecmul_dbl( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c );
void jar_matmul( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C );
void jar_matmul_dbl( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C );




/*-----------commented out for now as Peter will let Alex deal with this part
void jar_matvecmul_avx512( const int M, const int K, const UniJAR* A, const UniJAR* b, UniJAR* c );
void jar_matmul_avx512( const int M, const int N, const int K, const UniJAR* A, const UniJAR* B, UniJAR* C );
#if defined(__AVX512F__)
#include <immintrin.h>
inline __m512i jar_fma_avx512( const __m512i a, const __m512i b, const __m512i c );
#endif
 *-------------------------------------*/

#endif


