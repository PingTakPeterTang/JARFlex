#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "configure_jar.h"

#define MAX(x,y) ((x)>=(y)? (x):(y))
#define MIN(x,y) ((x)>=(y)? (y):(x))

#define DEBUG_generate 0

typedef union{
   unsigned int   I;
   float          F;
} UniJAR;


void write_line( FILE *myfile, char *myline );
void write_tbl( UniJAR *tbl, FILE *myfile, char *tbl_name, int tbl_len );

void gen_jar_type_top(FILE *myfile);

                    
int analyze_expo( int expo, int *expo_rnd,
                  int *regime_len, int *regime_val, 
                  int *frac_bits, int *magic_num );

UniJAR rnd_2_L_frac( UniJAR x, int L );

/*
 * The file configure_jar.h contains the JAR parameters
 * Posit_{L, m, alpha, beta, gamma}
 * Based on these parameters, this generator produces
 * 1) the file jar_type.h 
 * 2) the files {exp2_tbl, log2_tbl, Big_tbl, Mask_tbl}.h
 *    these files are not stand alone in the sense that they
 *    are included inside other non-generated programs
 */

UniJAR Big_tbl[256];
UniJAR Mask_tbl[256];
UniJAR exp2_tbl[4096];
UniJAR log2_tbl[4096];



void main(){

   FILE *myfile;
   char string[80];
   int  L, m, alpha, beta, gamma;
   int  expo, regime_len, regime_val, expo_rnd, frac_bits, magic_num;
   int  need_to_rnd, ind;
   int  exp2_ind_bits, exp2_tbl_len, log2_tbl_len;
   float  delta;
   UniJAR x;

   myfile = fopen("jar_type.h","w");
   gen_jar_type_top(myfile);
   L = Posit_L; m = Posit_m; 
   alpha = Posit_alpha; beta = Posit_beta; gamma = Posit_gamma;

   sprintf(string,"#define Posit_L      (%d)", L);
   write_line(myfile, string);
   sprintf(string,"#define Posit_m      (%d)", m);
   write_line(myfile, string);
   sprintf(string,"#define Posit_alpha  (%d)", alpha);
   write_line(myfile, string);
   sprintf(string,"#define Posit_beta   (%d)", beta);
   write_line(myfile, string);
   sprintf(string,"#define Posit_gamma  (%d)", alpha);
   write_line(myfile, string);
   write_line(myfile, "\n\n\n");

   exp2_ind_bits = MIN(L-3-m, Max_exp2_ind_bits);
   sprintf(string,"#define EXP2_IND_BITS    (%d)", exp2_ind_bits);   write_line(myfile, string);
   sprintf(string,"#define EXP2_IND_SHIFT   (23-EXP2_IND_BITS)");    write_line(myfile, string);
   sprintf(string,"#define LOG2_IND_BITS    (%d)", beta);            write_line(myfile, string);
   sprintf(string,"#define LOG2_IND_SHIFT   (23-LOG2_IND_BITS)");    write_line(myfile, string);
   sprintf(string,"#define EXP2_FRAC_BITS   (%d)", alpha);           write_line(myfile, string);
   sprintf(string,"#define LOG2_FRAC_BITS   (%d)", gamma);           write_line(myfile, string);

   sprintf(string,"\n\n");                                                  write_line(myfile, string);
   sprintf(string,"extern UniJAR Big_tbl[256];");                           write_line(myfile, string);
   sprintf(string,"extern UniJAR Mask_tbl[256];");                          write_line(myfile, string);
   sprintf(string,"extern UniJAR exp2_tbl[%d];",(int)pow(2,exp2_ind_bits)); write_line(myfile, string);
   sprintf(string,"extern UniJAR log2_tbl[%d];",(int)pow(2,beta));          write_line(myfile, string);
   sprintf(string,"\n\n\n");                                                write_line(myfile, string);

   /*
   sprintf(string,"#include \"Big_tbl.h\"");                         write_line(myfile, string);
   sprintf(string,"#include \"Mask_tbl.h\"");                        write_line(myfile, string);
   sprintf(string,"#include \"exp2_tbl.h\"");                        write_line(myfile, string);
   sprintf(string,"#include \"log2_tbl.h\"");                        write_line(myfile, string);
   */
                     

   sprintf(string,"\n\n#endif\n");                                  write_line(myfile, string);
   fclose(myfile);

   /* generate four table files */
   
   /* populate Big_tbl and Mask_tbl */
   for (ind = 0; ind <= 255; ind++)
   {
       expo = ind-127;
       need_to_rnd = analyze_expo( expo, &expo_rnd, 
                                   &regime_len, &regime_val, &frac_bits, &magic_num );
       if (need_to_rnd){
           Mask_tbl[ind].I = 0x7FFFFFFF;
       }
       else{
           Mask_tbl[ind].I = 0x00000000;
       }
       Big_tbl[ind].I = magic_num;
   }
   myfile = fopen("Big_tbl.h","w");
   write_tbl(Big_tbl, myfile, "Big_tbl", 256);
   fclose(myfile);

   myfile = fopen("Mask_tbl.h","w");
   write_tbl(Mask_tbl, myfile, "Mask_tbl", 256);
   fclose(myfile);

   /* populate exp2_tbl */
   exp2_tbl_len = pow(2,exp2_ind_bits); 
   delta = pow(2.0,-exp2_ind_bits);
   for (ind=0; ind < exp2_tbl_len; ind++)
   {
      x.F = exp2( (float) ind * delta );
      x = rnd_2_L_frac( x, alpha );
      x.I &=  0x007FFFFF;
      exp2_tbl[ind].I = x.I;
   }
   myfile = fopen("exp2_tbl.h","w");
   write_tbl(exp2_tbl, myfile, "exp2_tbl", exp2_tbl_len);
   fclose(myfile);

   /* populate log2_tbl */
   log2_tbl_len = pow(2, beta);
   delta = pow(2.0, -beta);
   for (ind = 0; ind < log2_tbl_len; ind++)
   {
      x.F = 1.0 + log2( 1.0 + (float)ind * delta );
      x   = rnd_2_L_frac( x, gamma );
      x.I &= 0X007FFFFF;
      log2_tbl[ind].I = x.I;
   }
   myfile = fopen("log2_tbl.h","w");
   write_tbl(log2_tbl, myfile, "log2_tbl", log2_tbl_len);
   fclose(myfile);

   return;
}


void write_line( FILE *myfile, char *myline ){
     fprintf(myfile, "%s\n", myline );
     return;
}

void gen_jar_type_top(FILE *myfile)
{
   char mystring[80];

   write_line(myfile,"/*");
   write_line(myfile," * This header file is generated ");
   write_line(myfile," * These are the parameters of (L,m,alpha,beta,gamma)Log of Johnson's arithmetic");
   sprintf(mystring, " * L = %d, m = %d, alpha = %d, beta = %d, gamma = %d",Posit_L, Posit_m, Posit_alpha, Posit_beta, Posit_gamma);
   write_line(myfile,mystring);
   write_line(myfile," */\n\n\n");

   write_line(myfile,"#ifndef JAR_TYPE\n\n");
   write_line(myfile,"#define JAR_TYPE");
   write_line(myfile,"typedef union{");
   write_line(myfile,"unsigned int   I;");
   write_line(myfile,"   float       F;");
   write_line(myfile,"} UniJAR;\n\n");

   write_line(myfile,"/*");
   write_line(myfile," * note that unsigned long takes precedence;");
   write_line(myfile," * initializing a table such as UniJAR my_tbl[8]"); 
   write_line(myfile," * with literals are interpreted as unit32."); 
   write_line(myfile," */");

   write_line(myfile,"#define JAR_ZERO   0x20000000");
   write_line(myfile,"#define SIGN_MASK  0x80000000");
   write_line(myfile,"#define FRAC_MASK  0x007FFFFF");
   write_line(myfile,"#define BEXP_MASK  0x7F800000");
   write_line(myfile,"#define CLEAR_SIGN 0x7FFFFFFF");
   write_line(myfile,"#define CLEAR_FRAC 0xFF800000");

   return;
}


int analyze_expo( int expo, int *expo_rnd,
                  int *regime_len, int *regime_val, 
                  int *frac_bits, int *magic_num )
{
   /* return regime_len, regime_val, frac_bits, magic_num */
   int L, m, k, ext_kmax, nrm_kmax, nrm_kmin, ext_kmin, ell, two_2_m;
   int ext_expo_min, nrm_expo_min, nrm_expo_max, ext_expo_max;
   int need_to_rnd, rnd, delta;

   L  = Posit_L; m = Posit_m;
   two_2_m = (int) pow(2, m);
   ext_kmin = -(L-2);
   nrm_kmin = -(L-2-m);
   nrm_kmax = L-3-m;
   ext_kmax = L-2;
   
   ext_expo_min = two_2_m*ext_kmin;
   nrm_expo_min = two_2_m*nrm_kmin;
   nrm_expo_max = two_2_m*nrm_kmax + (two_2_m - 1);
   ext_expo_max = two_2_m*ext_kmax;

   k = expo >> m;

   if (k < MAX(-63, 2*ext_kmin))
   {
      need_to_rnd = 0;
      *regime_val = ext_kmin;
      *regime_len = L-1;
      *frac_bits  = 0;
      *expo_rnd   = -63;
      *magic_num  = (127+(*expo_rnd)) << 23;
   }
   else if (k < ext_kmin)
   {
      need_to_rnd = 0;
      *regime_val = ext_kmin;
      *regime_len = L-1;
      *frac_bits  = 0;
      *expo_rnd   = ext_expo_min;
      *magic_num  = (127+ext_expo_min) << 23;
      
   }
   else if (k < nrm_kmin)
   {
      need_to_rnd = 0;
      *regime_val = k;
      *regime_len = 1-k;
      *frac_bits = 0;
      ell = (L-1)-(*regime_len);
      if (ell < m)
      {
         rnd = 1 << (m-ell-1);
         *expo_rnd = ((expo+rnd) >> (m-ell))<<(m-ell);
      }
      else
      {
         *expo_rnd = expo;
      }
      *magic_num = (127+(*expo_rnd)) << 23;
   }
   else if (k < 0)
   {
      need_to_rnd = 1;
      *regime_val = k;
      *regime_len = 1-k;
      *frac_bits  = L-1-m-(*regime_len);
      *expo_rnd   = expo;
      *magic_num  = (127+expo+(23-(*frac_bits))) << 23;

   }
   else if (k <= nrm_kmax)
   {
      need_to_rnd = 1;
      *regime_val = k;
      *regime_len = k+2;
      *frac_bits  = L-1-m-(*regime_len);
      *expo_rnd   = expo;
      *magic_num  = (127+expo+(23-(*frac_bits))) << 23;
   }
   else if (k <= ext_kmax)
   {
      need_to_rnd = 0;
      *regime_val = k;
      *regime_len = MIN(k+2, L-1);
      ell = (L-1)-*regime_len;
      *frac_bits  = 0;
      ell = (L-1)-(*regime_len);
      if (ell < m)
      {
         rnd = 1 << (m-ell-1);
         *expo_rnd = ((expo+rnd) >> (m-ell))<<(m-ell);
      }
      else
      {
         *expo_rnd = expo;
      }
      *magic_num = (127+(*expo_rnd)) << 23;
   }
   else
   {
      need_to_rnd = 0;
      *regime_val = ext_kmax;
      *regime_len = L-1;
      *frac_bits = 0;
      *expo_rnd  = ext_expo_max;
      *magic_num = (127+(*expo_rnd)) << 23;
   }

   return need_to_rnd;
}

void write_tbl( UniJAR *tbl, FILE *myfile, char *tbl_name, int tbl_len )
{
   int num_entries, num_entries_per_line, num_lines;
   int i, j, position, cnt;
   char mystring[120];

   num_entries = tbl_len;
   num_entries_per_line = 8;
   num_lines = num_entries / num_entries_per_line;

   sprintf(mystring, "UniJAR  %s[%d] = {", tbl_name, tbl_len); write_line(myfile, mystring);

   cnt = 0;
   for ( i=0; i<num_lines; i++ ){
       position = 0;
       for ( j=0; j<num_entries_per_line; j++ ){
           if (j < num_entries_per_line-1) {
               sprintf(&mystring[position], "0X%08X,",tbl[cnt].I);
               cnt++; position +=11;
           }
           else {
               if (i < num_lines-1) {
                   sprintf(&mystring[position], "0X%08X,",tbl[cnt].I);
                   cnt++;
               }
               else {
                   sprintf(&mystring[position], "0X%08X",tbl[cnt].I);
               }
           }
       }
       write_line(myfile, mystring);

   }
   write_line(myfile,"};");
   return;
}

UniJAR rnd_2_L_frac( UniJAR x, int L ) {
   UniJAR Big, y;
   
   assert (L >= 0 & L <= 12);
   y = x; y.I &= 0x7F800000;
   Big.F = pow(2.0, 23-L );

   Big.F *= y.F;
   x.F += Big.F;
   x.F -= Big.F;
   return x;
}
