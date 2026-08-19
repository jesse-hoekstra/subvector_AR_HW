/* mhg_core.c
 *
 * Standalone (no MATLAB / no Octave) version of Plamen Koev's mhg.c.
 *
 * The MEX gateway (mexFunction) has been replaced by a plain C entry point
 * mhg_eval(); the numerical core (the big while-loop) is copied verbatim from
 * the original mhg.c (mhg15). mxCalloc/mxMalloc/mxFree -> calloc/malloc/free,
 * mexErrMsgTxt -> error string + nonzero return.
 *
 * Original algorithm: Plamen Koev & Alan Edelman, "The Efficient Evaluation of
 * the Hypergeometric Function of a Matrix Argument", Math. Comp. 75 (2006),
 * 833-846. Copyright (c) Plamen Koev. See COPYRIGHT.TXT.
 *
 * Computes the truncated hypergeometric function  pFq^alpha(p;q;x[;y])  of one
 * or two matrix arguments (given by their eigenvalue vectors x and y).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Overflow-safe accumulation for the zonal coefficient `cc` in the h==n
 * branch.  For full-rank-p Omega with large kappa the running product
 * cc *= prodx[n]*prody[n] (~1e10 per step) overflows DBL_MAX by total
 * degree ~110-115, long before the series peak (~138), yielding inf*tiny
 * = nan even though the *final* series value is representable (~1e107).
 *
 * Fix: carry cc as a normalized (mantissa, log-scale) pair.  Whenever the
 * mantissa grows past CC_RENORM_HI it is scaled down by CC_RENORM and the
 * scale CC_RENORM_LOG (= log(CC_RENORM)) is moved into cc_log.  The branch
 * reconstructs the true contribution in log space and exp()s it back.
 * Where the original code sets cc = 1 we set (cc_mant, cc_log) = (1, 0),
 * so on every code path where the result was already finite the value is
 * bit-for-bit unchanged (cc_log stays 0, no exp/log is taken). */
#define CC_RENORM_HI   1e150
#define CC_RENORM      1e-150
#define CC_RENORM_LOG  (150.0 * 2.302585092994045684017991)  /* +150*ln(10) */

/* `z[h]` is the hypergeometric coefficient recurrence attached to the
 * current partition.  Unlike `cc`, it becomes extremely small: in the scalar
 * 0F1 case it contains the reciprocal Pochhammer/factorial terms.  Keeping it
 * as a bare double therefore used to underflow to zero around total degree
 * 102.  The traversal condition below interpreted that representational zero
 * as a mathematically terminating series, so increasing MAX produced a false
 * plateau (for example scalar 0F1(3.5;10000) was about 33% too small).
 *
 * Carry z as z_mantissa * exp(z_log).  Scaling the mantissa UP by 1e150
 * requires subtracting 150*ln(10) from the stored log scale.  As with the cc
 * fix, paths that never renormalize retain the original arithmetic exactly. */
#define Z_RENORM_LO      1e-150
#define Z_RENORM_UP      1e150
#define Z_RENORM_HI      1e150
#define Z_RENORM_DOWN    1e-150
#define Z_RENORM_LOG    (150.0 * 2.302585092994045684017991)

/* Multiply up to four ordinary factors and one logarithmic scale without
 * ever materializing a potentially under/overflowing scaled intermediate.
 * The final term is returned as a double; a non-finite final mathematical
 * term remains non-finite and is caught by the Python density guard. */
static double scaled_product(double scale_log,
                             double a, double b, double c, double d,
                             int nfactors)
{
   double factors[4] = {a,b,c,d};
   double lg = scale_log;
   double sign = 1.0;
   int i;

   for (i=0; i<nfactors; i++) {
      if (factors[i]==0.0) return 0.0;
      if (factors[i]<0.0) sign=-sign;
      lg += log(fabs(factors[i]));
   }
   return sign*exp(lg);
}

#define MHGERR(msg) do {                                          \
    if (err && errlen > 0) {                                      \
        size_t _l = strlen(msg);                                  \
        if (_l >= (size_t)errlen) _l = (size_t)errlen - 1;        \
        memcpy(err, msg, _l); err[_l] = 0;                        \
    }                                                             \
    return 1;                                                     \
} while (0)

/* arg0    : row vector (MAX, [K, [lambda_1, lambda_2, ...]])   length arg0len>=1
 * alpha   : the parameter alpha (1 = Schur, 2 = zonal, ...)
 * p,np    : numerator parameters a_1..a_p
 * q,nq    : denominator parameters b_1..b_q
 * x_in,n  : eigenvalues of the (first) matrix argument
 * y_in    : eigenvalues of the second matrix argument, or NULL for one argument
 * s_out   : (out) the value of the truncated series
 * coef_out: (out, optional) array of length MAX+1; coef[i] = sum over |kappa|=i.
 *           Pass NULL if not wanted.
 * err,errlen: error-message buffer.
 *
 * Returns 0 on success, nonzero on error.
 */
int mhg_eval(const double *arg0, int arg0len,
             double alpha,
             const double *p, int np,
             const double *q, int nq,
             const double *x_in, int n,
             const double *y_in,
             double *s_out, double *coef_out,
             char *err, int errlen)
{
   int    h,sl,nmu,i,j,k,*f, val, rankx,*lambda,ranky,lambdannz;
   double *x, *Sx, *xn, *prodx;
   double *y, *Sy, *yn, *prody;
   double c, *s, *z, *z_log, *coef, *kt, *mt, *blm, zn, dn,
          t, q1, q2, cc, cc_log, arg_log_scale, *lambdatemp;

   int lg, MAX, K, *D, heap, *l, w, *mu, *d, *g, *ww, *lmd,
        slm, nhstrip, gz;

   (void)val; (void)lambdatemp; /* declared in original, unused */

   if (n < 1)               MHGERR("x must have at least one element.");
   if (arg0len < 1)         MHGERR("First input must be (MAX,[K,[lambda]]).");

   s = s_out;

   /* mutable working copy of x (it is reordered in place below) */
   x = (double*)malloc(n*sizeof(double));
   memcpy(x, x_in, n*sizeof(double));

   /* compute rank x and reorder */
   rankx=0;
   for (i=0;i<n;i++) if (x[i]!=0) { x[rankx]=x[i]; rankx++; }
   for (i=rankx;i<n;i++) x[i]=0;

   /* retrieve y, compute its rank and reorder */
   if (y_in!=NULL) {
       y = (double*)malloc(n*sizeof(double));
       memcpy(y, y_in, n*sizeof(double));
       ranky=0;
       for (i=0;i<n;i++) if (y[i]!=0) { y[ranky]=y[i]; ranky++; }
       for (i=ranky;i<n;i++) y[i]=0;
   }
   else {
       y=NULL;
       ranky=n;
   }

   /* Jack polynomials are homogeneous in total degree.  Normalize each
    * matrix argument so its largest magnitude is at most ARG_TARGET, and bank
    * the removed scale in `arg_log_scale`: every degree-sl contribution is
    * multiplied back by exp(sl*arg_log_scale).  This keeps the Sx/Sy
    * recurrences finite after the z-underflow fix lets the series proceed to
    * its genuine tail. */
   arg_log_scale=0.0;
   {
       const double ARG_TARGET=4.0;
       double xmax=0.0, xscale=1.0;
       for (i=0;i<rankx;i++) if (fabs(x[i])>xmax) xmax=fabs(x[i]);
       if (xmax>ARG_TARGET) {
           xscale=xmax/ARG_TARGET;
           for (i=0;i<rankx;i++) x[i]/=xscale;
           arg_log_scale+=log(xscale);
       }
       if (y!=NULL) {
           double ymax=0.0, yscale=1.0;
           for (i=0;i<ranky;i++) if (fabs(y[i])>ymax) ymax=fabs(y[i]);
           if (ymax>ARG_TARGET) {
               yscale=ymax/ARG_TARGET;
               for (i=0;i<ranky;i++) y[i]/=yscale;
               arg_log_scale+=log(yscale);
           }
       }
   }

   MAX=(int) arg0[0];
   /* alpha is a parameter */

   /*------- retrieving lambda if it is available -------------------------*/
   /* in the second through last entries of arg0 */

   lambda=(int*)calloc(n+1,sizeof(int));
   lambda[0] = MAX+1;
   K=MAX;
   j=arg0len;
   if (j>n+2) j=n+2;  /* check if too many lambdas and discard */
   if (j>1) K=(int) arg0[1];
   if (j>2) for (i=1;i<=j-2;i++) lambda[i]=(int) arg0[i+1];
   else
       for(j=0;j<n;j++) {
           lambda[j+1] = MAX/(j+1); /* automatically does floor function */
           if (lambda[j+1]>K) lambda[j+1]=K;
       }

   /* counting the number of nonzeros for lambda and changing it to be the
    * same rank as min(rankx,ranky)*/
   if(rankx > ranky) {for (i=ranky+1;i<=n;i++) lambda[i]=0; }
   else {for (i=rankx+1;i<=n;i++) lambda[i]=0;}
   lambdannz=0;
   for (i=1;i<=n;i++) if (lambda[i]!=0) lambdannz++;

   /* make sure M,K,lambda are consistent */
   j=0;
   if ((MAX<K) || (K<lambda[1])) j=1;
   for (i=2;i<=n;i++) if (lambda[i-1]<lambda[i]) j=1;
   if (j==1) MHGERR("M,K,lambda inconsistent. Need M>=K>=lambda[1]>=lambda[2]>=...");

   /* p, q are parameters (np, nq their lengths) */

   /* if any of the p's equal beta/2, sum over partitions of only 1 part */
   j=0; /* none of the p's equal beta/2; note beta/2=1/alpha */
        /* this is a tricky check from a roundoff point of view */
   for (i=0;i<np;i++) if (1 == alpha*p[i]) j=1;
   if (j==1) {lambda[1]=MAX; for (i=2;i<=n;i++) lambda[i]=0;}

   if (coef_out!=NULL) coef=coef_out;
   else coef=(double*)malloc(sizeof(double)*(MAX+1));

   for (i=1;i<=MAX;i++) coef[i]=0;  /* set to zero, these are the coefficients of the polynomial */
   coef[0]=1; /* free term equals one */

   w = 0; /* index of the zero partition, currently l*/

   /* figure out the number of partitions |kappa|<= MAX with at most n parts */

   f=(int*) calloc (MAX+2,sizeof(int));
   for (i=1;i<=MAX+1;i++) f[i]=i;

   /* 9/23/14 change this loop to sum over the nonzeros of lambda */
   if (lambdannz == n)
     for (i=2;i<n;i++) for (j=i+1;j<=MAX+1;j++) f[j]+=f[j-i];
   else for(i=2;i<lambdannz+1;i++) for (j=i+1;j<=MAX+1;j++) f[j]+=f[j-i];

   w=f[MAX+1];

   free(f);

   D     = (int*) calloc(w+1,sizeof(int));
   Sx    = (double*) calloc(n*(w+1),sizeof(double));
   xn    = (double*) malloc(sizeof(double)*(n+1)*(MAX+2));
   prodx = (double*) malloc(sizeof(double)*(n+1));
   prodx[1]=x[0];
   for (i=2;i<=n;i++) prodx[i]=prodx[i-1]*x[i-1];
   for (i=1; i<=n; i++) {
     Sx[n+i-1]=1;
     xn[(MAX+2)*i+1]=1;
     for (j=2;j<=MAX+1;j++) xn[(MAX+2)*i+j]=xn[(MAX+2)*i+j-1]*x[i-1];
   }

   if (y!=NULL) {
      Sy    = (double*) calloc(n*(w+1),sizeof(double));
      yn    = (double*) malloc(sizeof(double)*(n+1)*(MAX+2));
      prody = (double*) malloc(sizeof(double)*(n+1));
      prody[1]=y[0];
      for (i=2;i<=n;i++) prody[i]=prody[i-1]*y[i-1];

      for (i=1; i<=n; i++) {
         Sy[n+i-1]=1;
         yn[(MAX+2)*i+1]=1;
         for (j=2;j<=MAX+1;j++) yn[(MAX+2)*i+j]=yn[(MAX+2)*i+j-1]*y[i-1];
      }
   }
   else { Sy=NULL; yn=NULL; prody=NULL; }

   l     = (int*)calloc(n+1,sizeof(int));


   l[0]=K;
   /* this is what limits l[1] by the second element of MAX if needed and
      allows for the check l[i]<l[i-1] to be OK even for i=1 */

   z     = (double*)malloc((n+1) * sizeof(double));
   z_log = (double*)calloc(n+1, sizeof(double));
   for (i=1;i<=n;i++) z[i]=1;

   mu    = (int*)calloc(n+1,sizeof(int));
   kt    = (double*)calloc(n+1,sizeof(double));
   for (i=1;i<=n;i++) kt[i]=-i;

   ww    = (int*)malloc((n+1) * sizeof(int));
   for (i=1;i<=n;i++) ww[i]=1;

   d     = (int*)calloc(n,sizeof(int));
   g     = (int*)calloc(n+1,sizeof(int));
   mt    = (double*)calloc(n+1,sizeof(double));
   blm   = (double*)calloc(n+1,sizeof(double));
   lmd   = (int*) calloc(n+1,sizeof(int));

   /*9/22/14 change heap = lambda[1] + 2 */
   heap  = lambda[1]+2;
   cc=1; cc_log=0;
   h=1;
   sl=1;  /* sl= sum(l) */


   /* 9/22/14 add l[h]<lambda[h] */
   while (h>0) {
       if ((l[h]<l[h-1]) && (MAX>=sl) && (z[h]!=0) && l[h]<lambda[h]) {

           l[h]++;

           /* 9/22/14 make sure summation only over partitions l<= lambda */
           for(i=1;i<=n;i++) if(l[i] > lambda[i])
           {MHGERR("Error,not summing over least amount of parts. Want l[i]<=lambda[i]");}

           if ((l[h]==1) && (h>1) && (h<n)) {
               D[ww[h]]=heap;
               ww[h]=heap;
               k=MAX-sl+l[h];
               if (lambda[h]<k) k = lambda[h];       /* 9/22/14 add lambda restriction */
               if (k>l[h-1]) k=l[h-1];
               heap+=k;

           }
           else ww[h]++;
           w=ww[h];

           /* Update Q */
           c=(1-h)/alpha+l[h]-1;
           zn=alpha;
           dn=kt[h]+h+1;
           for (j=0;j<np;j++)  zn*=p[j]+c;
           for (j=0;j<nq;j++)  dn*=q[j]+c;
           if (y!=NULL) {
               zn*=alpha*l[h];
               dn*=n+alpha*c;
               for (j=1;j<h;j++) {
                   t=kt[j]-kt[h];
                   zn*=t;
                   dn*=t-1;
               }
               zn/=dn;
               dn=1; /* trying to prevent overflow */
           }
           kt[h]+=alpha;
           for (j=1;j<h;j++) {
               t=kt[j]-kt[h];
               zn*=t;
               dn*=t+1;
           }
           z[h]*=zn/dn;
           if (z[h]!=0.0 && fabs(z[h])<Z_RENORM_LO) {
               z[h]*=Z_RENORM_UP;
               z_log[h]-=Z_RENORM_LOG;
           }
           else if (fabs(z[h])>Z_RENORM_HI) {
               z[h]*=Z_RENORM_DOWN;
               z_log[h]+=Z_RENORM_LOG;
           }

           /* Working hard only when l has less than n parts */


           if (h<n) {
               t=h+1-alpha; cc=1; cc_log=0; for (j=1; j<=h;j++) cc*=(t+kt[j])/(h+kt[j]);

               /* computing the index of l-ones(1,h) */
               nmu=l[1]; k=2; while ((k<=h)&&(l[k]>1)) nmu=D[nmu]+l[k++]-2;

               Sx[w*n+h-1]=cc*prodx[h]*Sx[nmu*n+h-1];

               if (y!=NULL) Sy[w*n+h-1]=cc*prody[h]*Sy[nmu*n+h-1];
               cc=1; cc_log=0; /* this way we can just update from 1 in the h=n case*/

               d[h-1]--; /* technically this has to execute only when h>1
                            but is OK if it is always executed; d[0] will
                            end up being -MAX at the end of the code */

               d[h]=l[h];  /* for (k=1;k<h;k++) d[k]=l[k]-l[k+1];
                              this happens automatically now via updates */

               lg=0; for (k=1;k<=h;k++) if (d[k]>0) {lg++; g[lg]=k;}
               slm=1; /* this is sum(l-mu) */
               nhstrip=1; for (k=1;k<=lg;k++) nhstrip*=d[g[k]]+1; nhstrip--;

               memcpy(&mu[1],&l[1],sizeof(int)*h);
               memcpy(&mt[1],&kt[1],sizeof(double)*h);
               for (k=1;k<=lg;k++) { blm[k]=1; lmd[k]=l[g[k]]-d[g[k]]; }

               for (i=1;i<=nhstrip;i++) {
                   j=lg;
                   gz=g[lg];
                   while (mu[gz]==lmd[j]) {
                       mu[gz]=l[gz];
                       mt[gz]=kt[gz];
                       slm-=d[gz];
                       j--;
                       gz=g[j];
                   }
                   t=kt[gz]-mt[gz];

                   zn=1+t;
                   dn=t+alpha;
                   for (k=1; k<gz; k++) {
                       q1=mt[k]-mt[gz];
                       q2=kt[k]-mt[gz];
                       zn*=(alpha-1+q1)*(1+q2);
                       dn*=q1*(alpha+q2);
                   }
                   blm[j]*=zn/dn;

                   mu[gz]--;
                   mt[gz]-=alpha;
                   slm++;

                   for (k=j+1;k<=lg;k++) blm[k]=blm[j];

                   /* next, find the index of mu */
                   nmu=mu[1]+1; for (k=2;k<=h-(mu[h]==0);k++) nmu=D[nmu]+mu[k]-1;


                   for (k=h+1; k<=n;k++)
                       Sx[w*n+k-1]+=blm[j]*Sx[nmu*n+k-2]*xn[k*(MAX+2)+slm];

                   if (y!=NULL) for (k=h+1; k<=n;k++)
                       Sy[w*n+k-1]+=blm[j]*Sy[nmu*n+k-2]*yn[k*(MAX+2)+slm];
              }

              for (k=h; k<n; k++) Sx[w*n+k]+=Sx[w*n+k-1];
              if (y!=NULL) {
                   for (k=h; k<n; k++) Sy[w*n+k]+=Sy[w*n+k-1];
                   if (z_log[h]!=0.0 || arg_log_scale!=0.0)
                       coef[sl]+=scaled_product(z_log[h]+sl*arg_log_scale, z[h],
                                               Sx[w*n+n-1], Sy[w*n+n-1],
                                               1.0, 3);
                   else coef[sl]+=z[h]*Sx[w*n+n-1]*Sy[w*n+n-1];
              }
              else {
                   if (z_log[h]!=0.0 || arg_log_scale!=0.0)
                       coef[sl]+=scaled_product(z_log[h]+sl*arg_log_scale, z[h],
                                               Sx[w*n+n-1], 1.0, 1.0, 2);
                   else coef[sl]+=z[h]*Sx[w*n+n-1];
              }

           } /* of "if h<n" */
           else {
               /* computing the index of the partition l-l[n]*ones(1,n) */
               nmu=l[1]-l[n]+1;
               k=2; while ((k<n)&&(l[k]>l[n])) nmu=D[nmu]+l[k++]-1-l[n];
               /* cc is 1 if l[n]==1, (guaranteed by the h<n case);
                  we then update from the previous */

               /* h==n branch.  cc accumulates a running product across
                * successive degrees up the n-th partition part; for full-rank
                * Omega the factor prodx[n]*prody[n] (~1e10) drives the bare
                * double cc past DBL_MAX before the series converges.  Carry cc
                * as (mantissa=cc, scale=cc_log) and renormalize so no
                * intermediate overflows; reconstruct the contribution in log
                * space.  Reduces to the original arithmetic exactly when no
                * renormalization fires (cc_log==0 => exp(cc_log)==1). */
               if (y!=NULL) {
                   t=(1/alpha+l[n]-1)/l[n];
                   for (k=1;k<n;k++) t*=(1+kt[k]-kt[n])/(alpha+kt[k]-kt[n]);
                   cc*=t*t*prodx[n]*prody[n];
                   if (fabs(cc)>CC_RENORM_HI) { cc*=CC_RENORM; cc_log+=CC_RENORM_LOG; }
                   else if (cc!=0.0 && fabs(cc)<CC_RENORM) {
                       cc*=CC_RENORM_HI; cc_log-=CC_RENORM_LOG;
                   }
                   if (cc_log!=0.0 || z_log[n]!=0.0 || arg_log_scale!=0.0) {
                       /* Reconstruct the contribution entirely in log space:
                          the true term z*cc_true*Sx*Sy ~ 1e107 is representable,
                          but cc_true ~ 1e313 alone is not -- so never form it. */
                       double sgn = (cc<0?-1.0:1.0)*(z[n]<0?-1.0:1.0)
                                    *(Sx[nmu*n+n-1]<0?-1.0:1.0)*(Sy[nmu*n+n-1]<0?-1.0:1.0);
                       (void)sgn; /* sign is reconstructed by scaled_product */
                       coef[sl]+=scaled_product(z_log[n]+cc_log+sl*arg_log_scale,
                                               z[n], cc,
                                               Sx[nmu*n+n-1],
                                               Sy[nmu*n+n-1], 4);
                   }
                   else coef[sl]+=z[n]*cc*Sx[nmu*n+n-1]*Sy[nmu*n+n-1];
               }
               else {
                   cc*=(1/alpha+l[n]-1)*prodx[n]/l[n];
                   for (k=1;k<n;k++) cc*=(1+kt[k]-kt[n])/(alpha+kt[k]-kt[n]);
                   if (fabs(cc)>CC_RENORM_HI) { cc*=CC_RENORM; cc_log+=CC_RENORM_LOG; }
                   else if (cc!=0.0 && fabs(cc)<CC_RENORM) {
                       cc*=CC_RENORM_HI; cc_log-=CC_RENORM_LOG;
                   }
                   if (cc_log!=0.0 || z_log[n]!=0.0 || arg_log_scale!=0.0) {
                       double sgn = (cc<0?-1.0:1.0)*(z[n]<0?-1.0:1.0)
                                    *(Sx[nmu*n+n-1]<0?-1.0:1.0);
                       (void)sgn; /* sign is reconstructed by scaled_product */
                       coef[sl]+=scaled_product(z_log[n]+cc_log+sl*arg_log_scale,
                                               z[n], cc,
                                               Sx[nmu*n+n-1], 1.0, 3);
                   }
                   else coef[sl]+=z[n]*cc*Sx[nmu*n+n-1];
               }
           }
           if (h<n) {
               z[h+1]=z[h];
               z_log[h+1]=z_log[h];
               h++;
               ww[h]=w;
           }
           sl++;
       }
       else {
           sl-=l[h];
           l[h]=0;
           kt[h]=-h;
           h--;
       }
   } /* of while h>0 */

   *s=0; for (i=0;i<MAX+1;i++) (*s)+=coef[i];

   free(lmd);
   free(blm);
   free(mt);
   free(g);
   free(d);
   free(ww);
   free(kt);
   free(mu);
   free(z_log);
   free(z);
   free(l);
   if (y!=NULL) { free(prody); free(yn); free(Sy); free(y); }
   free(prodx);
   free(xn);
   free(Sx);
   free(D);
   free(lambda);
   free(x);
   if (coef_out==NULL) free(coef);
   return 0;
}
