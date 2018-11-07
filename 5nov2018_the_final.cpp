/** Written by Sophie M. Fosson 2018 **/
/**  Code for paper "Non-convex approach to binary compressed" **/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <Eigen/Dense>
#include <ctime>
using namespace std;
using namespace Eigen;

#define pi 3.1415926535897932384626433832795028841971
#define MAX_ITERATIONS 1e5
#define ERROR_TOL 1e-6

double sigma, RSE,  rho, SNR, LAMBDA, RSE_save;
int N, M, h, i, j, k, t, t_save,t_out, t_tot, run, reshuffle, fp,fn, num_reweight, method;
bool thereisnoise, r ,q, knowk;

//Standard Gaussian r.v. (mean=0; std=sigma)  via Box-Muller Algorithm
double Gaussian_Noise(double std) {
    return std*(sqrt(-2*log( ((double) rand() / (RAND_MAX))  )))*cos(2*pi* ((double) rand() / (RAND_MAX))  );
}

int main (int argc, char *argv[]) {
	if (argc < 6) {
		cerr << "Usage: " << argv[0] << " <signal length N> <sparsity k> <# measurements M> <run> <0=LASSO, 1=RW/RWR, 2=BP> <RWR (reshuffle)? 0/1> <(only for LASSO/BP) final quant? 0/1> <You know k? 0/1> [0=NO; 1=YES]"  << endl;
                cerr << "Ex: ./gofinal 100 5 20 47 1 1 0 0" << endl;
		return EXIT_FAILURE;
	}
	N = atoi (argv[1]);
	k = atoi (argv[2]);
	M = atoi (argv[3]);
	run = atoi(argv[4]);
	method = atoi(argv[5]); // lasso - reweight - bp
	r = atoi(argv[6]); // reshuffle
	q = atoi(argv[7]); // quantize
	knowk = atoi(argv[8]); // know k
    
    if (method == 1) {
    	num_reweight=4; // 1 initial LASSO + 4 reweigthing 
    } else if (method == 0){
    	num_reweight=0; // no reweighting
    }
    rho = 1;
	srand(10*run*N);
	
	VectorXd y, x, xOriginal, z_old, primal_residual, dual_residual, difference, z, dual, beta, final_error,qq, x_save;
	MatrixXd A, A_T, QI, ID, Q, P, A_A_T;
	A.resize(M, N);
 	xOriginal.resize(N);
	x.resize(N);
	beta.resize(N);
 	ID.setIdentity(N,N);
	dual.resize (N);
 	z.resize (N);

 	for (i = 0; i < M; i++) {
		for (h=0; h< N; h++) {
			A(i,h)=1./sqrt(M)*Gaussian_Noise(1);
		}
	}
	

	if (knowk) { // in such case, input # meausurements must be but M+1, not M
		for (h=0; h< N; h++) {
			A(M-1,h)=1;
		}
	}
	

	A_T=A.transpose();
    
	/** 1. GENERATE SIGNAL **/
	VectorXd index_nonzero(N);
	for (j = 0; j < N; j++) {
		index_nonzero(j)=j;
	}

	PermutationMatrix<Dynamic,Dynamic> perm(N);
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
        index_nonzero = perm * index_nonzero; 
	
	for (j = 0; j < k; j++) {
        xOriginal(index_nonzero(j))= 1;
	}
	
	
	/** 2. GENERATE MEASUREMENTS **/
	y = A*xOriginal;
        
	LAMBDA = 1e-2;
	
    time_t tstart, tend; 
 	tstart = time(0);
	if (method < 2) {

    	/** 3. RECOVERY VIA  LASSO - RW **/	
    	RSE_save = 1e4;
		t_save = MAX_ITERATIONS;
    	Q=(A_T*A+rho*ID); 
    	QI=Q.inverse();
    	for (i = 0; i < N; i++) {
			z(i)=0;
			dual(i)=0;
			beta(i)=1;  
    	}
     

		t_tot=0;
		for (reshuffle = 0; reshuffle < 20; reshuffle ++)    {
    
    		for (t_out = 0; t_out < num_reweight+1; t_out ++)    {

				/** ADMM TO SOLVE LASSO **/
        		for (t = 1; t < MAX_ITERATIONS; t++)    {
        		
        			// update x
            		x=QI*( A_T*y + rho*z - dual );
            	
            		// update z
                	z_old=z;
            		for (i = 0; i < N; i++) {
                		if ( abs( x(i)+ dual(i)/rho ) <   LAMBDA*beta(i)/rho )
                    		z(i) = 0;
                		else if ( x(i) + dual(i)/rho > 0)
                    		z(i) = x(i) + dual(i)/rho - LAMBDA*beta(i)/rho;
                		else if ( x(i)+ dual(i)/rho < 0)
							z(i) = x(i)+ dual(i)/rho + LAMBDA*beta(i)/rho;
            		}
            
            		// update dual
            		dual=dual+rho*(x-z);

            		//stop criterion (see Boyd book Section 3.3.1)
            		primal_residual = x- z;
                	dual_residual= z-z_old;
            		if (primal_residual.squaredNorm()+dual_residual.squaredNorm() < ERROR_TOL) {
                		t_tot = t_tot+t;
                		break;
            		}
            	
        		}/** end t, END ADMM **/

        		t_tot = t_tot+t;
        		//projection
        		for (i=0; i<N; i++) {         
        			if (x(i)>1)
            			x(i)=1;
            		if (x(i)<0)
            			x(i)=0;
        		}
        
        		//reweight initialize
        		for (i = 0; i < N; i++) {
					beta(i)=1-abs(x(i));  //weight MCP
					dual(i)=0;
            		z(i)=0;
        		} 
    		} //end tout

    		if (r) {
        		difference = x - xOriginal;
            	RSE=difference.squaredNorm();
            	if ( RSE < 1e-4 ) {
            		x_save = x;
					t_save = t;
					break;
				}
				else {
					if (RSE < RSE_save) {
						x_save = x;
						t_save = t;
						RSE_save = RSE;
					}
					for (i = 0; i < N; i++) {
						z(i) = 0;  
						dual(i) = 0;
						beta(i) = ((double) rand() / (RAND_MAX)); //random shuffle!
					}  
				}
			}
			else
				break;
		} // end reshuffle
	}
	else { // if BP
		 
		/** 3. RECOVERY VIA BP**/	
    	RSE_save = 1e4;
		t_save = MAX_ITERATIONS;
    	A_A_T = A*A_T;
		P = ID - A_T * A_A_T.inverse() * A;
		qq = A_T * A_A_T.inverse()*y;
    	for (i = 0; i < N; i++) {
			z(i)=0;
			dual(i)=0;
			beta(i)=1;  
    	}
		t_tot=0;
	

		/** ADMM TO SOLVE BP**/
        for (t = 1; t < MAX_ITERATIONS; t++)    {
        		
        	// update x
        	x = P*(z - dual) + qq;
    	
            // update z
            z_old=z;
    		for (i = 0; i < N; i++) {
    			if ( abs( x(i)+ dual(i)/rho ) <   beta(i)/rho )
        			z(i) = 0;
        		else if ( x(i) + dual(i)/rho > 0)
               		z(i) = x(i) + dual(i)/rho - beta(i)/rho;
        		else if ( x(i)+ dual(i)/rho < 0)
					z(i) = x(i)+ dual(i)/rho + beta(i)/rho;
   			}
            
            // update dual
            dual=dual+rho*(x-z);

			//stop criterion (see Boyd book Section 3.3.1)
            primal_residual = x-z;
            dual_residual= z-z_old;
            if (primal_residual.squaredNorm()+dual_residual.squaredNorm() < ERROR_TOL) {
            	t_tot = t_tot+t;
            	break;
            }
        }/** end t, END ADMM **/
		
		t_tot = t_tot+t;
        
        // projection
        for (i=0; i<N; i++) {         
        	if (x(i)>1)
        		x(i)=1;
        	if (x(i)<0)
        		x(i)=0;
        }
   	}

   	/** post processing **/
   	
	//cleaning
    for (i=0; i<N; i++) {         
        if (x(i)<1e-2)
            x(i)=0;
    }

    // possible final quantization
    if (q) {
    	for (i=0; i<N; i++) {
    		if (x(i)<0.5)
            	x(i)=0;
        	else 
        		x(i)=1;	
        }
    }

    fn=0;
    fp=0;
    for (i=0; i<N; i++) {         
        if ( (x(i) == 0) && (xOriginal(i)) )
        	fn=fn+1;
        if ( (x(i)) && (xOriginal(i) == 0) )
        	fp=fp+1;
    }
    
	tend = time(0);
    final_error = x- xOriginal;
    cout << final_error.squaredNorm()/xOriginal.squaredNorm() << "  " << fn << "  " << fp << "  " <<t_tot << "  "<<difftime(tend, tstart) << " "<< ((final_error.squaredNorm()/xOriginal.squaredNorm() <1e-3) && (fp == 0) && (fn == 0))<< endl;

	return EXIT_SUCCESS;
}
