data {
  int<lower=0> G;                 // number of total genotypes
  int<lower=0> O;                 // number of observed genotypes
  
  int<lower=0> N;                 // number of non-zero entries in L
  int<lower=0> M;                 // number of rows with non-zero entries in L
  
  int<lower=0> D;                 // number of possible hamming distances
  
  int<lower=0> L_v[N];            // indexes of L non-zero values
  int<lower=0> L_u[M];            // row indexes of L non-zero values
  vector[N] L_w;                  // L non-zero values
  
  matrix[D, D] W_kd;              // Krawchauk matrix   
  matrix[D, D] M_inv;             // L polynomials entries matrix inverse
  vector[D] m_k;                  // eigenvalue multiplicities
    
  int<lower=0, upper=G> idx[O];   // indexes of observed phenotypes
  real y[O];                      // observed phenotypic values
  real<lower=0> sigma;
  
}

parameters {
    real<lower=0> log_lambda_beta_inv;
    real log_lambda0;
    
    real ymean;
    
    vector[G] y_raw;
}

transformed parameters {
    vector<lower=0>[D] lambdas;
    matrix[G, D] L_powers;
    vector[D] beta;
    vector[G] yhat;
    
    L_powers[,1] = y_raw;
    lambdas[1] = 0;
    for(i in 2:D){
        L_powers[,i] = csr_matrix_times_vector(G, G, L_w, L_v, L_u, L_powers[,i-1]);
        lambdas[i] = exp(log_lambda0 - (i-2) / log_lambda_beta_inv);
    }
    
    beta = M_inv * (W_kd' * lambdas);
    yhat = ymean + L_powers * beta;
}


model {
    ymean ~ normal(0, 1);
    
    log_lambda0 ~ normal(0, 5);
    log_lambda_beta_inv ~ exponential(5);
    
    y_raw ~ std_normal();
    y ~ normal(yhat[idx], sigma);
}
