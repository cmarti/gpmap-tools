data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of shifts

  matrix[G, F] X[S];              // One hot encoded sequence  
  vector[G] log_gfp;              // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
    real positions[S];
    
    for(i in 1:S)
        positions[i] = i;
}

parameters {
    real mu_mean;
    real<lower=0> mu_sigma;
    row_vector[S] mu_raw;
    
    real<lower=0> alpha;
    real<lower=0> rho_inv;
    
    vector[F-1] theta_mu;
    matrix[S, F-1] theta_raw;
    
    real<lower=0> sigma; 
}

transformed parameters {
    matrix[G, S] log_ki;
    matrix[F, S] theta;
    row_vector[S] mu;
    vector[G] ki_sum;
    vector[G] yhat;
    real rho;
    
    matrix[S, S] L_K;

    rho = 1 / rho_inv;    
    L_K = cholesky_decompose(cov_exp_quad(positions, alpha, rho));
    
    mu = mu_mean + mu_raw * mu_sigma;
    
    theta[1,] = mu;
    theta[2:F,] = rep_matrix(theta_mu, S) + (L_K * theta_raw)';
    
    for(i in 1:S){
        log_ki[,i] = mu[i] + X[i] * theta[,i];
    }
    
    ki_sum = exp(log_ki) * ones;
    yhat = log(ki_sum) - log(1 + ki_sum);
}


model {
    // Prior on position dependent effects
    mu_mean ~ normal(0, 5);
    mu_raw ~ std_normal();             
    mu_sigma ~ normal(0, 2);
    
    // Prior on mean mutational effects
    theta_mu ~ std_normal();           
    to_row_vector(theta_raw) ~ std_normal();
    alpha ~ normal(0, 1);
    rho_inv ~ exponential(1);

    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    log_gfp ~ normal(yhat, sigma);
}