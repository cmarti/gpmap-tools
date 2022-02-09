data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations
  int<lower=0> C;                 // number of contexts
  int<lower=0> N;                 // number of filters

  vector[S] distances;
  
  matrix[G, F] X[C, S];              // feature matrix for each configuration
  matrix[G, C] y;                    // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
    
    real mean_dist = (max(distances) - min(distances)) / 2;
    real dist_sd = (max(distances) - min(distances)) / 4;
}

parameters {
    matrix[N, C] mu_opt;
    ordered[N] opt_dist;
    vector<lower=0>[N] mu_scale;
    
    vector[N] theta_mu;
    vector<lower=0>[N] theta_sigma;
    matrix[N, F] theta_raw;
    
    vector<lower=0>[C] background;
    vector<lower=0>[C] sigma; 
}

transformed parameters {
    matrix[S, N] mu;
    vector[S] dd;

    matrix[F, N] theta;
    
    matrix[G, C] ki_sum;
    matrix[G, C] yhat;
    
    // Sequence independent effects
    for(n in 1:N){
        dd = distances - opt_dist[n];
        mu[,n] = mu_scale[n] * (dd .* dd);
    }
    
    // Sequence dependent effects
    theta = (rep_matrix(theta_mu, F) + diag_pre_multiply(theta_sigma, theta_raw))';

    for(c in 1:C){    
        ki_sum[,c] = rep_vector(0, G);
        
        for(i in 1:S)
            for(n in 1:N)
                ki_sum[,c] = ki_sum[,c] + exp(-X[c, i] * theta[,n]) * exp(-(mu_opt[n, c] + mu[i, n]));
        yhat[,c] = log(background[c] + ki_sum[,c]);
    }
    
}


model {
    to_row_vector(mu_opt) ~ normal(0, 5);                  // Prior on wt effect at the optimal distance
    opt_dist ~ normal(mean_dist, dist_sd);  // Prior on optimal distance: centered in the observed values
    mu_scale ~ exponential(1);
    
    theta_mu ~ normal(1, 1);                           // Prior on mean mutational effects: detrimental
    theta_sigma ~ normal(0, 1);                        // Prior on mutational effects variances 
    to_row_vector(theta_raw) ~ std_normal();           // Prior on mutational effects

    background ~ exponential(1);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    for(c in 1:C)
        y[,c] ~ normal(yhat[,c], sigma[c]);
}