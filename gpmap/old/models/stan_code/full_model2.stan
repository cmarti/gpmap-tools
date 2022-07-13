data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations
  int<lower=0> C;                 // number of contexts

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
    vector[C] mu_opt;
    real opt_dist;
    real<lower=0> mu_scale;
    
    real theta0_mu;
    real<lower=0> theta0_sigma;
    vector[F] theta0_raw;
    vector<lower=0>[C-1] dtheta_sigma;
    matrix[C-1, F] dtheta_raw;
    
    vector<lower=0>[C] background;
    vector<lower=0>[C] sigma; 
}

transformed parameters {
    matrix[G, S] log_ki[C];
    vector[S] mu;
    vector[S] dd;

    matrix[C, F] theta;
    
    matrix[G, C] ki_sum;
    matrix[G, C] yhat;
    
    // Sequence independent effects
    dd = distances - opt_dist;
    mu = mu_scale * (dd .* dd);
    
    // Sequence dependent effects
    theta[1] = to_row_vector(theta0_mu + theta0_sigma * theta0_raw);
    for (c in 2:C)
        theta[c] = theta[1] + dtheta_sigma[c-1] * dtheta_raw[c-1];

    for(c in 1:C){    
        for(i in 1:S)
            log_ki[c][,i] = -X[c, i] * theta[c]';
        
        
        ki_sum[,c] = exp(log_ki[c]) * exp(-(mu_opt[c] + mu));
        yhat[,c] = log(background[c] + ki_sum[,c]);
    }
    
}


model {
    mu_opt ~ normal(0, 5);                  // Prior on wt effect at the optimal distance
    opt_dist ~ normal(mean_dist, dist_sd);  // Prior on optimal distance: centered in the observed values
    mu_scale ~ exponential(1);
    
    theta0_mu ~ normal(1, 1);                           // Prior on mean mutational effects: detrimental
    theta0_sigma ~ normal(0, 1);                        // Prior on mutational effects variances 
    theta0_raw ~ std_normal();                          // Prior on mutational effects
    
    dtheta_sigma ~ exponential(2);
    to_row_vector(dtheta_raw) ~ std_normal();           

    background ~ exponential(1);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    for(c in 1:C)
        y[,c] ~ normal(yhat[,c], sigma[c]);
}