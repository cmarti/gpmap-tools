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
    real alpha;
    real beta;
    
    real theta_mu;
    real <lower=0> theta_sigma;
    vector[F] theta_raw;
    
    vector<lower=0>[C] background;
    vector<lower=0>[C] sigma; 
}

transformed parameters {
    matrix[G, S] log_ki[C];
    vector[S] mu;
    vector[S] dd;

    vector[F] theta;
    
    matrix[G, C] ki_sum;
    matrix[G, C] yhat;
    
    // Sequence independent effects
    mu = alpha + beta * distances;
    
    // Sequence dependent effects
    theta = theta_mu + theta_sigma * theta_raw;

    for(c in 1:C){    
        for(i in 1:S)
            log_ki[c][,i] = -X[c, i] * theta;
        
        
        ki_sum[,c] = mu_opt[c] * exp(log_ki[c]) * inv_logit(mu);
        yhat[,c] = log(background[c] + ki_sum[,c]);
    }
    
}


model {
    mu_opt ~ normal(0, 5);                  // Prior on wt effect at the optimal distance
    alpha ~ normal(0, 2);  // Prior on optimal distance: centered in the observed values
    beta ~ normal(0, 1);
    
    theta_mu ~ normal(1, 1);                           // Prior on mean mutational effects: detrimental
    theta_sigma ~ normal(0, 1);                        // Prior on mutational effects variances 
    theta_raw ~ std_normal();           // Prior on mutational effects

    background ~ exponential(1);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    for(c in 1:C)
        y[,c] ~ normal(yhat[,c], sigma[c]);
}