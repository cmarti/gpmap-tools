data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations
  int<lower=0> C;                 // number of contexts
  int<lower=0> B;                 // number of bulge features

  int<lower=1, upper=F> bulges_features[B];
  int<lower=1, upper=F> mut_features[F-B];
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
    
    real theta_mu;
    real <lower=0> theta_sigma;
    vector[F-B] theta_mut_raw;
    vector<lower=0>[B] theta_bulge_raw;
    
    vector<lower=0>[C] background;
    vector<lower=0>[C] sigma; 
}

transformed parameters {
    vector[S] mu;
    vector[S] dd;

    vector[F] theta;
    
    matrix[G, C] ki_sum;
    matrix[G, C] yhat;
    
    // Sequence independent effects
    dd = distances - opt_dist;
    mu = mu_scale * (dd .* dd);
    
    // Sequence dependent effects
    theta[mut_features] = theta_mu + theta_sigma * theta_mut_raw;
    theta[bulges_features] = 1 ./ theta_bulge_raw;

    for(c in 1:C){
        ki_sum[,c] = rep_vector(0, G);
            
        for(i in 1:S)
            ki_sum[,c] = ki_sum[,c] + exp(-X[c, i] * theta - mu_opt[c] - mu[i]);
        
        yhat[,c] = log(background[c] + ki_sum[,c]);
    }
}


model {
    mu_opt ~ normal(0, 5);                  // Prior on wt effect at the optimal distance
    opt_dist ~ normal(mean_dist, dist_sd);  // Prior on optimal distance: centered in the observed values
    mu_scale ~ exponential(1);
    
    theta_mu ~ normal(1, 1);                           // Prior on mean mutational effects: detrimental
    theta_sigma ~ normal(0, 1);                        // Prior on mutational effects variances 
    theta_mut_raw ~ std_normal();            // Prior on mutational effects
    theta_bulge_raw ~ exponential(2);        // Prior on inverse bulge effects

    background ~ exponential(1);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    for(c in 1:C)
        y[,c] ~ normal(yhat[,c], sigma[c]);
}
