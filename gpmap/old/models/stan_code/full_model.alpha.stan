data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations
  int<lower=0> C;                 // number of contexts

  real slope;                     // slope of mRNA-Protein linear relationship  
  real intercept;                 // intercept of mRNA-Protein linear relationship

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
    
    real<lower=0, upper=1> alpha;
    
    real theta_mu;
    cholesky_factor_corr[C] Lcorr_theta;
    vector<lower=0>[C] theta_sigma;
    matrix[C, F] theta_raw;
    
    vector<lower=0>[C] background;
    vector<lower=0>[C] sigma; 
}

transformed parameters {
    matrix[G, S] log_ki[C];
    vector[S] mu;
    vector[S] dd;

    matrix[C, C] L_theta;
    matrix[C, F] theta;
    
    real theta_0;
    real translation_rate;
    
    matrix[G, C] ki_sum;
    matrix[G, C] yhat;
    
    // ribosome shielding related parameters
    translation_rate = (1 - alpha) / slope;
    theta_0 = translation_rate * intercept;

    // Sequence independent effects
    dd = distances - opt_dist;
    mu = mu_scale * (dd .* dd);
    
    // Sequence dependent effects
    L_theta = diag_pre_multiply(theta_sigma, Lcorr_theta);
    theta = theta_mu + L_theta * theta_raw;

    for(c in 1:C){    
        for(i in 1:S)
            log_ki[c][,i] = -X[c, i] * theta[c]';
        
        
        ki_sum[,c] = exp(log_ki[c]) * exp(-(mu_opt[c] + mu));
        yhat[,c] = log(background[c] + theta_0 * ki_sum[,c] ./ (1 + alpha * ki_sum[,c]));
    }
    
}


model {
    alpha ~ beta(1, 1);                     // Prior on modification of degradation rate by ribosome binding

    mu_opt ~ normal(0, 5);                  // Prior on wt effect at the optimal distance
    opt_dist ~ normal(mean_dist, dist_sd);  // Prior on optimal distance: centered in the observed values
    mu_scale ~ exponential(1);
    
    theta_mu ~ normal(1, 1);                           // Prior on mean mutational effects: detrimental
    theta_sigma ~ normal(0, 1);                        // Prior on mutational effects variances 
    Lcorr_theta ~ lkj_corr_cholesky(0.1);              // Prior on corrolation of mutational effects
    to_row_vector(theta_raw) ~ std_normal();           // Prior on mutational effects

    background ~ exponential(1);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    for(c in 1:C)
        y[,c] ~ normal(yhat[,c], sigma[c]);
}