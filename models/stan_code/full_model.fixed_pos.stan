data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations
  int<lower=0> C;                 // number of contexts

  matrix[G, F] X[C, S];              // feature matrix for each configuration
  matrix[G, C] y;                    // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
    
}

parameters {
    vector[C] mu;
    
    real theta_mu;
    real <lower=0> theta_sigma;
    vector[F] theta_raw;
    
    vector<lower=0>[C] background;
    vector<lower=0>[C] sigma; 
}

transformed parameters {
    vector[F] theta;
    
    matrix[G, C] ki_sum;
    matrix[G, C] yhat;
    
    // Sequence dependent effects
    theta = theta_mu + theta_sigma * theta_raw;

    for(c in 1:C){
        ki_sum[,c] = rep_vector(0, G);
            
        for(i in 1:S)
            ki_sum[,c] = ki_sum[,c] + exp(-X[c, i] * theta - mu[c]);
        
        yhat[,c] = log(background[c] + ki_sum[,c]);
    }
}


model {
    mu ~ normal(0, 5);                      // Prior on wt effect 
    
    theta_mu ~ normal(1, 1);                           // Prior on mean mutational effects: detrimental
    theta_sigma ~ normal(0, 1);                        // Prior on mutational effects variances 
    theta_raw ~ std_normal();           // Prior on mutational effects

    background ~ exponential(1);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    for(c in 1:C)
        y[,c] ~ normal(yhat[,c], sigma[c]);
}