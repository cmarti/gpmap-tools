data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations

  vector[S] distances;
  matrix[G, F] X[S];              // One hot encoded sequence  
  vector[G] y;              // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
    
    real mean_dist = (max(distances) - min(distances)) / 2;
    real dist_sd = (max(distances) - min(distances)) / 4;
}

parameters {
    real mu_opt;
    real opt_dist;
    real<lower=0> mu_scale;
    
    vector[F] theta;
    
    real<lower=0> background;
    real<lower=0> sigma; 
}

transformed parameters {
    vector[S] mu;
    vector[S] dd;
    
    vector[G] ki_sum;
    vector[G] yhat;
    
    dd = distances - opt_dist;
    mu = mu_opt + mu_scale * (dd .* dd);
    
    ki_sum = rep_vector(0, G);
    for(i in 1:S)
        ki_sum = ki_sum + exp(-(mu[i] + X[i] * theta));
    
    yhat = log(background + ki_sum);
}


model {
    mu_opt ~ normal(0, 5);                  // Prior on wt effect at the optimal distance
    opt_dist ~ normal(mean_dist, dist_sd);  // Prior on optimal distance: centered in the observed values
    mu_scale ~ exponential(1);
    
    theta ~ std_normal();           // Prior on mutational effects

    background ~ exponential(2);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    y ~ normal(yhat, sigma);
}