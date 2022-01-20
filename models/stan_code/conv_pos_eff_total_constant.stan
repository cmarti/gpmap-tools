data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations
  int<lower=0> P;                 // number of positions

  int<lower=1, upper=P> positions[S];
  matrix[G, F] X[S];              // One hot encoded sequence  
  vector[G] y;              // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
}

parameters {
    vector[P] mu_raw;
    real mu_mean;
    real<lower=0> mu_sd;
    
    real theta_0;
    vector[F] theta;
    
    real<lower=0> background;
    real<lower=0> sigma; 
}

transformed parameters {
    matrix[G, S] log_ki;
    vector[P] mu;
    
    vector[G] ki_sum;
    vector[G] yhat;
    
    mu = mu_mean + mu_raw * mu_sd;
    
    for(i in 1:S){
        log_ki[,i] = -(mu[positions[i]] + X[i] * theta);
    }
    
    ki_sum = exp(log_ki) * ones;
    yhat = log(background + exp(theta_0) * ki_sum ./ (1 + ki_sum));
}


model {
    theta_0 ~ normal(0, 5);           // Prior on saturation point
    mu_mean ~ normal(0, 5);         // Prior on wt on each position
    mu_sd ~ normal(0, 1);
    mu_raw ~ std_normal();
    
    theta ~ std_normal();           // Prior on mutational effects

    background ~ exponential(2);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    y ~ normal(yhat, sigma);
}