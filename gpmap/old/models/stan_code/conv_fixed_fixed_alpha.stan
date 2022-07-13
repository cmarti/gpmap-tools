data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations

  real alpha;                     // alpha
  matrix[G, F] X[S];              // One hot encoded sequence  
  vector[G] y;                    // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
}

parameters {
    real theta_0;
    real mu;
    vector[F] theta;
    
    real<lower=0> background;
    real<lower=0> sigma; 
}

transformed parameters {
    matrix[G, S] log_ki;
    
    vector[G] ki_sum;
    vector[G] yhat;
    
    for(i in 1:S){
        log_ki[,i] = -(mu + X[i] * theta);
    }
    
    ki_sum = exp(log_ki) * ones;
    yhat = log(background + exp(theta_0) * ki_sum ./ (1 + alpha * ki_sum));
}


model {
    theta_0 ~ normal(0, 5);           // Prior on saturation point
    mu ~ normal(0, 5);                // Prior on wt
    theta ~ normal(0, 0.5);           // Prior on mutational effects

    background ~ exponential(2);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    y ~ normal(yhat, sigma);
}