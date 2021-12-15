data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of shifts

  matrix[G, F] X[S];              // One hot encoded sequence  
  vector[G] log_gfp;              // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
}

parameters {
    vector[F] theta;
    
    real<lower=0> sigma; 
}

transformed parameters {
    matrix[G, S] log_ki;
    
    vector[G] ki_sum;
    vector[G] yhat;
    
    for(i in 1:S){
        log_ki[,i] = X[i] * theta;
    }
    
    ki_sum = exp(log_ki) * ones;
    yhat = log(ki_sum) - log(1 + ki_sum);
}


model {
    theta[1] ~ normal(0, 5);        // Prior on reference genotype
    theta[2:F] ~ std_normal();      // Prior on mutational effects

    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    log_gfp ~ normal(yhat, sigma);
}