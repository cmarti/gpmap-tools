data {
  int<lower=0> G;                 // number of genotypes
  int<lower=0> F;                 // number of features
  int<lower=0> S;                 // number of configurations

  real slope;                     // slope of mRNA-Protein linear relationship  
  real intercept;                 // intercept of mRNA-Protein linear relationship

  matrix[G, F] X[S];              // One hot encoded sequence  
  vector[G] y;                    // phenotype measurements
}

transformed data{
    vector[S] ones = rep_vector(1, S);
}

parameters {
    real<lower=0, upper=1> alpha;
    real mu;
    vector[F] theta;
    
    real<lower=0> background;
    real<lower=0> sigma; 
}

transformed parameters {
    matrix[G, S] log_ki;
    
    vector[G] ki_sum;
    vector[G] yhat;
    
    real theta_0;
    real translation_rate;
    
    translation_rate = (1 - alpha) / slope;
    theta_0 = translation_rate * intercept;
    
    for(i in 1:S){
        log_ki[,i] = -(mu + X[i] * theta);
    }
    
    ki_sum = exp(log_ki) * ones;
    yhat = log(background + theta_0 * ki_sum ./ (1 + alpha * ki_sum));
}


model {
    alpha ~ beta(1, 1);               // Prior on modification of degradation rate by ribosome binding
    mu ~ normal(0, 5);                // Prior on wt
    theta ~ normal(0, 0.5);           // Prior on mutational effects

    background ~ exponential(2);
    sigma ~ normal(0, 0.5); 
    
    // Likelihood
    y ~ normal(yhat, sigma);
}