data {
  int<lower=0> N;
  int<lower=1> D;
  matrix[N, D] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[D] beta;
}
model {
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  y ~ bernoulli_logit(alpha + X * beta);
}
