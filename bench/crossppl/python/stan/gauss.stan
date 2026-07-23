data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> s;
}
model {
  mu ~ normal(0, 1);
  s ~ gamma(2, 1);
  y ~ normal(mu, s);
}
