# %%
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random, ops
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %% Code 15.16
milk = pd.read_csv("../data/milk.csv", sep=";")
df = milk
df["neocortex.prop"] = df["neocortex.perc"] / 100
df["logmass"] = df["mass"].apply(jnp.log).apply(float)

dat = dict(
    K=df["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std()).values,
    B=df["neocortex.prop"].pipe(lambda x: (x - x.mean()) / x.std()).values,
    M=df["logmass"].pipe(lambda x: (x - x.mean()) / x.std()).values,
)


# %% Code 15.17
# model milk energy using body mass (M) and neocortex proportion (B)
# model missing B values from a normal
# B is a likelihood when observed
# B is a prior when missing
def model(B, M, K):
    a = numpyro.sample("a", dist.Normal(0, 0.5))
    bB = numpyro.sample("bB", dist.Normal(0, 0.5))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))

    sigma = numpyro.sample("sigma", dist.Exponential(1))

    # impute missing neocortex proportion
    nu = numpyro.sample("nu", dist.Normal(0, 0.5))
    sigma_B = numpyro.sample("sigma_B", dist.Exponential(1))
    # impute when B is missing
    B_impute = numpyro.sample(
        "B_impute", dist.Normal(0, 1).expand([int(np.isnan(B).sum())]).mask(False)
    )
    B = ops.index_update(B, np.nonzero(np.isnan(B))[0], B_impute)
    numpyro.sample("B", dist.Normal(nu, sigma_B), obs=B)

    mu = a + bB * B + bM * M
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0), **dat)

mcmc.print_summary(0.89)


# %% Code 15.22
# B and M are associated as a result of U
# model them jointly from a bivariate normal
def model(B, M, K):
    a = numpyro.sample("a", dist.Normal(0, 0.5))
    bB = numpyro.sample("bB", dist.Normal(0, 0.5))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))

    sigma = numpyro.sample("sigma", dist.Exponential(1))

    # impute missing neocortex proportion
    # impute when B is missing
    B_impute = numpyro.sample(
        "B_impute", dist.Normal(0, 1).expand([int(np.isnan(B).sum())]).mask(False)
    )
    # define B_merge as mix of observed and imputed values
    B_merge = ops.index_update(B, np.nonzero(np.isnan(B))[0], B_impute)

    # M and B correlation
    muB = numpyro.sample("muB", dist.Normal(0, 0.5))
    muM = numpyro.sample("muM", dist.Normal(0, 0.5))
    Rho_BM = numpyro.sample("Rho_BM", dist.LKJ(2, 2))
    Sigma_BM = numpyro.sample("Sigma_BM", dist.Exponential(1).expand([2]))

    MB = jnp.stack([M, B_merge], axis=1)
    cov = jnp.outer(Sigma_BM, Sigma_BM) * Rho_BM
    numpyro.sample("MB", dist.MultivariateNormal(jnp.stack([muM, muB]), cov), obs=MB)

    mu = a + bB * B_merge + bM * M
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0), **dat)

mcmc.print_summary(0.89)
