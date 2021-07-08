# %%
import pandas as pd
import arviz as az
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO, log_likelihood
from numpyro.infer.autoguide import AutoLaplaceApproximation

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")

# %% GDP and terrain ruggedness data
rugged = pd.read_csv("../data/rugged.csv", sep=";")
df = rugged

# make log version of outcome
df["log_gdp"] = df["rgdppc_2000"].apply(jnp.log).apply(float)

# extract countries with GDP data
df = df[df["rgdppc_2000"].notnull()].copy()

# rescale variables
df["log_gdp_std"] = df["log_gdp"] / df["log_gdp"].mean()
df["rugged_std"] = df["rugged"] / df["rugged"].max()

df["cid"] = jnp.where(df["cont_africa"].values == 1, 0, 1)


# %% Code 8.13
# model GDP with ruggedness for Africa/not-Africa
def model(cid, rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1).expand([2]))
    b = numpyro.sample("b", dist.Normal(0, 0.3).expand([2]))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged_std - 0.215))

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    rugged_std=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp_std=df["log_gdp_std"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
samples.pop("mu")
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% Code 8.15
# calculate WAIC
logprob = log_likelihood(
    guide.model,
    samples,
    rugged_std=df["rugged_std"].values,
    cid=df["cid"].values,
    log_gdp_std=df["log_gdp_std"].values,
)
ll = az.from_dict(log_likelihood={k: v[None] for k, v in logprob.items()})

az.waic(ll, pointwise=True, scale="deviance")

# %% tulip growth under different conditions
tulips = pd.read_csv("../data/tulips.csv", sep=";")
df = tulips

df["blooms_std"] = df["blooms"] / df["blooms"].max()
df["water_cent"] = df["water"] - df["water"].mean()
df["shade_cent"] = df["shade"] - df["shade"].mean()


# %% Code 8.24
# model with intercation between (continuous) water and shade
def model(water_cent, shade_cent, blooms_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 0.25))
    bw = numpyro.sample("bw", dist.Normal(0, 0.25))
    bs = numpyro.sample("bs", dist.Normal(0, 0.25))
    bws = numpyro.sample("bws", dist.Normal(0, 0.25))
    mu = a + bw * water_cent + bs * shade_cent + bws * water_cent * shade_cent

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("blooms_std", dist.Normal(mu, sigma), obs=blooms_std)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    water_cent=df["water_cent"].values,
    shade_cent=df["shade_cent"].values,
    blooms_std=df["blooms_std"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
