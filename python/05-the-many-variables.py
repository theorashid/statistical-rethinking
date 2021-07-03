# %%
import pandas as pd
import arviz as az
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")

# %% marriage data
WaffleDivorce = pd.read_csv("../data/WaffleDivorce.csv", sep=";")
df = WaffleDivorce

# standardize variables
df["A"] = df["MedianAgeMarriage"].pipe(lambda x: (x - x.mean()) / x.std())
df["D"] = df["Divorce"].pipe(lambda x: (x - x.mean()) / x.std())
df["M"] = df["Marriage"].pipe(lambda x: (x - x.mean()) / x.std())


# %% Code 5.19
# estimate influence of A (median age of marriage) and M (marriage rate) on D (divorce rate)
def model(A, M=None, D=None):
    # A -> M
    aM = numpyro.sample("aM", dist.Normal(0, 0.2))
    bAM = numpyro.sample("bAM", dist.Normal(0, 0.5))
    sigma_M = numpyro.sample("sigma_M", dist.Exponential(1))
    mu_M = aM + bAM * A
    M = numpyro.sample("M", dist.Normal(mu_M, sigma_M), obs=M)

    # A -> D <- M
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bM * M + bA * A
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(0.1),
    loss=Trace_ELBO(),
    A=df["A"].values,
    M=df["M"].values,
    D=df["D"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% Code 5.21
# simulate M, D given a (standardised) sequence of median ages of mariage
post = samples
A_seq = jnp.linspace(-2, 2, num=30)

pred = Predictive(guide.model, post)
sim = pred(random.PRNGKey(2), A=A_seq)
# %% Code 5.24
# simulate the effect of M on D (fixing A)
M = jnp.linspace(-2, 2, num=30)
sim = pred(random.PRNGKey(2), A=0, M=M)
sim["D"]

# %% milk data
milk = pd.read_csv("../data/milk.csv", sep=";")
df = milk
df["K"] = df["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std())
df["N"] = df["neocortex.perc"].pipe(lambda x: (x - x.mean()) / x.std())
df["M"] = df["mass"].map(jnp.log).map(float).pipe(lambda x: (x - x.mean()) / x.std())

df = df.iloc[df[["K", "N", "M"]].dropna(how="any", axis=0).index]


# %% Code 5.39
# model milk energy (K) using female body mass (M) and neocortex mass (N)
def model(N, M, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bN = numpyro.sample("bN", dist.Normal(0, 0.5))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + bN * N + bM * M)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    N=df["N"].values,
    M=df["M"].values,
    K=df["K"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% Code 5.40
az.plot_forest(
    guide.sample_posterior(random.PRNGKey(1), params, (1, 1000)),
    var_names=["bM", "bN"],
    hdi_prob=0.89,
)
# %% Code 5.52
# categorical model for each clade
df["clade_id"] = df["clade"].astype("category").cat.codes


def model(clade_id, K):
    a = numpyro.sample("a", dist.Normal(0, 0.5).expand([len(set(clade_id))]))
    mu = a[clade_id]
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=K)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    clade_id=df["clade_id"].values,
    K=df["K"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

az.plot_forest(
    guide.sample_posterior(random.PRNGKey(1), params, (1, 1000)),
    hdi_prob=0.89,
)
