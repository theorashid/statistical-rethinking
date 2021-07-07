# %%
import pandas as pd
import arviz as az
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO, log_likelihood
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


# %% Code 7.32
def model(A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    mu = numpyro.deterministic("mu", a + bA * A)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    A=df["A"].values,
    D=df["D"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% Code 7.33
# calculate WAIC and PSIS
logprob = log_likelihood(guide.model, samples, A=df["A"].values, D=df["D"].values)["D"]
ll = az.from_dict(
    posterior={k: v[None, ...] for k, v in samples.items()},
    log_likelihood={"D": logprob[None, ...]},
)

az.loo(ll, pointwise=True, scale="deviance")
az.waic(ll, pointwise=True, scale="deviance")
