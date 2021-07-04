# %%
import pandas as pd
import arviz as az
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")

# %% Code 6.8
# milk data
milk = pd.read_csv("../data/milk.csv", sep=";")
df = milk
df["K"] = df["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std())
df["F"] = df["perc.fat"].pipe(lambda x: (x - x.mean()) / x.std())
df["L"] = df["perc.lactose"].pipe(lambda x: (x - x.mean()) / x.std())


# %% Code 6.10
# model with both F (fat %) and L (lactose %)
def model(F, L, K):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bF = numpyro.sample("bF", dist.Normal(0, 0.5))
    bL = numpyro.sample("bL", dist.Normal(0, 0.5))
    mu = a + bF * F + bL * L

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    K=df["K"].values,
    F=df["F"].values,
    L=df["L"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
# %% Code 6.11
# perc.fat and perc.lactose are strongly (negatively) correlated
# multicollinearity (collider) L -> K <- F
az.plot_pair(df[["kcal.per.g", "perc.fat", "perc.lactose"]].to_dict("list"))
