# %%
import pandas as pd
import arviz as az
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %% chipanzee prosocial experiment
chimpanzees = pd.read_csv("../data/chimpanzees.csv", sep=";")
df = chimpanzees
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]


# %% Code 11.11
# logistic regression model
def model(actor, treatment, pulled_left=None, link=False):
    a = numpyro.sample("a", dist.Normal(0, 1.5).expand([7]))
    b = numpyro.sample("b", dist.Normal(0, 0.5).expand([4]))
    logit_p = a[actor] + b[treatment]
    if link:
        numpyro.deterministic("p", expit(logit_p))
    numpyro.sample("pulled_left", dist.Binomial(logits=logit_p), obs=pulled_left)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    treatment=df["treatment"].values,
    actor=df["actor"].values - 1,
    pulled_left=df["pulled_left"].values,
)

mcmc.print_summary(0.89)

# %% Code 11.12
post = mcmc.get_samples(group_by_chain=True)
p_left = expit(post["a"])
az.plot_forest({"p_left": p_left}, combined=True, hdi_prob=0.89)
idata = az.from_numpyro(mcmc)
az.waic(idata)

# %% number of tools on pacific islands
Kline = pd.read_csv("../data/Kline.csv", sep=";")
df = Kline
df["P"] = (
    df["population"]
    .apply(jnp.log)
    .apply(float)
    .pipe(lambda x: (x - x.mean()) / x.std())
)
df["contact_id"] = (df["contact"] == "high").astype(int)


# %% Code 11.45
# Poisson regression model
def model(cid, P, T=None):
    a = numpyro.sample("a", dist.Normal(3, 0.5).expand([2]))
    b = numpyro.sample("b", dist.Normal(0, 0.2).expand([2]))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(a[cid] + b[cid] * P))
    numpyro.sample("T", dist.Poisson(lambda_), obs=T)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    T=df["total_tools"].values,
    P=df["P"].values,
    cid=df["contact_id"].values,
)
mcmc.print_summary(0.89)

# %% Berkeley admissions data
UCBadmit = pd.read_csv("../data/UCBadmit.csv", sep=";")
df = UCBadmit


# %% Code 11.61
# binomial model of overall admission probability
def model(applications, admit):
    a = numpyro.sample("a", dist.Normal(0, 100))
    logit_p = a
    numpyro.sample("admit", dist.Binomial(applications, logits=logit_p), obs=admit)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    applications=df["applications"].values,
    admit=df["admit"].values,
)
mcmc.print_summary(0.89)
