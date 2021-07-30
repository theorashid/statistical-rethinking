# %%
import pandas as pd
import arviz as az
import jax.numpy as jnp
from jax import random
from jax.scipy.special import expit
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.distributions.transforms import OrderedTransform
from numpyro.infer import MCMC, NUTS, Predictive

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %% Berkeley admissions data
UCBadmit = pd.read_csv("../data/UCBadmit.csv", sep=";")
df = UCBadmit
df["gid"] = (df["applicant.gender"] != "male").astype(int)
dat = dict(N=df["applications"].values, A=df["admit"].values, gid=df["gid"].values)


# %% Code 12.2
# beta-binomial model of overall admission probability allowing overdispersion
def model(gid, N, A=None):
    a = numpyro.sample("a", dist.Normal(0, 1.5).expand([2]))
    phi = numpyro.sample("phi", dist.Exponential(1))
    theta = numpyro.deterministic("theta", phi + 2)
    pbar = expit(a[gid])
    numpyro.sample("A", dist.BetaBinomial(pbar * theta, (1 - pbar) * theta, N), obs=A)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0), **dat)
mcmc.print_summary(0.89)

# %% Code 12.3
post = mcmc.get_samples()
post["theta"] = Predictive(mcmc.sampler.model, post)(random.PRNGKey(1), **dat)["theta"]
post["da"] = post["a"][:, 0] - post["a"][:, 1]
print_summary(post, 0.89, False)

# %% number of tools on pacific islands
Kline = pd.read_csv("../data/Kline.csv", sep=";")
df = Kline
df["contact_id"] = (df["contact"] == "high").astype(int)


# %% Code 12.6
# Gamma-Poisson (negative binomial) regression model allowing overdispersion
def model(cid, P, T):
    a = numpyro.sample("a", dist.Normal(1, 1).expand([2]))
    b = numpyro.sample("b", dist.Exponential(1).expand([2]))
    g = numpyro.sample("g", dist.Exponential(1))
    phi = numpyro.sample("phi", dist.Exponential(1))
    lambda_ = jnp.exp(a[cid]) * jnp.power(P, b[cid]) / g
    numpyro.sample("T", dist.GammaPoisson(lambda_ / phi, 1 / phi), obs=T)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    T=df["total_tools"].values,
    P=df["population"].values,
    cid=df["contact_id"].values,
)
mcmc.print_summary(0.89)

# %% Code 12.7
# simulate monks drinking on 0.2 days (zero-inflating process, no manuscripts made)
# producing manuscripts on the other 0.8 (Poisson process)
prob_drink = 0.2  # 20% of days
rate_work = 1  # average 1 manuscript per day

# sample one year of production
N = 365

with numpyro.handlers.seed(rng_seed=365):
    # simulate days monks drink
    drink = numpyro.sample("drink", dist.Binomial(1, prob_drink).expand([N]))

    # simulate manuscripts completed
    y = (1 - drink) * numpyro.sample("work", dist.Poisson(rate_work).expand([N]))


# %% Code 12.9
# zero-inflated Poisson model
def model(y):
    ap = numpyro.sample("ap", dist.Normal(-1.5, 1))
    al = numpyro.sample("al", dist.Normal(1, 0.5))
    p = expit(ap)
    lambda_ = jnp.exp(al)
    numpyro.sample("y", dist.ZeroInflatedPoisson(p, lambda_), obs=y)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0), y=y)
mcmc.print_summary(0.89)

post = mcmc.get_samples()
print(jnp.mean(expit(post["ap"])))  # probability drink
print(jnp.mean(jnp.exp(post["al"])))  # rate finish manuscripts, when not drinking

# %% Trolley problem data
Trolley = pd.read_csv("../data/Trolley.csv", sep=";")
df = Trolley
edu_levels = [
    "Elementary School",
    "Middle School",
    "Some High School",
    "High School Graduate",
    "Some College",
    "Bachelor's Degree",
    "Master's Degree",
    "Graduate Degree",
]
cat_type = pd.api.types.CategoricalDtype(categories=edu_levels, ordered=True)
df["edu_new"] = df["edu"].astype(cat_type).cat.codes


# %% Code 12.43
# ordered categorical predictors, ordered categorical outcome
def model(action, intention, contact, E, alpha, R):
    bA = numpyro.sample("bA", dist.Normal(0, 1))
    bI = numpyro.sample("bI", dist.Normal(0, 1))
    bC = numpyro.sample("bC", dist.Normal(0, 1))
    bE = numpyro.sample("bE", dist.Normal(0, 1))
    delta = numpyro.sample("delta", dist.Dirichlet(alpha))
    kappa = numpyro.sample(
        "kappa",
        dist.TransformedDistribution(
            dist.Normal(0, 1.5).expand([6]), OrderedTransform()
        ),
    )
    delta_j = jnp.pad(delta, (1, 0))
    delta_E = jnp.sum(jnp.where(jnp.arange(8) <= E[..., None], delta_j, 0), -1)
    phi = bE * delta_E + bA * action + bI * intention + bC * contact
    numpyro.sample("R", dist.OrderedLogistic(phi, kappa), obs=R)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    R=df["response"].values - 1,
    action=df["action"].values,
    intention=df["intention"].values,
    contact=df["contact"].values,
    E=df["edu_new"].values,  # edu_new as an index
    alpha=jnp.repeat(2, 7),
)
mcmc.print_summary(0.89)
