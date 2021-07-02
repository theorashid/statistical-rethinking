# %%
import pandas as pd
from scipy.interpolate import BSpline
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation

numpyro.set_platform("cpu")

# %% weight and height data
Howell1 = pd.read_csv("../data/Howell1.csv", sep=";")
df = Howell1[Howell1["age"] >= 18]


# %% Code 4.27, 4.28
# basic model
def model(height):
    mu = numpyro.sample("mu", dist.Normal(178, 20))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    height=df["height"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=2000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)


# %% Code 4.42
# linear model
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    # mu = a + b * (weight - xbar)
    mu = numpyro.deterministic("mu", a + b * (weight - jnp.mean(weight)))  # monitor mu

    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    weight=df["weight"].values,
    height=df["height"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=2000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
samples.pop("mu")
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% Code 4.59
# simulate height given a sequence of weights
post = samples
# define sequence of weights to compute predictions
weight_seq = jnp.arange(start=25, stop=71, step=1)

pred_height = Predictive(guide.model, post, return_sites=["height"])
sim_height = pred_height(random.PRNGKey(2), weight_seq, None)

# %%
# quadratic polynomial model
df = Howell1.copy()
df["weight_s"] = (df["weight"] - df["weight"].mean()) / df["weight"].std()
df["weight_s2"] = df["weight_s"] ** 2


def model(weight_s, weight_s2, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    # mu = a + b1 * weight_s + b2 * weight_s2
    mu = numpyro.deterministic("mu", a + b1 * weight_s + b2 * weight_s2)  # monitor mu

    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(0.3),
    loss=Trace_ELBO(),
    weight_s=df["weight_s"].values,
    weight_s2=df["weight_s2"].values,
    height=df["height"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=3000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
samples.pop("mu")
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% Code 4.69
# cubic polynomial model
df["weight_s3"] = df["weight_s"] ** 3


def model(weight_s, weight_s2, weight_s3, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    b3 = numpyro.sample("b3", dist.Normal(0, 1))
    # mu = a + b1 * weight_s + b2 * weight_s2 + b3 * weight_s3
    mu = numpyro.deterministic(
        "mu", a + b1 * weight_s + b2 * weight_s2 + b3 * weight_s3
    )  # monitor mu

    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(0.3),
    loss=Trace_ELBO(),
    weight_s=df["weight_s"].values,
    weight_s2=df["weight_s2"].values,
    weight_s3=df["weight_s3"].values,
    height=df["height"].values,
)

svi_result = svi.run(random.PRNGKey(0), num_steps=3000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
samples.pop("mu")
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)

# %% japanese cherry blossom data, day of year of blossom
cherry_blossoms = pd.read_csv("../data/cherry_blossoms.csv", sep=";")
df = cherry_blossoms

# %% Code 4.73, 4.74
# set knots for spline and basis
df = df[df["doy"].notna()]  # complete cases on doy
num_knots = 15
knot_list = jnp.quantile(
    df["year"].values.astype(float), q=jnp.linspace(0, 1, num=num_knots)
)

knots = jnp.pad(knot_list, (3, 3), mode="edge")  # pad ends with 3 extra
B = BSpline(knots, jnp.identity(num_knots + 2), k=3)(
    df["year"].values
)  # cubic (degree 3)
B.shape  # 827 x 17 (years in df x number of basis functions (num_knots + 2 ends))


# %% Code 4.76
# cubic spline model
def model(B, D):
    a = numpyro.sample("a", dist.Normal(100, 10))
    w = numpyro.sample("w", dist.Normal(0, 10).expand(B.shape[1:]))
    # mu = a + B @ w
    mu = numpyro.deterministic("mu", a + B @ w)

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


start = {"w": jnp.zeros(B.shape[1])}
guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
svi = SVI(
    model=model,
    guide=guide,
    optim=optim.Adam(1),
    loss=Trace_ELBO(),
    B=B,
    D=df["doy"].values,
)
svi_result = svi.run(random.PRNGKey(0), 20000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))

# %% Code 4.77, 4.78
w = jnp.mean(samples["w"], axis=0)
post_splines = [w[i] * B[:, i] for i in range(len(w))]

mu = jnp.mean(samples["w"], axis=0)
