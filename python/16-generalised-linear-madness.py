# %%
import pandas as pd
import jax.numpy as jnp
from jax import random
from jax.experimental.ode import odeint
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %% Code 16.13
# dynamics of lynx and hare populations
Lynx_Hare = pd.read_csv("../data/Lynx_Hare.csv", sep=";")


# %% Code 16.17
def dpop_dt(pop_init, t, theta):
    """
    Lokta-Volterra model of the evolution of predator-prey populations:
    - prey: population * (birth rate - death rate * number of predators)
    - predator: population * (number of predators * birth rate - death rate)

    :param t: time
    :param pop_init: initial state {lynx, hares}
    :param theta: parameters
    """
    L, H = pop_init[0], pop_init[1]
    bh, mh, ml, bl = theta[0], theta[1], theta[2], theta[3]
    # differential equations
    dH_dt = H * (bh - mh * L)
    dL_dt = L * (bl * H - ml)
    return jnp.stack([dL_dt, dH_dt])


# %% Code 16.18
def Lynx_Hare_model(N, pelts=None):
    """
    :param int N: number of measurement times
    :param pelts: measured populations
    """
    # Half-Noraml priors for birth and death rates bh, mh, ml, bl (positive)
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.tile(jnp.array([1, 0.05]), 2),
            scale=jnp.tile(jnp.array([0.5, 0.05]), 2),
        ),
    )

    # uncorrelated measurement errors on hare and lynx pelts
    sigma = numpyro.sample("sigma", dist.Exponential(1).expand([2]))

    # initial population state
    pop_init = numpyro.sample("pop_init", dist.LogNormal(jnp.log(10), 1).expand([2]))
    # trap rate
    p = numpyro.sample("p", dist.Beta(40, 200).expand([2]))

    # Evolution of the hare and lynx populations by Lokta-Volterra
    # N including the first time (initial state)
    times_measured = jnp.arange(float(N))
    pop = numpyro.deterministic(
        "pop",
        odeint(
            dpop_dt, pop_init, times_measured, theta, rtol=1e-5, atol=1e-3, mxstep=500
        ),
    )

    # observation model
    # connect latent population state to observed pelts
    # population * trap rates -> pelts
    numpyro.sample("pelts", dist.LogNormal(jnp.log(pop * p), sigma), obs=pelts)


mcmc = MCMC(
    NUTS(Lynx_Hare_model, target_accept_prob=0.95),
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
)
mcmc.run(random.PRNGKey(0), N=Lynx_Hare.shape[0], pelts=Lynx_Hare.values[:, 1:3])

mcmc.print_summary(0.89)

# %% Code 16.19
# posterior predictions of species populations
post = mcmc.get_samples()
predict = Predictive(mcmc.sampler.model, post, return_sites=["pelts", "pop"])
post = predict(random.PRNGKey(1), N=Lynx_Hare.shape[0])

# pelt has uncorrelated measurement error
# populations do not have measurement error (smooth)
# dimensions are [sample, year, lynx/hare]
post["pelts"][:, :, :], post["pop"][:, :, :]
