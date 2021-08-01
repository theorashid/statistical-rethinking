# %%
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)


# %% Code 13.26
# The Devil's funnel, centred
def model():
    v = numpyro.sample("v", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(0, jnp.exp(v)))


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0))
mcmc.print_summary()


# %% Code 13.26
# The Devil's funnel, uncentred
def model():
    v = numpyro.sample("v", dist.Normal(0, 3))
    z = numpyro.sample("z", dist.Normal(0, 1))
    numpyro.deterministic("x", z * jnp.exp(v))


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0))
mcmc.print_summary(exclude_deterministic=False)
