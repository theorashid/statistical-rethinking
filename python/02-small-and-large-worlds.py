# %%
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation

numpyro.set_platform("cpu")


# %% Code 2.6
# quadratic approximation
def model(W, L):
    p = numpyro.sample("p", dist.Uniform(0, 1))
    numpyro.sample("W", dist.Binomial(W + L, p), obs=W)


guide = AutoLaplaceApproximation(model)

svi = SVI(
    model=model, guide=guide, optim=optim.Adam(0.001), loss=Trace_ELBO(), W=6, L=3
)

svi_result = svi.run(random.PRNGKey(0), num_steps=1000)
params = svi_result.params

samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
