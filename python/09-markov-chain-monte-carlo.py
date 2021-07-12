# %%
import pandas as pd
import arviz as az
import jax.numpy as jnp
from jax import ops, random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)


# Overthinking: Hamiltonian Monte Carlo in the raw.
# %% Code 9.5
# 1. U - returns negative log-probability of the data
def U(q, a=0, b=1, k=0, d=1):
    muy = q[0]
    mux = q[1]
    logprob_y = jnp.sum(dist.Normal(muy, 1).log_prob(y))
    logprob_x = jnp.sum(dist.Normal(mux, 1).log_prob(x))
    logprob_muy = dist.Normal(a, b).log_prob(muy)
    logprob_mux = dist.Normal(k, d).log_prob(mux)
    U = logprob_y + logprob_x + logprob_muy + logprob_mux
    return -U


# %% Code 9.6
# 2. grad_U - returns the gradient of the negative log-probability
def U_gradient(q, a=0, b=1, k=0, d=1):
    muy = q[0]
    mux = q[1]
    G1 = jnp.sum(y - muy) + (a - muy) / b ** 2  # dU/dmuy for Gaussian U = log(N(y|a,b))
    G2 = jnp.sum(x - mux) + (k - mux) / d ** 2  # dU/dmuy
    return jnp.stack([-G1, -G2])  # negative because energy is neg-log-prob


# test data
with numpyro.handlers.seed(rng_seed=7):
    y = numpyro.sample("y", dist.Normal().expand([50]))
    x = numpyro.sample("x", dist.Normal().expand([50]))
    x = (x - jnp.mean(x)) / jnp.std(x)
    y = (y - jnp.mean(y)) / jnp.std(y)


# 3. a step size, epsilon
# 4. a count of leapfrog steps, L
# 5. a starting position (current value)
# %% Code 9.8
def HMC2(U, grad_U, epsilon, L, current_q, rng):
    q = current_q

    # random flick - p is momentum
    p = dist.Normal(0, 1).sample(random.fold_in(rng, 0), (q.shape[0],))
    current_p = p
    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2

    # initialize bookkeeping - saves trajectory
    qtraj = jnp.full((L + 1, q.shape[0]), jnp.nan)
    ptraj = qtraj
    qtraj = ops.index_update(qtraj, 0, current_q)
    ptraj = ops.index_update(ptraj, 0, p)

    # Loop over leapfrog steps (linear jumps of size epsilon over log-posterior surface)
    # using graident to compute a linear approximation of the log-posterior surface
    # Alternate full steps for position and momentum
    for i in range(L):
        q = q + epsilon * p  # Full step for the position
        # Make a full step for the momentum, except at end of trajectory
        if i != (L - 1):
            p = p - epsilon * grad_U(q)
            ptraj = ops.index_update(ptraj, i + 1, p)
        qtraj = ops.index_update(qtraj, i + 1, q)

    # Make a half step for momentum at the end
    p = p - epsilon * grad_U(q) / 2
    ptraj = ops.index_update(ptraj, L, p)
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p

    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = jnp.sum(current_p ** 2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p ** 2) / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    accept = 0
    runif = dist.Uniform().sample(random.fold_in(rng, 1))
    if runif < jnp.exp(current_U - proposed_U + current_K - proposed_K):
        new_q = q  # accept
        accept = 1
    else:
        new_q = current_q  # reject
    return {
        "q": new_q,
        "traj": qtraj,
        "ptraj": ptraj,
        "accept": accept,
        "dH": proposed_U + proposed_K - (current_U + current_K),
    }


# %% Code 9.7
Q = {}
Q["q"] = jnp.array([-0.1, 0.2])
pr = 0.31
step = 0.03
L = 11  # 0.03/28 for U-turns --- 11 for working example
n_samples = 4
path_col = (0, 0, 0, 0.5)

for i in range(n_samples):
    print(i)
    Q = HMC2(U, U_gradient, step, L, Q["q"], random.fold_in(random.PRNGKey(0), i))
    print(Q)

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

dat_slim = {
    "log_gdp_std": df["log_gdp_std"].values,
    "rugged_std": df["rugged_std"].values,
    "cid": df["cid"].values,
}


# %% Code 9.16
def model(cid, rugged_std, log_gdp_std):
    a = numpyro.sample("a", dist.Normal(1, 0.1).expand([2]))
    b = numpyro.sample("b", dist.Normal(0, 0.3).expand([2]))
    mu = a[cid] + b[cid] * (rugged_std - 0.215)
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=2)
mcmc.run(random.PRNGKey(0), **dat_slim)

mcmc.print_summary(0.89)

# %% Code 9.19
idata = az.from_numpyro(mcmc)
az.summary(idata)
az.waic(idata)
az.loo(idata)

az.plot_pair(idata)

# %% Code 9.20
az.plot_trace(idata)

# %% Code 9.21
az.plot_rank(idata)
