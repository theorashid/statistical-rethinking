# %%
import pandas as pd
import jax.numpy as jnp
from jax import random, ops
from jax.scipy.special import expit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# %% chipanzee prosocial experiment
chimpanzees = pd.read_csv("../data/chimpanzees.csv", sep=";")
df = chimpanzees
df["treatment"] = df["prosoc_left"] + 2 * df["condition"]
df["block_id"] = df["block"]


# %% Code 14.18
# Varing intercepts modelled jointly
def model(tid, actor, block_id, L):
    # fixed priors
    g = numpyro.sample("g", dist.Normal(0, 1).expand([4]))
    sigma_actor = numpyro.sample("sigma_actor", dist.Exponential(1).expand([4]))
    Rho_actor = numpyro.sample("Rho_actor", dist.LKJ(4, 4))
    sigma_block = numpyro.sample("sigma_block", dist.Exponential(1).expand([4]))
    Rho_block = numpyro.sample("Rho_block", dist.LKJ(4, 4))

    # adaptive priors
    cov_actor = jnp.outer(sigma_actor, sigma_actor) * Rho_actor
    alpha = numpyro.sample("alpha", dist.MultivariateNormal(0, cov_actor).expand([7]))
    cov_block = jnp.outer(sigma_block, sigma_block) * Rho_block
    beta = numpyro.sample("beta", dist.MultivariateNormal(0, cov_block).expand([6]))

    logit_p = g[tid] + alpha[actor, tid] + beta[block_id, tid]
    numpyro.sample("L", dist.Binomial(logits=logit_p), obs=L)


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    tid=df["treatment"].values - 1,
    actor=df["actor"].values - 1,
    block_id=df["block_id"].values - 1,
    L=df["pulled_left"].values,
)

mcmc.print_summary(0.89)


# %% Code 14.19
# Same as above but
def model(tid, actor, block_id, L=None, link=False):
    # fixed priors
    g = numpyro.sample("g", dist.Normal(0, 1).expand([4]))
    sigma_actor = numpyro.sample("sigma_actor", dist.Exponential(1).expand([4]))
    L_Rho_actor = numpyro.sample("L_Rho_actor", dist.LKJCholesky(4, 2))
    sigma_block = numpyro.sample("sigma_block", dist.Exponential(1).expand([4]))
    L_Rho_block = numpyro.sample("L_Rho_block", dist.LKJCholesky(4, 2))

    # adaptive priors - non-centered
    z_actor = numpyro.sample("z_actor", dist.Normal(0, 1).expand([4, 7]))
    z_block = numpyro.sample("z_block", dist.Normal(0, 1).expand([4, 6]))
    alpha = numpyro.deterministic(
        "alpha", ((sigma_actor[..., None] * L_Rho_actor) @ z_actor).T
    )
    beta = numpyro.deterministic(
        "beta", ((sigma_block[..., None] * L_Rho_block) @ z_block).T
    )

    logit_p = g[tid] + alpha[actor, tid] + beta[block_id, tid]
    numpyro.sample("L", dist.Binomial(logits=logit_p), obs=L)

    # compute ordinary correlation matrixes from Cholesky factors
    if link:
        numpyro.deterministic("Rho_actor", L_Rho_actor @ L_Rho_actor.T)
        numpyro.deterministic("Rho_block", L_Rho_block @ L_Rho_block.T)
        numpyro.deterministic("p", expit(logit_p))


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    tid=df["treatment"].values - 1,
    actor=df["actor"].values - 1,
    block_id=df["block_id"].values - 1,
    L=df["pulled_left"].values,
)

mcmc.print_summary(0.89)

# %% Code 14.23
# simulate effect of Education (E) on Wages (W)
# Instrumental variable Q associated with more education
# Q only influences W through E
with numpyro.handlers.seed(rng_seed=73):
    N = 500
    U_sim = numpyro.sample("U_sim", dist.Normal().expand([N]))
    Q_sim = numpyro.sample("Q_sim", dist.Categorical(logits=jnp.ones(4)).expand([N]))
    E_sim = numpyro.sample("E_sim", dist.Normal(U_sim + Q_sim))
    W_sim = numpyro.sample("W_sim", dist.Normal(U_sim + 0 * E_sim))
    dat_sim = dict(
        W=(W_sim - W_sim.mean()) / W_sim.std(),
        E=(E_sim - E_sim.mean()) / E_sim.std(),
        Q=(Q_sim - Q_sim.mean()) / Q_sim.std(),
    )


# %% Code 14.26
def model(E, Q, W):
    # influence of Q -> E
    aE = numpyro.sample("aE", dist.Normal(0, 0.2))
    bQE = numpyro.sample("bQE", dist.Normal(0, 0.5))
    muE = aE + bQE * Q

    # influence of E -> W
    aW = numpyro.sample("aW", dist.Normal(0, 0.2))
    bEW = numpyro.sample("bEW", dist.Normal(0, 0.5))
    muW = aW + bEW * E

    # Jointly model means of E and W
    # Off-diagonal correlation in rho is common influence of unobserved U
    Rho = numpyro.sample("Rho", dist.LKJ(2, 2))
    Sigma = numpyro.sample("Sigma", dist.Exponential(1).expand([2]))

    cov = jnp.outer(Sigma, Sigma) * Rho
    numpyro.sample(
        "W,E",
        dist.MultivariateNormal(jnp.stack([muW, muE], -1), cov),
        obs=jnp.stack([W, E], -1),
    )


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=4)
mcmc.run(random.PRNGKey(0), **dat_sim)

mcmc.print_summary(0.89)

# %% Code 14.37
# load the distance matrix
islandsDistMatrix = pd.read_csv("../data/islandsDistMatrix.csv", index_col=0)
Dmat = islandsDistMatrix

# %%  islands and tools data
Kline2 = pd.read_csv("../data/Kline2.csv", sep=";")
df = Kline2
df["society"] = range(1, 11)


# %% Code 14.39
def cov_GPL2(x, sq_eta, sq_rho, sq_sigma):
    N = x.shape[0]
    K = sq_eta * jnp.exp(-sq_rho * jnp.square(x))
    K = ops.index_add(K, jnp.diag_indices(N), sq_sigma)
    return K


def model(Dmat, P, society, T):
    a = numpyro.sample("a", dist.Exponential(1))
    b = numpyro.sample("b", dist.Exponential(1))
    g = numpyro.sample("g", dist.Exponential(1))

    # positive length scale and amplitude
    etasq = numpyro.sample("etasq", dist.Exponential(2))
    rhosq = numpyro.sample("rhosq", dist.Exponential(0.5))

    # covariance matrix
    SIGMA = cov_GPL2(Dmat, etasq, rhosq, 0.01)

    # intercepts based on distance between islands with GP prior
    k = numpyro.sample("k", dist.MultivariateNormal(0, SIGMA))

    lambda_ = a * P ** b / g * jnp.exp(k[society])
    numpyro.sample("T", dist.Poisson(lambda_), obs=T)


mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    T=df["total_tools"].values,
    P=df["population"].values,
    society=df["society"].values - 1,
    Dmat=islandsDistMatrix.values,
)

mcmc.print_summary(0.89)
