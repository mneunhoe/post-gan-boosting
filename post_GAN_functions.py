import numpy as np
from scipy.special import logit
from scipy.special import expit as logistic

# The rejection_sample function uses source code from 
# https: https://github.com/uber-research/metropolis-hastings-gans/blob/master/mhgan/mh.py,
# Copyright (c) 2018 Uber Technologies, Inc.

def rejection_sample(d_score, epsilon=1e-6, shift_percent=95.0, score_max=None,
                     random=np.random):
    '''Rejection scheme from:
    https://arxiv.org/pdf/1810.06758.pdf
    '''
    assert(np.ndim(d_score) == 1 and len(d_score) > 0)
    assert(0 <= np.min(d_score) and np.max(d_score) <= 1)
    assert(np.ndim(score_max) == 0)

    # Chop off first since we assume that is real point and reject does not
    # start with real point.
    d_score = d_score[1:]

    # Make sure logit finite
    d_score = np.clip(d_score.astype(np.float), 1e-14, 1 - 1e-14)
    max_burnin_d_score = np.clip(score_max.astype(np.float),
                                 1e-14, 1 - 1e-14)

    log_M = logit(max_burnin_d_score)

    D_tilde = logit(d_score)
    # Bump up M if found something bigger
    D_tilde_M = np.maximum(log_M, np.maximum.accumulate(D_tilde))

    D_delta = D_tilde - D_tilde_M
    F = D_delta - np.log(1 - np.exp(D_delta - epsilon))

    if shift_percent is not None:
        gamma = np.percentile(F, shift_percent)
        F = F - gamma

    P = logistic(F)
    accept = random.rand(len(d_score)) <= P

    if np.any(accept):
        idx = np.argmax(accept)  # Stop at first true, default to 0
    else:
        idx = np.argmax(d_score)  # Revert to cherry if no accept

    # Now shift idx because we took away the real init point
    return idx + 1, P[idx]
    
def post_gan_boosting(d_score_fake,
           d_score_real,
           B,
           real_N,
           steps = 400,
           N_generators = 200,
           uniform_init = True,
           dp = True,
           MW_epsilon = 0.1,
           weighted_average = False,
           averaging_window = None):
  if uniform_init:
    phi = np.repeat(1 / np.shape(B)[0], np.shape(B)[0])
  else:
    phi_unnormalized = np.random.uniform(size = np.shape(B)[0])
    phi = phi_unnormalized / np.sum(phi_unnormalized)

  # Set epsilon0 the privacy parameter for each update step
  epsilon0 = MW_epsilon / steps

  # Initialize matrices to hold the results to calculate the mixture distribution.
  # phi_matrix stores all distributions of phi from 1 to T (steps)
  phi_matrix = np.zeroes(shape = (steps, np.shape(B)[0]))

  # best_D_matrix stores the best response after each step
  best_D_matrix = np.zeroes(shape = (N_generators, steps), dtype = np.bool_)

  for step in range(0, steps):
    # Set learning rate (eta in the paper)
    learning_rate = 1 / np.sqrt(step)

    # Calculate the payoff U
    U_fake = np.matmul(d_score_fake, phi)
    # U as defined in Part 3.1
    U = -(((1 - d_score_real) + U_fake))

    # Selecting a discriminator using the payoff U as the quality score

    if dp:
      # With dp the exponential mechanism is applied to choose D
      # The best D is sampled with probability porportional to
      # exp(epsilon0 * U * nrow(X))/2
      # add a small constant for numerical stability
      exp_U = np.exp((epsilon0 * U * real_N) / (2)) + np.exp(-500)
        
      sum_exp_U = np.sum(exp_U)
      
       # Normalize to get valid probability distribution
      p_U = exp_U / sum_exp_U             

      # Sample the best discriminator. If multiple Discriminators have the same score,
      # pick the first one.
      best_D = np.arange(U.shape[0]) == np.where(U == np.random.choice(U, size = 1, p = p_U))
    else:
      best_D = U == np.max(U)

    best_D_matrix[:, step] = best_D

    # Update phi using the picked Discriminator
    phi_update = phi * np.exp(learning_rate * d_score_fake[best_D, :])
      
    # Normalize to get a valid probability distribution
    phi =  phi_update / np.sum(phi_update)
      
    # Store updated phi
    phi_matrix[step, :] = phi
      
    # Print progress
    print('Step: {}'.format(step))
    print()

  if weighted_average:
    PGB_prob = np.matmul(phi_matrix.T, np.sqrt(np.arange(steps)+1)) / np.sum(np.matmul(phi_matrix.T, np.sqrt(np.arange(steps)+1)))
  else:
    if averaging_window is None:
      averaging_window = steps
    PGB_prob = np.mean(phi_matrix[(phi_matrix.shape[0] - averaging_window):phi_matrix.shape[0], :], axis = 1)

  PGB_sel =   np.random.choice(np.arrange(B.shape[0]), p = PGB_prob, replace = True)
  
  # Select each sample at most once
  PGB_sel = np.unique(PGB_sel)
    
  # Subset B to the PGB examples
  PGB_sample = B[PGB_sel, :]
  
  # Calculate weighted discriminator scores D bar
  mix_D = np.mean(best_D_matrix, axis = 0)
  d_bar_fake = np.sum(d_scores_fake * mix_D[:, np.newaxis], axis = 1)
    
  d_score_PGB = d_bar_fake[PGB_sel]
  

  return PGB_sample, d_score_PGB  
      
