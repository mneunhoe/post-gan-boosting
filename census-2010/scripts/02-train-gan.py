import tensorflow as tf

import numpy as np
import os
import time
from tensorflow_privacy.privacy.optimizers import dp_optimizer
import tensorflow_probability as tfp
import random
import pandas as pd

runs = 3

first_run = 4

GPU =  0

filename = "gan-input/gan_input.csv"
N = sum(1 for line in open(filename)) - 1


s = 10 #desired sample size
skip = sorted(random.sample(range(1,N+1),N-s)) #the 0-indexed header will not be included in the skip list

df = pd.read_csv(filename, skiprows=skip)

data = pd.read_csv(filename, header = 0)

Dim = np.shape(df)[1]

Z_dim = 512
mb_size = 500
D_learning_rate = 0.01

epochs = 10

# DP declarations

# DP-SGD
num_microbatches = mb_size
l2_norm_clip = 2.
noise_multiplier = 1.1

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)


X = tf.compat.v1.placeholder(tf.float32, shape=[None, Dim])

D_W1 = tf.Variable(xavier_init([Dim, 256]), name = "D_W1")
D_b1 = tf.Variable(tf.zeros(shape=[256]), name = "D_b1")

D_W1_1 = tf.Variable(xavier_init([256, 128]), name = "D_W1_1")
D_b1_1 = tf.Variable(tf.zeros(shape=[128]), name = "D_b1_1")

D_W1_2 = tf.Variable(xavier_init([128, 128]), name = "D_W1_2")
D_b1_2 = tf.Variable(tf.zeros(shape=[128]), name = "D_b1_2")

D_W2 = tf.Variable(xavier_init([128, 1]), name = "D_W2")
D_b2 = tf.Variable(tf.zeros(shape=[1]), name = "D_b2")

theta_D = [D_W1, D_W1_1, D_W1_2, D_W2, D_b1, D_b1_1, D_b1_2, D_b2]


Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, 256]), name = "G_W1")
G_b1 = tf.Variable(tf.zeros(shape=[256]), name = "G_b1")

G_W1_1 = tf.Variable(xavier_init([256, 128]), name = "G_W1_1")
G_b1_1 = tf.Variable(tf.zeros(shape=[128]), name = "G_b1_1")

G_W1_2 = tf.Variable(xavier_init([128, 128]), name = "G_W1_2")
G_b1_2 = tf.Variable(tf.zeros(shape=[128]), name = "G_b1_2")

G_W2 = tf.Variable(xavier_init([128, Dim]), name = "G_W2")
G_b2 = tf.Variable(tf.zeros(shape=[Dim]), name = "G_b2")

theta_G = [G_W1, G_W1_1, G_W1_2, G_W2, G_b1, G_b1_1, G_b1_2, G_b2]


def sample_Z(m, n):
    return np.random.normal(size = [m, n])

num_sample = tf.constant(1, tf.int32)
temperature = 0.00001

def generator(z, temperature):
    G_h1 = tf.nn.leaky_relu(tf.matmul(z, G_W1) + G_b1)
    G_h1 = tf.nn.dropout(G_h1, rate = 0.5)
    G_h1_1 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W1_1) + G_b1_1)
    G_h1_1 = tf.nn.dropout(G_h1_1, rate = 0.5)
    G_h1_2 = tf.nn.leaky_relu(tf.matmul(G_h1_1, G_W1_2) + G_b1_2)
    G_h1_2 = tf.nn.dropout(G_h1_2, rate = 0.5)
    G_log_prob = tf.matmul(G_h1_2, G_W2) + G_b2
    
    G_age_logits = G_log_prob[:, 0:1]
    
    G_sex_logits = tf.concat([tf.zeros_like(G_log_prob[:, 0:1]), G_log_prob[:, 1:2]], 1)
    G_sex_binary_dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits = G_sex_logits)
    G_sex_binary = G_sex_binary_dist.sample()
    
    G_hispan_logits = G_log_prob[:, 2:27]
    G_hispan_multinom_dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits = G_hispan_logits)
    G_hispan_multinom = G_hispan_multinom_dist.sample()
    
    G_race_logits = G_log_prob[:, 27:38]
    G_race_multinom_dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits = G_race_logits)
    G_race_multinom = G_race_multinom_dist.sample()
    
    G_puma_logits = G_log_prob[:, 38:303]
    G_puma_multinom_dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits = G_puma_logits)
    G_puma_multinom = G_puma_multinom_dist.sample()
    
    G_prob = tf.concat([G_age_logits, G_sex_binary[:, 1:], G_hispan_multinom[:, 0:], G_race_multinom[:, 0:], G_puma_multinom[:, 0:]], 1)
    
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.leaky_relu(tf.matmul(x, D_W1) + D_b1)
    D_h1_1 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W1_1) + D_b1_1)
    D_h1_2 = tf.nn.leaky_relu(tf.matmul(D_h1_1, D_W1_2) + D_b1_2)
    D_logit = tf.matmul(D_h1_2, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    
    return D_prob, D_logit



G_sample = generator(Z, temperature)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)


D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real))
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake))
vector_D_loss = tf.concat([D_loss_real, D_loss_fake], 0)
scalar_D_loss = tf.reduce_mean(vector_D_loss)
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


D_optimizer = dp_optimizer.DPAdamGaussianOptimizer(
        learning_rate=D_learning_rate,
        num_microbatches=num_microbatches,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        ledger=False)

D_solver = D_optimizer.minimize(loss=vector_D_loss, var_list = theta_D, global_step=tf.compat.v1.train.get_global_step())

from tensorflow_privacy.privacy.analysis import rdp_accountant

def compute_epsilon(steps):
  orders = [1 + x /10.0 for x in range(1, 1200)]
  rdp = rdp_accountant.compute_rdp(q=mb_size/N,
                                  noise_multiplier=noise_multiplier,
                                  steps=steps,
                                  orders=orders)
  eps, _, _ = rdp_accountant.get_privacy_spent(orders=orders,
                                              rdp=rdp,
                                              target_delta=1/(2*N))
  return eps

G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

config = tf.compat.v1.ConfigProto(device_count = {'GPU': GPU})

for run in range(first_run, runs+1):
	sess = tf.Session(config = config)
	sess.run(tf.global_variables_initializer())

	if not os.path.exists('./models/run'+str(run)):
	    os.makedirs('./models/run'+str(run))

	i = 0
	saver = tf.train.Saver(tf.trainable_variables(), max_to_keep = 15000)

	steps_per_epoch = N // mb_size

	for epoch in range(1, epochs + 1):
	    for step in range(1, steps_per_epoch + 1):
	        
	        X_mb = data.sample(mb_size)
	        Z_mb = sample_Z(mb_size, Z_dim)
	        
	        _, D_loss_curr = sess.run([D_solver, vector_D_loss], feed_dict={X: X_mb, Z: Z_mb})
	        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_mb})
	        g_step = epoch*steps_per_epoch+step-steps_per_epoch
	        if g_step % 5 == 0:
	            saver.save(sess, './models/run'+str(run)+'/nmp-' + str(noise_multiplier) +'/pums2010', global_step = g_step)
	            print('Step: {}'.format(g_step))
	            D_loss_ep = np.mean(D_loss_curr)
	            print('D loss: {:.4}'. format(D_loss_ep))
	            print('G loss: {:.4}'.format(G_loss_curr))
	            print()
	    

	eps = compute_epsilon(steps_per_epoch*epochs)
	print('Epsilon after %d epochs is: %.3f' % (epochs, eps))
	sess.close()

