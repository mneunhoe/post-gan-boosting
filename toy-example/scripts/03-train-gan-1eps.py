import tensorflow as tf
import numpy as np
import os
import time
from tensorflow_privacy.privacy.optimizers import dp_optimizer

"""Set hyperparameters"""

# How many times do you want to run the experiment
runs = 1
# What's the first run (to resume later or add runs later)
first_run = 1

# If you have a GPU you can set this to 1. Training will be run on the GPU
GPU = 0

# Path to the gan-input file
data_path = "gan-input/gaussian_df_z.csv"

data = np.genfromtxt(data_path, delimiter=',', skip_header = 1)

# Set whether you want to train a DP-GAN or vanilla GAN
dp = True

# Get dimensions of data set
N = np.shape(data)[0]
Dim = np.shape(data)[1]

# Set minibatch size (this will influence your epsilon under DP)
mb_size = 50

# Set the Discriminator Learning Rate
if dp:
  D_learning_rate = 0.01
else:
  D_learning_rate = 0.001

# Set the Generator learning rate
G_learning_rate = 0.001

# Set the number of "epochs": epochs* (N // mb_size) = number of update steps
epochs = 38

# Set the dimension of the noise vector Z
Z_dim = 32

# The number of neurons in the hidden layers
n_hidden = 128


# DP hyperparameters

# DP-Adam
num_microbatches = mb_size
l2_norm_clip = 1
noise_multiplier = 1.5

# Function for xavier initialization of the weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Function to sample noise for the input to the generator
def sample_Z(m, n):
    return np.random.normal(size = [m, n])


"""Set up the GAN architecture"""


clip_val = tf.placeholder(tf.float32, shape=[])

# An empty tensor to feed in real data
X = tf.placeholder(tf.float32, shape=[None, Dim])

# The weights for the discriminator layers
D_W1 = tf.Variable(xavier_init([Dim, n_hidden]), name = "D_W1")
D_b1 = tf.Variable(tf.zeros(shape=[n_hidden]), name = "D_b1")

D_W1_1 = tf.Variable(xavier_init([n_hidden, n_hidden]), name = "D_W1_1")
D_b1_1 = tf.Variable(tf.zeros(shape=[n_hidden]), name = "D_b1_1")

D_W1_2 = tf.Variable(xavier_init([n_hidden, n_hidden]), name = "D_W1_2")
D_b1_2 = tf.Variable(tf.zeros(shape=[n_hidden]), name = "D_b1_2")

D_W2 = tf.Variable(xavier_init([n_hidden, 1]), name = "D_W2")
D_b2 = tf.Variable(tf.zeros(shape=[1]), name = "D_b2")

theta_D = [D_W1, D_W1_1,D_W1_2, D_W2, D_b1, D_b1_1, D_b1_2, D_b2]

# Define the discriminator network
def discriminator(x):
    D_h1 = tf.nn.leaky_relu(tf.matmul(x, D_W1) + D_b1)
    D_h1_1 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W1_1) + D_b1_1)
    D_h1_2 = tf.nn.leaky_relu(tf.matmul(D_h1_1, D_W1_2) + D_b1_2)
    
    D_logit = tf.matmul(D_h1_2, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

# An empty tensor to feed in noise draws as the generator input
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

# The weights for the generator layers
G_W1 = tf.Variable(xavier_init([Z_dim, n_hidden]), name = "G_W1")
G_b1 = tf.Variable(tf.zeros(shape=[n_hidden]), name = "G_b1")

G_W1_1 = tf.Variable(xavier_init([n_hidden, n_hidden]), name = "G_W1_1")
G_b1_1 = tf.Variable(tf.zeros(shape=[n_hidden]), name = "G_b1_1")

G_W1_2 = tf.Variable(xavier_init([n_hidden, n_hidden]), name = "G_W1_2")
G_b1_2 = tf.Variable(tf.zeros(shape=[n_hidden]), name = "G_b1_2")

G_W2 = tf.Variable(xavier_init([n_hidden, Dim]), name = "G_W2")
G_b2 = tf.Variable(tf.zeros(shape=[Dim]), name = "G_b2")

theta_G = [G_W1, G_W1_1,G_W1_2, G_W2, G_b1, G_b1_1, G_b1_2, G_b2]

# Define the generator network
def generator(z):
    G_h1 = tf.nn.leaky_relu(tf.matmul(z, G_W1) + G_b1)
    G_h1 = tf.nn.dropout(G_h1, rate = 0.5)
    G_h1_1 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W1_1) + G_b1_1)
    G_h1_1 = tf.nn.dropout(G_h1_1, rate = 0.5)
    G_h1_2 = tf.nn.leaky_relu(tf.matmul(G_h1_1, G_W1_2) + G_b1_2)
    G_h1_2 = tf.nn.dropout(G_h1_2, rate = 0.5)
    G_log_prob = tf.matmul(G_h1_2, G_W2) + G_b2

    G_prob = G_log_prob

    return G_prob

# Define tensors for loss calculation
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# Discriminator losses
D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real))
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake))

# Generator loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Define optimizers
if dp == True:
    D_loss = tf.concat([D_loss_real, D_loss_fake], 0)
    D_optimizer = dp_optimizer.DPAdamGaussianOptimizer(
            learning_rate=D_learning_rate,
            num_microbatches=num_microbatches*2,
            l2_norm_clip=clip_val,
            noise_multiplier=noise_multiplier,
            ledger=False)
            
    

    D_solver = D_optimizer.minimize(loss=D_loss, var_list = theta_D, global_step=tf.train.get_global_step())
    
    
    from tensorflow_privacy.privacy.analysis import rdp_accountant

    def compute_epsilon(epochs = epochs, mb_size = mb_size, N = N, noise_multiplier = noise_multiplier):
      orders = [1 + x /10.0 for x in range(1, 800)]
      steps = (N/mb_size) * epochs
      rdp = rdp_accountant.compute_rdp(q=mb_size/N,
                                      noise_multiplier=noise_multiplier,
                                      steps=steps,
                                      orders=orders)
      eps, _, _ = rdp_accountant.get_privacy_spent(orders=orders,
                                                  rdp=rdp,
                                                  target_delta=1/(2*N))
      return eps

if dp == False:
    vector_D_loss = D_loss_real + D_loss_fake
    D_loss = tf.reduce_mean(vector_D_loss)
    D_solver = tf.train.AdamOptimizer(learning_rate = D_learning_rate).minimize(D_loss, var_list=theta_D)
    
    
G_solver = tf.train.AdamOptimizer(learning_rate = G_learning_rate).minimize(G_loss, var_list=theta_G)



# Initialize tf if an GPU is available
config = tf.ConfigProto(device_count = {'GPU': GPU})

# Start Experiment loop
for run in (range(first_run, runs + 1)):
    # Initialize tf session
    sess = tf.Session(config = config)
    # Initialize GAN weights
    sess.run(tf.global_variables_initializer())
    
    # Create a directory to store intermediate weights. Needed for PGB.
    if not os.path.exists('./models/run'+str(run)+'/'):
        os.makedirs('./models/run'+str(run)+'/')
    
    # Create tf saver. How many collections of weights do you want to keep?
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep = 4000)
    
    # How many update steps does one "epoch" have
    steps_per_epoch = N // mb_size
    
    # Start Training loop
    for epoch in (range(1, epochs + 1)):
        for step in range(1, steps_per_epoch + 1):
            # Randomly sample minibatch from data
            ind = np.random.permutation(np.shape(data)[0])
            X_mb = data[ind[0:mb_size],:]
            # Create noise for generator input
            Z_mb = sample_Z(mb_size, Z_dim)
            
            # Update Discriminator
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_mb, clip_val: l2_norm_clip})
            # Update Generator
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_mb})
            
            # Save weights after every fifth update step
            if step % 5 == 0:
                if dp == True:
                    saver.save(sess, './models/run' + str(run) + '/nmp-' + str(noise_multiplier) + '-mb-' + str(mb_size) + '-l2clip-' + str(l2_norm_clip) + '/toy-example', global_step = epoch*steps_per_epoch+step-steps_per_epoch)
                if dp == False:
                    saver.save(sess, './models/run' + str(run) + '/nodp/toy-example', global_step = epoch*steps_per_epoch+step-steps_per_epoch)
        # Print training progress
        D_loss_ep = np.mean(D_loss_curr)
        print('Epoch: {}'.format(epoch))
        print('D loss: {:.4}'. format(D_loss_ep))
        print('G loss: {:.4}'.format(G_loss_curr))
        print()
    
    # Calculate and print final epsilon if DP    
    if dp == True:
            eps = compute_epsilon(epochs = epochs, mb_size = mb_size, N = N, noise_multiplier = noise_multiplier)
            print('Epsilon after %d epochs is: %.3f' % (epoch, eps))
          
    sess.close()
