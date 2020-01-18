#
# Neural style transfer
# References:
# [1] https://www.virtosuart.com/blog/abstract-art
# [2] https://www.cs.toronto.edu/~frossard/post/vgg16/
# [3] https://www.tensorflow.org/tutorials/generative/style_transfer
#

import tensorflow as tf
import numpy as np
import scipy.io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

def imshow(image1, image2 = None):
    """
    Show one or two images

    Arguments:
    image1      -- first image to show
    image2      -- second image to show, if ignore show one image only

    Returns: None
    """
    plt.close('all')

    if image2 is not None:
        plt.subplot(1, 2, 1)

    plt.imshow(image1)
    plt.axis('off')

    if image2 is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(image2)
        plt.axis('off')
    
    plt.show()

# DEBUG
#content_path = "Yellow-Red-Blue640.jpg"
#style_path = "Pablo-Picassoa640.jpg"
#content_image = mpimg.imread(content_path)
#style_image = mpimg.imread(style_path)
#imshow(content_image)
#imshow(content_image, style_image)

# implementation outline
def load_vgg_weights(model_weights_file):
    """
    Load pretrained VGG weights from file. Apply to load vgg weights from ref [2]

    Arguments:
    model_weights_file  -- path to model weights file

    Return:
    vgg_weights         -- weights data in dictionary format
    """
    weights = np.load(model_weights_file)
    vgg_weights = sorted(weights.items())

    for i, (k, w) in enumerate(vgg_weights):
        print("{} : {} - {} - {}".format(i, k, w.shape, w.dtype))
    return vgg_weights

# DEBUG
#vgg16_weights = "vgg16_weights.npz"
#load_vgg_weights(vgg16_weights)

def load_mat_file(mat_model_file):
    model = scipy.io.loadmat(mat_model_file)
    for i, (k, w) in enumerate(model.items()):
        print("{}".format(k))
    return model

# DEBUG
#vgg19_model_file = 'imagenet-vgg-verydeep-19.mat'
#vgg19_model = load_mat_file(vgg19_model_file)

def generate_output(input_image):
    """
    Randomly generated output image has same size as input image

    Arguments:
    input_image     -- input image

    Returns:
    random_image    -- output image with random generated value pixels
    """
    random_image = np.random.randint(255, size=input_image.shape)
    return random_image

# DEBUG
#imshow(style_image, generate_output(style_image))

def compute_content_cost(a_C, a_G):
    """
    Compute content cost

    Arguments:
    a_C         -- a tensor shape (m, n_H, n_W, n_C) hidden layer activation of content image
    a_G         -- a tensor shape (m, n_H, n_W, n_C) hidden layer activation of generated image

    Returns:
    J_content   -- content cost
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))
    J_content = J_content / (4 * n_H * n_W * n_C)

    return J_content

def gram_matrix(A):
    """
    Arguments:
    A           -- (n_C, n_H * n_W)

    Returns:
    GA          -- Gram matrix of A, size = (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    # numpy implement
    # GA = np.multiply(A, np.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Compute style cost (style lost) from a chosen style layer

    Arguments:
    a_S     -- shape (m, n_H, n_W, n_C) hidden layer activation of style image
    a_G     -- shape (m, n_H, n_W, n_C) hidden layer activation of generated image

    Returns:
    J_style_cost    -- Scalar value
    """
    m, n_H, n_W, n_C = a_S.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, shape = [-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape = [-1, n_C]))
    assert(a_S.shape == (n_C, n_H * n_W))

    Ga_S = gram_matrix(a_S)
    Ga_G = gram_matrix(a_G)
    assert(Ga_S.shape == (n_C, n_C))

    J_layer_stype = tf.reduce_sum(tf.square(tf.subtract(Ga_S, Ga_G)))
    J_layer_style = J_layer_style / np.square(2 * n_C * n_H * n_W)

    return J_layer_style

STYLE_LAYERS = [        # (layer_name, weight)
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Compute total style cost (style loss) from all chosen layers

    Arguments:
    model       -- full model
    STYLE_LAYER -- chosen style layers in list format which contains layer names and their weights (coefficients)

    Returns:
    J_style     -- Scalar value repesents total style cost
    """
    J_style = 0

    for layer_name, weight in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += weight * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 30):
    """
    Computes the total cost function
    
    Arguments:
    J_content   -- content cost
    J_style     -- style cost
    alpha       -- importance weight of content cost
    beta        -- importance weight of style cost
    
    Returns:
    J           -- total cost as defined by (J = J_content * alpha + J_style * beta)
    """
    J = J_content * alpha + J_style * beta
    return J

# STEPS (whole picture)
"""
1. create a Interactive Session
2. load content image
3. load style image
4. randomly initialize being generated image
5. load vgg19 model
6. build Tensorflow graph
  + run content image through vgg19 model to compute content cost
  + run style image through vgg19 model to compute style cost
  + compute total cost
  + define optimizer and learning rate
7. init Tensorflow graph and run it for a huge number of iterations
  + update the being generated image every step
"""

# step 1:
#tf.reset_default_graph()

#sess = tf.InteractiveSession()


