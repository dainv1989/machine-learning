#
# Neural style transfer
# References:
# [1] https://www.virtosuart.com/blog/abstract-art
# [2] https://www.cs.toronto.edu/~frossard/post/vgg16/
# [3] https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee
#

import tensorflow as tf
import numpy as np
import scipy.io
from collections import OrderedDict
import re               # regular expression

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
def load_weights(model_weights_file, end_layer = 'all'):
    """
    Load pretrained VGG weights from file. Apply to load vgg weights from ref [2]

    Arguments:
    model_weights_file  -- path to model weights file

    Return:
    vgg_weights         -- weights data in dictionary format
    """
    layers = OrderedDict()
    weights = np.load(model_weights_file)
    vgg_weights = sorted(weights.items())

    layer_count = 0
    for i, (k, w) in enumerate(vgg_weights):
        if (end_layer != 'all') and (layer_count >= int(end_layer)):
            break

        # k = conv1_2_W --> k[:-2] = conv1_2
        if k[:-2] not in layers:
            layers[k[:-2]] = {}

        if re.search(r'conv\d+_\d+_W', k) is not None or re.search(r'fc\d+_W', k) is not None:
            layers[k[:-2]]['weights'] = w
        if re.search(r'conv\d+_\d+_b', k) is not None or re.search(r'fc\d+_b', k) is not None:
            layers[k[:-2]]['bias'] = w

        print("loaded layer {}: {} - {}".format(layer_count, k, w.shape))
        if (k[:-2] in layers) and ('bias' in layers[k[:-2]]):
            layer_count += 1
    return vgg_weights

# DEBUG
vgg16_weights = "vgg16_weights.npz"
load_weights(vgg16_weights, 7)

def vgg_nst(input_image, layer_ids, pool_ids, vgg_layers):
    outputs = OrderedDict()
    out = input_image

    for l_id in layer_ids:
        with tf.variable_scope(l_id, reuse = tf.AUTO_REUSE):
            print("Computing output at layer {}".format(l_id))
            w, b = tf.get_variable('weights'), tf.get_variable('bias')
            out = tf.nn.conv2d(filter=w, input=out, strides=[1, 1, 1, 1], padding='SAME')
            out = tf.nn.relu(tf.nn.bias_add(value=out, bias=b))
            outputs["conv" + l_id] = out

        if l_id in pool_ids:
            out = tf.nn.max_pool(input=out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            outputs["pool" + l_id] = out
    return outputs

def initalize_input(input_shape):
    content = tf.placeholder(name='content', shape=input_shape, dtype=tf.float32)
    style = tf.placeholder(name='style', shape=input_shape, dtype=tf.float32)
    generated = tf.placeholder(name='generated', shape=input_shape, dtype=tf.float32, 
                               initializer=tf.random_normal_initializer, trainable=True)
    return {'content': content, 'style': style, 'generated': generated}

def define_tf_weights(vgg_layers):
    for k, wb in vgg_layers.items():
        w, b = wb['weights'], wb['bias']
        with tf.variable_scope(k):
            tf.get_variable(name='weights', initializer=tf.constant(w, dtype=tf.float32), trainable=False)
            tf.get_variable(name='bias', initializer=tf.constant(b, dtype=tf.float32), trainable=False)

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


