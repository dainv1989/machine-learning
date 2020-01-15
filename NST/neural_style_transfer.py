import tensorflow as tf
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# style image source:
# https://www.virtosuart.com/blog/abstract-art
# VGG16 pre-trained weights:
# https://www.cs.toronto.edu/~frossard/post/vgg16/

content_path = "Yellow-Red-Blue640.jpg"
style_path = "Pablo-Picassoa640.jpg"

content_image = mpimg.imread(content_path)
style_image = mpimg.imread(style_path)

plt.subplot(1, 2, 1)
plt.imshow(content_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(style_image)
plt.axis('off')

#plt.show()

vgg16_weights = "vgg16_weights.npz"

# implementation outline
def load_vgg_weights(file_path):
    weights = np.load(vgg16_weights)
    vgg_weights = sorted(weights.items())

    for i, (k, w) in enumerate(vgg_weights):
        print("{} : {} - {} - {}".format(i, k, w.shape, w.dtype))
    return vgg_weights

#DEBUG: load_vgg_weights(vgg16_weights)

def compute_content_cost():
    return J_content

def gram_matrix(A):
    return GA

def compute_layer_style_cost():
    return J_layer_style

def compute_style_cost():
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
    return J

# DEBUG

