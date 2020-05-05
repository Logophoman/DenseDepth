import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def image_scaling(array):
    copy = np.copy(array)
    min_val = np.min(outputs)
    max_val = np.max(outputs)
    diff = max_val - min_val

    for index_row, row in enumerate(array):
        for index_col, value in enumerate(row):
            copy[index_row,index_col] = (value - min_val) / diff

    return copy


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='scaledImages/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
print(outputs.shape)

for i in range (outputs.shape[0]):
    print(np.min(outputs[i,:,:,0]))
    print(np.max(outputs[i,:,:,0]))
    # scaled_image = image_scaling(outputs[i,:,:,0])
    # print(np.min(scaled_image))
    # print(np.max(scaled_image))
    image = Image.fromarray((outputs[i,:,:,0] * (pow(2, 16) - 1)).astype(np.uint16))
    image.save('result/testresult' + str(i) + '.png', format='PNG')

# matplotlib problem on ubuntu terminal fix
# matplotlib.use('TkAgg')

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
