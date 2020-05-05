# simple slicing
from numpy import array
import numpy as np
from PIL import Image
# define array
data = array([[[[0.1],[0.9]], [[0.5],[0.5]], [[0.4],[0.8]]],
              [[[0.5],[0.5]], [[0.5],[0.5]], [[0.5],[0.5]]]])
print(data.shape)

for i in range (data.shape[0]):
    data2 = data[i,:,:,0]
    print(data2.shape)
    image = Image.fromarray((data2 * 255).astype(np.uint8))
    image.save('testresult' + str(i) + '.png', format='PNG')

print(Image.open('C:/Users/Andreas/Documents/GitRepositories/DenseDepth/scaled_depth/testresult0resized.png'))