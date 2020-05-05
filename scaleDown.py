from PIL import Image
import os, sys

path = "C:/Users/Andreas/Documents/GitRepositories/DenseDepth/result/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            print("asd")
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((640,480), Image.ANTIALIAS)
            imResize.save(f + 'resized.png', 'PNG', quality=90)
print(dirs)
resize()