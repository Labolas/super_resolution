#Fichier utils.py
#----------------------

import sys
sys.path.append('/your/dir/to/tensorflow/models') # point to your tensorflow dir
sys.path.append('/your/dir/to/tensorflow/models/slim') # point ot your slim dir

import keras
import numpy as np
import scipy.signal
from skimage import io, exposure, img_as_uint, img_as_float

def save_feature(path, feature):
    #io.use_plugin('libfreeimage3')
    im = np.array(feature, dtype='float64')
    im = exposure.rescale_intensity(im, out_range='float')
    im = img_as_uint(im)
    io.imsave(path, im)

def load_model(model_name, image_shape, include_top=True):
    # Load wanted model (resnet, vgg16 or vgg19)

    # set params
    weights = 'imagenet'
    model = None

    if model_name == 'resnet':
        model = keras.applications.resnet50.ResNet50(
            include_top=include_top,
            weights=weights,
            input_shape=image_shape
        )
    elif model_name == 'vgg16':
        model = keras.applications.vgg16.VGG16(
            include_top=include_top,
            weights=weights,
            input_shape=image_shape
        )
    elif model_name == 'vgg19':
        model = keras.applications.vgg19.VGG19(
            include_top=include_top,
            weights=weights,
            input_shape=image_shape
        )
    else:
        print("Model name is unknown")
    return model


def show_layers(model):
    # Show all layers of model
    print("Layers :")
    for layer in model.layers:
        print("  %s\t--> %s" % (layer.name, layer.output.shape))


#Fichier qui permet de classifier une image et qui appelle le "fichier" ci-dessus
#----------------------

import cv2, numpy as np
import keras

#import utils


def classify(model_name, image_name):
    image_shape = (425,282,3)

    image = load_image(image_name)
    model = load_model(model_name, image_shape, False)

    predict(model, model_name, image)
    extract_features(model, image)


def load_image(image_name, image_shape=None):
    # load image
    # image = image_name
    image = cv2.imread(image_name).astype(np.float32)
    if image_shape != None:
        image = cv2.resize(image, image_shape)
    # Remove train image mean (imagenet)
    image[:,:,0] -= 103.939
    image[:,:,1] -= 116.779
    image[:,:,2] -= 123.68
    image = np.expand_dims(image, axis=0)
    return image


def predict(model, model_name, image):
    out = model.predict(image)

    print("Model : %s" % model_name)
    print("Class : %s" % np.argmax(out))
    print("Probability : %s" % out[0][np.argmax(out)])
    show_layers(model)


def extract_features(model, output_folder, image, n):
    # Try to extract every layer features
    from keras import backend as K
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+[K.learning_phase()], outputs )  # evaluation function
    # Forward the network
    layer_outs = functor([image, 1.])
    print(len(layer_outs))
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(output_folder)

    for i,features in enumerate(layer_outs[len(layer_outs)-10:]): #select 2 last layers
        layer_output_folder = output_folder+str(len(layer_outs)-10+i)+"/"
        if not os.path.exists(layer_output_folder):
            os.makedirs(layer_output_folder)
        print(layer_output_folder)
        try:
            #for features in layer:
            features = features[0] # take the first image
            print(features.shape)
            f_range = range(0, features.shape[2])
            if features.shape[2] == 3:
                for j in f_range: #iterate on the features of the layer
                    image = features[:,:,j:3] # take the RGB channels
                    image[:,:,0] += 103.939
                    image[:,:,1] += 116.779
                    image[:,:,2] += 123.68
                    image = image.astype(np.int)
                    print(layer_output_folder+"%s_%s%s.png" % (n,model.layers[i].name,j))
                    cv2.imwrite(layer_output_folder+"%s_%s.png" % (model.layers[i].name,j), image)
            else:
                #nbCanaux = len(features[0,0])
                #i = 0
                #while i < nbCanaux:
                ma, mi = 0, 200

                for j in f_range: #iterate on the features of the layer
                    image = features[:,:,j] # Take the first filter output
                    if True or np.max(image) != 0: # if empty, drop feature
                        ma, mi = max(np.max(image), ma), min(np.min(image), mi)
                        #np.savetxt(layer_output_folder+"%s_%s_%s.csv" % (i+1, model.layers[i].name,j), image)
                        save_feature(layer_output_folder+"%s_%s_%s.png" % (i+1, model.layers[i].name,j), image)
                        #image[:,:] *= 255.0 / max(np.max(image),1)  # change dynamic to [0,255]
                        #image = image.astype(np.int)
                        #print(layer_output_folder+"%s_%s_%s.png" % (i+1, model.layers[i].name,j))
                        #cv2.imwrite(layer_output_folder+"%s_%s_%s.png" % (i+1, model.layers[i].name,j), image)
                print(ma, mi)
                #i+=1
        except IndexError as e:
            # Occurs when layer isn't an image (fully connected layers...)
            # print("%s : %s" % (model.layers[i].name, e))
            pass



def main():
    # model_name = 'resnet'
    model_name = 'vgg16'
    #model_name = 'vgg19'


    # image_name = 'data/lena.png'
    #image_name = 'data/elephant.jpg'
    #image_name = '../data/chaton-gris.jpg'
    # image_name = 'data/desk.jpg'

    # run classification
    #classify(model_name, image_name)


    images_folder_path = "data/img/"
    output_folder = "data/out/"+model_name+"/"

    #get all the image names (whose feature folder does not already exist)
    from os import walk
    from os import path
    folders = []
    for (dir_path, dir_names, _) in walk(output_folder):
        folders.extend(dir_names)
        break
    print(len(folders),"features folders found")
    f = []
    for (dir_path, dir_names, files_names) in walk(images_folder_path):
        for f_n in files_names:
            fn = path.splitext(f_n)[0] #remove extension
            if not(fn in folders):
                print(fn)
                f.append(f_n)
        break
    files_names = f
    print(len(files_names),"images found")
    if len(files_names) == 0:
        exit(0)

    #load the images
    images = []
    for file_name in files_names:
        image_name = images_folder_path+file_name
        image = load_image(image_name)
        images.append(image)
    print(len(images),"images loaded")

    #load the model
    model = load_model(model_name, images[0].shape[1:4], False)
    print("model loaded")

    #extract the features and save them
    from os import path
    for i in range(len(images)):
        image = images[i]
        image_name = path.splitext(files_names[i])[0] #remove the extension
        extract_features(model, output_folder+image_name+"/", image, 0)
    print("features extracted")


if __name__ == "__main__":
    main()
