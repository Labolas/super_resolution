Fichier utils.py
----------------------
import keras
import numpy as np
import scipy.signal



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
		
		
Fichier qui permet de classifier une image et qui appelle le "fichier" ci-dessus
----------------------	
		
import cv2, numpy as np
import keras

import utils


def classify(model_name, image_name):
    image_shape = (224,224,3)

    image = load_image(image_name)
    model = utils.load_model(model_name, image_shape)

    predict(model, model_name, image)
    extract_features(model, image)


def load_image(image_name):
    # load image
    image = cv2.resize(cv2.imread(image_name), (224, 224)).astype(np.float32)
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
    utils.show_layers(model)


def extract_features(model, image):
    # Try to extract every layer features
    from keras import backend as K
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
    # Forward the network
    layer_outs = functor([image, 1.])

    for i,features in enumerate(layer_outs):
        try:
            features = features[0] # take the first image
            if features.shape[2] == 3:
                image = features[:,:,0:3] # take the RGB channels
                image[:,:,0] += 103.939
                image[:,:,1] += 116.779
                image[:,:,2] += 123.68
            else:
                image = features[:,:,0] # Take te first filter output
                image[:,:] *= 255.0 / np.max(image)  # change dynamic to [0,255]
            image = image.astype(np.int)
            cv2.imwrite("data/out/%s.png" % model.layers[i].name, image)
        except IndexError as e:
            # Occurs when layer isn't an image (fully connected layers...)
            # print("%s : %s" % (model.layers[i].name, e))
            pass



def main():
    # model_name = 'resnet'
    model_name = 'vgg16'
    # model_name = 'vgg19'

    # image_name = 'data/lena.png'
    image_name = 'data/elephant.jpg'
    # image_name = 'data/desk.jpg'

    # run classification
    classify(model_name, image_name)




if __name__ == "__main__":
    main()
		
		
		

		