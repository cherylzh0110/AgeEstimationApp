import keras
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout, GlobalMaxPooling2D
from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.regularizers import l2

img_height,img_width = 119,119
target_size=(img_width,img_height)
classnumber = 101
def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def get_model(model_name="ResNet"):
    base_model = None
    # ResNet for 101 classes classification
    if model_name == "ResNet":
        base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (img_height,img_width,3), pooling="avg")
        x = base_model.output
        predictions = Dense(units=classnumber, kernel_initializer="he_normal",use_bias=False,
                              kernel_regularizer=l2(0.0005), activation="softmax",
                              name="pred_age")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3), classes = classnumber)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(units=classnumber, activation='softmax'))
        model = Sequential()
        model.add(base_model)
        model.add(top_model)
    # ResNet for 8 classes classification (architecture after FC layers tuning in res8train.py)
    if model_name == "ResNet50":
        base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (img_height,img_width,3), pooling="avg")
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(8, activation= 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = predictions)

    return model

def main():
    model = get_model("ResNet")
    model.summary()

if __name__ == '__main__':
    main()
