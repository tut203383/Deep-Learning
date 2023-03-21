import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from BoF_layers import BoF_Pooling
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input
from sklearn.model_selection import KFold


data_df = pd.read_csv('./Target.csv', delimiter=',', header=None, names=["id","r1","g1","r2","g2","r3","g3","r4","g4","r5","g5","r6","g6","r7","g7","r8","g8","r9","g9","r10","g10","r11","g11","r12","g12","r13","g13","r14","g14","r15","g15","r16","g16","r17","g17","r18","g18","r19","g19"])

datagen = ImageDataGenerator(rescale=1./255, rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.5,
                             horizontal_flip=True,
                             vertical_flip=True)


valid_datagen = ImageDataGenerator(rescale=1./255)

def build_masked_loss():#the saturated pixels (label=-1) are removed from the loss evaluation
    def custom_rmse(y_true, y_pred,mask=-1):
        mask = K.cast(K.not_equal(y_true, mask), K.floatx())
        return K.sqrt(K.mean(K.square(mask*(y_pred - y_true)), axis=-1)+0.000001) # adding this epsilon to avoid Nan when the gray output is saturated
    return custom_rmse

def build_model():
    n_codewords = 150
    num_classes = 2
    input = Input(shape=[224, 224, 3])
    x = Conv2D(60, (3, 3), activation='relu')(input)
    x = MaxPooling2D(8, 8)(x)
    x = Conv2D(30, (3, 3), activation='relu')(x)
    x = MaxPooling2D(4, 4)(x)
    x = Conv2D(30, (3, 3), activation='relu')(x)
    x = BoF_Pooling(n_codewords, spatial_level=0)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    preds = Dense(num_classes, activation='sigmoid')(x)
    model = Model(input, preds)
    return model

img_width=224
img_height=224
bs=8
#opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0009, decay=0.0001)
opt = tf.keras.optimizers.Adam(lr=0.0003)
EPOCHS=3000

kf = KFold(n_splits=3, random_state=42, shuffle=True)
X = np.array(data_df["id"])
i = 1
for train_index, test_index in kf.split(X):
    trainData = X[train_index]
    testData = X[test_index]
    train_df = data_df.loc[data_df["id"].isin(list(trainData))]
    val_df = data_df.loc[data_df["id"].isin(list(testData))]
    model = build_model()
    model.compile(loss=build_masked_loss(), optimizer=opt)

    train_generator = datagen.flow_from_dataframe(dataframe=train_df, directory='./IMGs',
                                              x_col="id", y_col=["r19","g19"], has_ext=True,
                                              class_mode="raw", target_size=(img_width, img_height), shuffle=True,
                                              batch_size=bs, color_mode='rgb')

    validation_generator = valid_datagen.flow_from_dataframe(dataframe=val_df, directory='./IMGs',
                                                         x_col="id", y_col=["r19","g19"], has_ext=True,
                                                         class_mode="raw", target_size=(img_width, img_height), shuffle=False,
                                                         batch_size=bs, color_mode='rgb')

    H = model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS, verbose=2)
    Ypred = model.predict(validation_generator)
    pred_df = pd.DataFrame(Ypred)
    with open('Pred_grayCV2_'+str(i)+'.csv', mode='w') as f:
        pred_df.to_csv(f)
    with open('Test_grayCV2_'+str(i)+'.csv', mode='w') as f:
        val_df.to_csv(f)
    model.save('C:\Users\yong21\Documents\IMLEX\UJM\DL & CV\Project\code'+str(i))
    i += 1

    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training and Validation losses")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.pause(1)

