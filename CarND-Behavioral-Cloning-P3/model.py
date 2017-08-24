import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
import cv2
from keras.preprocessing.image import load_img
from keras.regularizers import l2

def load_data(log_file):
    """
    read the data use pd library.
    shuffle the data and split it into train and validation
    
    """
    data_frame = pd.read_csv(log_file, usecols = [0, 1, 2, 3])
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    
    ## split the data 
    rows = int(data_frame.shape[0] * 0.8)
    training = data_frame.loc[0:rows - 1]
    validation = data_frame.loc[rows:]
    
    return training, validation

def img_brightness(image):
    """
    Uniform darker the image to reduce the V channel 
    
    """
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img_bright = .25 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * img_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img

def img_resize(image):
    """
    resize the image to (64,64)

    """
    return cv2.resize(image, (64, 64))

def img_crop(image):
    """
    crop the image with to remove the unnecessary feature

    """
    cropped_image = image[55:135, :, :]
    processed_image = img_resize(cropped_image)
    return processed_image

def preprocess_image(image):
    """
    crop the image
    convert to np.float32
    normalized the image

    """
    image = img_crop(image)
    image = image.astype(np.float32)
    image = image/255.0 - 0.5

    return image



def augmented_data(row):
    """
    read the data in each row:
    set the angle offset value for left and right image

    """
    angle = row['steering']
    img_choice = np.random.choice(['center', 'left', 'right'])
    
    if img_choice == 'left':
        angle += 0.25
    if img_choice == 'right':
        angle -= 0.25

    ## load and preproces the data                         
    image = load_img("data/" + row[img_choice].strip())
    image = np.array(image, dtype=np.float32)
    image = img_brightness(image)
    image = preprocess_image(image)
    
    ## flip the image with 50% chance 

    if np.random.random() > 0.5:
        angle = -1 * angle
        image = cv2.flip(image, 1)
                              
    return image, angle


def generator(data_frame, batch_size=32):
    """
    number of the batches: 
    generate the data in each baches 
    regenerated again 
    """
    num_batch = data_frame.shape[0] // batch_size
    
    while(True):
        for n in range(num_batch):
            start = n * batch_size
            end = start + batch_size - 1
            
            X_data = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
            y_data = np.zeros((batch_size,), dtype=np.float32)
            
            i = 0
            for index, row in data_frame.loc[start:end].iterrows():
                X_data[i], y_data[i] = augmented_data(row)
                i += 1
                
            yield X_data, y_data


def get_model():
    #Nivida model
    model = Sequential([
        Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), input_shape=(64, 64, 3)),
        
        Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2)),
        Dropout(.2),
        
        Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2)),
        Dropout(.2),
        
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1)),
        Dropout(.2),
        
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1)),
        # Flatten
        Flatten(),
        
        Dense(100, activation='elu', W_regularizer=l2(0.001)),
        Dropout(.5),
        Dense(50, activation='elu', W_regularizer=l2(0.001)),
        Dropout(.5),
        Dense(10, activation='elu', W_regularizer=l2(0.001)),
        Dropout(.5),
        
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    return model

if __name__ == "__main__":
    batch_size = 32
    training_data, validation_data = load_data('data/driving_log.csv')
    
    training_generator = generator(training_data, batch_size)
    validation_generator = generator(validation_data, batch_size)

    model = get_model()
    model.summary()
    
    model.fit_generator(training_generator, 
                        validation_data = validation_generator,
                        samples_per_epoch = (20000//batch_size) * batch_size, 
                        nb_epoch = 5, 
                        nb_val_samples = 3000)

    print("Model saved ! ")
    model.save('model.h5')