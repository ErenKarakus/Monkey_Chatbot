import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from os import environ
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import itertools as it
from keras.constraints import maxnorm

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_cnn():
    output_classes = 2
    classNames = ['Monkey', 'Not_Monkey']

    # Define the data directories
    train_dir = "monkeyDataset/data/train"
    validation_dir = "monkeyDataset/data/val"
    test_dir = "monkeyDataset/data/test"

    # Define the image size and batch size
    img_size = (256, 256)
    batch_size = 32

    # Define the data generators
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()  

   # Create the data generators
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
        classes= classNames,
        shuffle=True,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
        classes= classNames,
        shuffle=True,
        seed=42
        
    )
        
    validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
        classes= classNames,
        shuffle=True,
        seed=42
    )

    
    def hyperparameter_tuning(train_generator, validation_generator):
        params_nn: dict = {
            'neurons': [32, 64, 128],
            'activation': ['relu', 'sigmoid'],
            'batch_size': [32, 64],
            'epochs': [5, 10]
        }

        all_names = sorted(params_nn)
        combinations = list(it.product(*(params_nn[Name] for Name in all_names)))
        print(combinations)
        print("Length = ", len(combinations))

        for index, x in enumerate(combinations):
            print(f"\nIndex: {index}\n Combo: {x}")
            activation = x[0]
            batch_size = x[1]
            epochs = x[2]
            neurons = x[3]


    # Define the CNN architecture
    # model = keras.Sequential(
    #     [
    #         keras.Input(shape=(256, 256, 3)),
    #         keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
    #         keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #         keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         keras.layers.Flatten(),
    #         keras.layers.Dropout(0.5),
    #         keras.layers.Dense(256, activation="relu"),
    #         #keras.layers.Dense(1, activation="sigmoid"),
    #         keras.layers.Dense(output_classes, activation="softmax")
    #     ]
    # )

    #Hyperparameter tuning
    hyperparameter_tuning(train_generator, validation_generator)
    model = keras.Sequential(
        [
            keras.Input(shape=(256, 256, 3)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            # keras.layers.Dense(64, activation="relu", kernel_constraint=maxnorm(3)),  # 64 neurons
            keras.layers.Dense(128, activation="relu", kernel_constraint=maxnorm(3)),  # 128 neurons
            keras.layers.Dropout(0.2),
            keras.layers.Dense(output_classes, activation="sigmoid")  # 2 neurons
        ]
    )

    
    stepSizeTrain = train_generator.n // train_generator.batch_size
    stepSizeValid = validation_generator.n // validation_generator.batch_size
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=stepSizeTrain,
              validation_data=validation_generator,
              validation_steps=stepSizeValid,
              epochs=10)
    model.evaluate(x=train_generator,
                   steps=stepSizeValid)
    
    # Get the true and predicted labels for the validation set
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    true_labels = test_generator.classes

    # Print the classification report and confusion matrix
    print(classification_report(true_labels, y_pred))
    print(confusion_matrix(true_labels, y_pred))

    # Evaluate the trained CNN model
    test_loss, test_acc = model.evaluate(test_generator)
    print('\n Test Accuracy:', test_acc)

    # Save the model
    model.save('monkey_model.h5')

if __name__ == '__main__':
    train_cnn()