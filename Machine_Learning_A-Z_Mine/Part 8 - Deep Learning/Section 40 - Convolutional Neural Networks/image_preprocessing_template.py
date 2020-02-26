# %% md
# Image Prepocessing Template

# %% codecell
# importing libaries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# pd.options.display.html.table_schema = True
# pd.options.display.max_rows = None


# %% codecell
# Augmenting images
# Image Data Generator / Aumentation --> augments amount of images for tranining,
# by applying random transformation, creating more batches of images
def augment_images():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return train_datagen, test_datagen


# %% codecell
# Encode Categorial Data
def import_images(train_datagen, test_datagen):
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    return training_set, test_set


# %% codecell
def preprocess_images():
    training_set, test_set = augment_images()
    return import_images(training_set, test_set)


# %% codecell
# main
if __name__ == '__main__':
    preprocess_images()
