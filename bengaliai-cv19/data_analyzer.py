import pandas as pd
import numpy as np
import cv2
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

from multi_output_data_generator import MultiOutputDataGenerator


class DataAnalyzer:

    def __init__(self):
        self.train = pd.read_csv('./data/train.csv')
        self.test = pd.read_csv('./data/test.csv')
        self.class_map = pd.read_csv('./data/class_map.csv')
        self.sample_sub = pd.read_csv('./data/sample_submission.csv')
        self.model = None

    def get_n(self, df, field, n, top=True):
        top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[
                        :n]
        top_grapheme_roots = top_graphemes.index
        top_grapheme_counts = top_graphemes.values
        top_graphemes = self.class_map[self.class_map['component_type'] == field].reset_index().iloc[top_grapheme_roots]
        top_graphemes.drop(['component_type', 'label'], axis=1, inplace=True)
        top_graphemes.loc[:, 'count'] = top_grapheme_counts
        return top_graphemes

    @staticmethod
    def image_from_char(char, width=236, height=236):
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        my_font = ImageFont.truetype('./data/Kalpurush.ttf', 120)
        w, h = draw.textsize(char, font=my_font)
        draw.text(((width - w) / 2, (height - h) / 3), char, font=my_font)
        return image

    def image_from_chars(self, data: pd.DataFrame):
        size = len(data.index)
        f, ax = plt.subplots(int(size / 5), 5, figsize=(16, 8))
        ax = ax.flatten()

        for i in range(size):
            ax[i].imshow(self.image_from_char(data['component'].iloc[i]), cmap='Greys')
        plt.show()

    @staticmethod
    def convert_train(train: pd.DataFrame):
        train = train.drop(['grapheme'], axis=1, inplace=False)
        train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train[
            ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
        return train

    def resize(self, df, resize_size=64):
        resized = {}
        for i in range(df.shape[0]):
            df_index = df.index[i]
            image_values = df.loc[df_index].values
            resized[df_index] = self.resize_image(image_values, resize_size)
        resized = pd.DataFrame(resized).T
        return resized

    def resize_image(self, image_values, resize_size=64):
        image = image_values.reshape(137, 236)
        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        idx = 0
        ls_xmin = []
        ls_ymin = []
        ls_xmax = []
        ls_ymax = []
        for cnt in contours:
            idx += 1
            x, y, w, h = cv2.boundingRect(cnt)
            ls_xmin.append(x)
            ls_ymin.append(y)
            ls_xmax.append(x + w)
            ls_ymax.append(y + h)
        xmin = min(ls_xmin)
        ymin = min(ls_ymin)
        xmax = max(ls_xmax)
        ymax = max(ls_ymax)

        roi = image[ymin:ymax, xmin:xmax]
        resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        return resized_roi.reshape(-1)

    @staticmethod
    def get_dummies(df):
        cols = []
        for col in df:
            cols.append(pd.get_dummies(df[col].astype(str)))
        return pd.concat(cols, axis=1)

    @staticmethod
    def create_model(img_size=64):
        inputs = Input(shape=(img_size, img_size, 1))

        model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu',
                       input_shape=(img_size, img_size, 1))(inputs)
        model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
        model = Dropout(rate=0.3)(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = Dropout(rate=0.3)(model)

        model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = Dropout(rate=0.3)(model)

        model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = Dropout(rate=0.3)(model)

        model = Flatten()(model)
        model = Dense(1024, activation="relu")(model)
        model = Dropout(rate=0.3)(model)
        dense = Dense(512, activation="relu")(model)

        head_root = Dense(168, activation='softmax')(dense)
        head_vowel = Dense(11, activation='softmax')(dense)
        head_consonant = Dense(7, activation='softmax')(dense)

        model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
        model.summary()

        plot_model(model, to_file='./results/model.png')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_data(self, img_size=64, n_channels=1, batch_size=256, epochs=30):
        learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_3_accuracy',
                                                         patience=3,
                                                         verbose=1,
                                                         factor=0.5,
                                                         min_lr=0.00001)
        learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_4_accuracy',
                                                          patience=3,
                                                          verbose=1,
                                                          factor=0.5,
                                                          min_lr=0.00001)
        learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_5_accuracy',
                                                              patience=3,
                                                              verbose=1,
                                                              factor=0.5,
                                                              min_lr=0.00001)
        mc = ModelCheckpoint("./results/model.h5", monitor='val_loss', mode='min',
                             save_best_only=True)
        self.model = self.create_model(img_size)

        train = self.convert_train(self.train)
        histories = []
        for i in range(4):
            train_df = pd.merge(pd.read_parquet(f'./data/train_image_data_{i}.parquet'), train, on='image_id').drop(
                ['image_id'], axis=1)

            # Visualize few samples of current training dataset
            fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
            count = 0
            for row in ax:
                for col in row:
                    col.imshow(self.resize(
                        train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[
                            [count]]).values.reshape(-1).reshape(img_size, img_size).astype(
                        np.float64))
                    count += 1

            plt.savefig(f"./results/train_image_data_{i}.png")
            plt.close()

            X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
            X_train = self.resize(X_train) / 255

            # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
            X_train = X_train.values.reshape(-1, img_size, img_size, n_channels)

            Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
            Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
            Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values

            print(f'Training images: {X_train.shape}')
            print(f'Training labels root: {Y_train_root.shape}')
            print(f'Training labels vowel: {Y_train_vowel.shape}')
            print(f'Training labels consonants: {Y_train_consonant.shape}')

            # Divide the data into training and validation set
            x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(
                X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)

            # Data augmentation for creating more training data
            datagen = MultiOutputDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range=0.15,  # Randomly zoom image
                width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            datagen.fit(x_train)

            # Fit the model
            history = self.model.fit_generator(
                datagen.flow(x_train, {'dense_2': y_train_root, 'dense_3': y_train_vowel, 'dense_4': y_train_consonant},
                             batch_size=batch_size),
                epochs=epochs, validation_data=(x_test, [y_test_root, y_test_vowel, y_test_consonant]),
                steps_per_epoch=x_train.shape[0] // batch_size,
                callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel,
                           learning_rate_reduction_consonant, mc],
                verbose=2)
            histories.append(history)

        for dataset in range(4):
            self.plot_loss(histories[dataset], epochs, f'Training Dataset: {dataset}')
            self.plot_acc(histories[dataset], epochs, f'Training Dataset: {dataset}')



    @staticmethod
    def plot_loss(his, epoch, title):
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
        plt.plot(np.arange(0, epoch), his.history['dense_2_loss'], label='train_root_loss')
        plt.plot(np.arange(0, epoch), his.history['dense_3_loss'], label='train_vowel_loss')
        plt.plot(np.arange(0, epoch), his.history['dense_4_loss'], label='train_consonant_loss')

        plt.plot(np.arange(0, epoch), his.history['val_dense_2_loss'], label='val_train_root_loss')
        plt.plot(np.arange(0, epoch), his.history['val_dense_3_loss'], label='val_train_vowel_loss')
        plt.plot(np.arange(0, epoch), his.history['val_dense_4_loss'], label='val_train_consonant_loss')

        plt.title(title)
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(f"./results/loss_plot_{title}.png")
        plt.close()

    @staticmethod
    def plot_acc(his, epoch, title):
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, epoch), his.history['dense_2_accuracy'], label='train_root_acc')
        plt.plot(np.arange(0, epoch), his.history['dense_3_accuracy'], label='train_vowel_accuracy')
        plt.plot(np.arange(0, epoch), his.history['dense_4_accuracy'], label='train_consonant_accuracy')

        plt.plot(np.arange(0, epoch), his.history['val_dense_2_accuracy'], label='val_root_acc')
        plt.plot(np.arange(0, epoch), his.history['val_dense_3_accuracy'], label='val_vowel_accuracy')
        plt.plot(np.arange(0, epoch), his.history['val_dense_4_accuracy'], label='val_consonant_accuracy')
        plt.title(title)
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(f"./results/acc_plot_{title}.png")
        plt.close()

    def predict_test(self, img_size=64, n_channels=1):
        preds_dict = {
            'grapheme_root': [],
            'vowel_diacritic': [],
            'consonant_diacritic': []
        }
        components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
        target = []  # model predictions placeholder
        row_id = []  # row_id place holder
        for i in range(4):
            df_test_img = pd.read_parquet('./data/test_image_data_{}.parquet'.format(i))
            df_test_img.set_index('image_id', inplace=True)

            X_test = self.resize(df_test_img) / 255
            X_test = X_test.values.reshape(-1, img_size, img_size, n_channels)

            preds = self.model.predict(X_test)

            for i, p in enumerate(preds_dict):
                preds_dict[p] = np.argmax(preds[i], axis=1)

            for k, id in enumerate(df_test_img.index.values):
                for i, comp in enumerate(components):
                    id_sample = id + '_' + comp
                    row_id.append(id_sample)
                    target.append(preds_dict[comp][k])

        df_sample = pd.DataFrame({
                'row_id': row_id,
                'target': target
            },
            columns=['row_id', 'target']
        )
        df_sample.to_csv('./results/submission.csv', index=False)


if __name__ == '__main__':
    loader = DataAnalyzer()
    loader.train_data()
    loader.predict_test()
