from keras.models import Model, model_from_json
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import numpy as np
import time

class RegularizedDeepAutoencoder(object):
    model_name = 'RegularizedDeepAutoencoder'

    def __init__(self):
        self.model = None
        self.threshold = None
        self.config = None

    def load_model(self, model_dir_path):
        config_file_path = RegularizedDeepAutoencoder.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.input_dim = self.config['input_dim']
        self.threshold = self.config['threshold']

        architecture_file_path = RegularizedDeepAutoencoder.get_architecture_file_path(model_dir_path)
        self.model = model_from_json(open(architecture_file_path, 'r').read())
        weight_file_path = RegularizedDeepAutoencoder.get_weight_file_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_model(self, input_dim, encoding_dim):
        input_layer = Input(shape=(input_dim,))

        encoder = Dense(encoding_dim, activation="tanh",
                        activity_regularizer=regularizers.l1(10e-6))(input_layer)
        encoder = Dense(200 , activation="relu")(encoder)

        decoder = Dense(200, activation='relu')(encoder)
        decoder = Dense(encoding_dim, activation='relu')(decoder)
        decoder = Dense(input_dim, activation='tanh')(decoder)


        model = Model(inputs=input_layer, outputs=decoder)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, RegularizedDeepAutoencoder.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, RegularizedDeepAutoencoder.model_name + '-weights.h5')

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, RegularizedDeepAutoencoder.model_name + '-config.npy')

    def fit(self, data,validate_data,validate_label,input_dim, encoding_dim,
     model_dir_path, epochs=None, batch_size=None, test_size=None,
      random_state=None,estimated_negative_sample_ratio=None):
        start_time = time.time()
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if epochs is None:
            epochs = 10
        if batch_size is None:
            batch_size = 32
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        weight_file_path = RegularizedDeepAutoencoder.get_weight_file_path(model_dir_path)
        architecture_file_path = RegularizedDeepAutoencoder.get_architecture_file_path(model_dir_path)

        # X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)
        checkpointer = ModelCheckpoint(filepath=weight_file_path,
                                       verbose=0,
                                       save_best_only=True)
        self.input_dim = input_dim
        self.model = self.create_model(input_dim,encoding_dim)
        open(architecture_file_path, 'w').write(self.model.to_json())
        best_acc = 0.0
        best_iter = -1
        for ep in range(epochs):
            history = self.model.fit(data, data,
                                 epochs=1,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 verbose=1,
                                 callbacks=[checkpointer]).history
            scores = self.predict(data)
            scores.sort()
            cut_point = int(estimated_negative_sample_ratio * len(scores))
            self.threshold = scores[cut_point]

            y_predict = self.predict(validate_data)
            y_result = [0 if(x<=self.threshold) else 1 for x in y_predict]
            # Caculate metrics
            acc = metrics.accuracy_score(validate_label, y_result)
            report = metrics.classification_report(validate_label,y_result,digits=4)
            print ('\n clasification report:\n', report)
            print ('Acc:', acc)
            if acc > best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
                # if acc > 0.93:
                #     break
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 10:
                    break
        self.model = best_model
        self.model.save_weights(weight_file_path)

        self.model.save_weights(weight_file_path)

        scores = self.predict(data)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        print(self.threshold)
        self.config['input_dim'] = self.input_dim
        self.config['threshold'] = self.threshold
        config_file_path = RegularizedDeepAutoencoder.get_config_file_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)
        print("best acc: "+str(best_acc))
        print("--- %s seconds ---" % (time.time() - start_time))
        return history

    def predict(self, data):
        target_data = self.model.predict(x=data)
        dist = np.linalg.norm(data - target_data, axis=-1)
        return dist

    def anomaly(self, data, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(data)
        return zip(dist >= self.threshold, dist)
    def setThreshold(self, data, est):
        scores = self.predict(data)
        scores.sort()
        cut_point = int(est * len(scores))
        self.threshold = scores[cut_point]
    def setThresholdStd(self, data, std):
        scores = self.predict(data)
        scores.sort()
        Mean = scores.mean()
        threshold = Mean + std*scores.std()
        print("Mean: ",Mean)
        print("Std: ",scores.std())
        print("threshold: ",threshold)
        print("________________________________________________--")
        self.threshold = threshold