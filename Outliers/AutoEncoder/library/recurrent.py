from keras.layers import Conv1D,TimeDistributed, GlobalMaxPool1D, Dense, Flatten, LSTM, Bidirectional, RepeatVector, MaxPooling1D
from keras.models import Sequential
from sklearn import metrics
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np


class LstmAutoEncoder(object):
    model_name = 'lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    # def create_model(time_window_size, metric):
    #     model = Sequential()
    #     model.add(LSTM(units=128, input_shape=(time_window_size, 1), return_sequences=False))

    #     model.add(Dense(units=time_window_size, activation='tanh'))

    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     print(model.summary())
    #     return model

    def create_model(input_dim, encoding_dim):
        model = Sequential()
        model.add(LSTM(64,return_sequences=True,go_backwards=True))
        model.add(LSTM(input_dim,return_sequences=True))
        # # # model.add(Dropout(0.5))
        # model.add(Dense())
        # # # # model.add(Activation('softmax'))
        # model.add(Lambda(lambda x: K.argmax(x)))
        model.compile(loss='mean_squared_error',optimizer='adam')
        # print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder.create_model(349, 200)
        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-architecture.json'
    def updateThreshold(self,timeseries_dataset,estimated_negative_sample_ratio):
        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]


    def fit(self, timeseries_dataset,validate_data,validate_label, model_dir_path, batch_size=None, epochs=None, metric=None,
            estimated_negative_sample_ratio=None,input_dim =200, encoding_dim = 200):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20 
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = 1
        # input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder.create_model(input_dim, encoding_dim)
        open(architecture_file_path, 'w').write(self.model.to_json())
        best_acc = 0.0
        best_iter = -1
        for ep in range(epochs):
            self.model.fit(x=timeseries_dataset, y= np.flip(timeseries_dataset,2),
                       batch_size=batch_size, epochs=1,
                       verbose=LstmAutoEncoder.VERBOSE,
                       callbacks=[checkpoint])
            scores = self.predict(timeseries_dataset)
            scores.sort()
            cut_point = int(estimated_negative_sample_ratio * len(scores))
            self.threshold = scores[cut_point]

            y_predict = self.predict(validate_data)
            y_result = [0 if(x<=self.threshold) else 1 for x in y_predict]
            # Caculate metrics
            acc = metrics.accuracy_score(validate_label, y_result)
            report = metrics.classification_report(validate_label,y_result,digits=4)
            print '\n clasification report:\n', report
            print 'Acc:', acc
            if acc > best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
            else:
                break
            # No longer improving...break and calc statistics
            if (ep-best_iter) > 3:
                break
        self.model = best_model
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - np.flip(target_timeseries_dataset,2), axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
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

class CnnLstmAutoEncoder(object):
    model_name = 'cnn-lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                         input_shape=(time_window_size, 1)))
        model.add(MaxPooling1D(pool_size=4))

        model.add(LSTM(128))
        # 82% acc
        model.add(Dense(units=time_window_size, activation='linear'))
        # model.add(Dense(units=time_window_size, activation='tanh')) 
        model.compile(optimizer='adam', loss='mean_squared_error')

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = CnnLstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = CnnLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = CnnLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset,validate_data,validate_label, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = CnnLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = CnnLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = CnnLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        best_acc = 0.0
        best_iter = -1
        for ep in range(epochs):
            print("this is the current ep: "+str(ep))
            self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=1,
                       verbose=LstmAutoEncoder.VERBOSE,
                       callbacks=[checkpoint])
            scores = self.predict(timeseries_dataset)
            scores.sort()
            cut_point = int(estimated_negative_sample_ratio * len(scores))
            self.threshold = scores[cut_point]
            y_predict = self.predict(validate_data)
            #y_result = [np.argmax(x) for x in y_probs]
            y_result = [0 if(x<=self.threshold) else 1 for x in y_predict]
            # print(y_predict)
            # print(y_result)
            # print(validate_label)
            # Caculate metrics
            acc = metrics.accuracy_score(validate_label, y_result)
            # print y_result
           
            # f1_score = metrics.f1_score(y_holdout, y_result,average="macro")
            # precision = metrics.precision_score(y_holdout, y_result,average="macro")
            # recall = metrics.recall_score(y_holdout, y_result,average="macro")
            report = metrics.classification_report(validate_label,y_result,digits=4)
            # print '\n clasification report:\n', report
            print 'Acc:', acc
            if acc > best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
                # if best_acc > 0.93:
                #     break
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 3:
                    break
        self.model = best_model
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = CnnLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        print(target_timeseries_dataset)
        print(timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class BidirectionalLstmAutoEncoder(object):
    model_name = 'bidirectional-lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(time_window_size, 1)))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        # print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_datasetvali,date_data,validate_label, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = BidirectionalLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        
        best_acc = 0.0
        best_iter = -1
        for ep in range(epochs):
            self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=1,
                       verbose=LstmAutoEncoder.VERBOSE,
                       callbacks=[checkpoint])
            scores = self.predict(timeseries_dataset)
            scores.sort()
            cut_point = int(estimated_negative_sample_ratio * len(scores))
            self.threshold = scores[cut_point]
            y_predict = self.predict(validate_data)
            #y_result = [np.argmax(x) for x in y_probs]
            y_result = [0 if(x<=self.threshold) else 1 for x in y_predict]
            # print(y_predict)
            # print(y_result)
            # print(validate_label)
            # Caculate metrics
            acc = metrics.accuracy_score(validate_label, y_result)
            # print y_result
           
            # f1_score = metrics.f1_score(y_holdout, y_result,average="macro")
            # precision = metrics.precision_score(y_holdout, y_result,average="macro")
            # recall = metrics.recall_score(y_holdout, y_result,average="macro")
            report = metrics.classification_report(validate_label,y_result,digits=4)
            # print '\n clasification report:\n', report
            print 'Acc:', acc
            if acc > best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 3:
                    break
        self.model = best_model

        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)

        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class BidirectionalGruAutoEncoder(object):
    model_name = 'bidirectional-gru-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Bidirectional(GRU(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(time_window_size, 1)))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = BidirectionalGruAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset,validate_data,validate_label, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = BidirectionalLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        best_acc = 0.0
        best_iter = -1
        for ep in range(epochs):
            self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=1,
                       verbose=LstmAutoEncoder.VERBOSE,
                       callbacks=[checkpoint])
            scores = self.predict(timeseries_dataset)
            scores.sort()
            cut_point = int(estimated_negative_sample_ratio * len(scores))
            self.threshold = scores[cut_point]
            y_predict = self.predict(validate_data)
            #y_result = [np.argmax(x) for x in y_probs]
            y_result = [0 if(x<=self.threshold) else 1 for x in y_predict]
            # print(y_predict)
            # print(y_result)
            # print(validate_label)
            # Caculate metrics
            acc = metrics.accuracy_score(validate_label, y_result)
            # print y_result
           
            # f1_score = metrics.f1_score(y_holdout, y_result,average="macro")
            # precision = metrics.precision_score(y_holdout, y_result,average="macro")
            # recall = metrics.recall_score(y_holdout, y_result,average="macro")
            report = metrics.classification_report(validate_label,y_result,digits=4)
            # print '\n clasification report:\n', report
            print 'Acc:', acc
            if acc > best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 3:
                    break
        self.model = best_model
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)

        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)
class GruAutoEncoder(object):
    model_name = 'gru-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(GRU(units=128, input_shape=(time_window_size, 1), return_sequences=False))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = GruAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset,validate_data,validate_label, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        best_acc = 0.0
        best_iter = -1
        for ep in range(epochs):
            self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=1,
                       verbose=LstmAutoEncoder.VERBOSE,
                       callbacks=[checkpoint])
            scores = self.predict(timeseries_dataset)
            scores.sort()
            cut_point = int(estimated_negative_sample_ratio * len(scores))
            self.threshold = scores[cut_point]
            y_predict = self.predict(validate_data)
            #y_result = [np.argmax(x) for x in y_probs]
            y_result = [0 if(x<=self.threshold) else 1 for x in y_predict]
            # print(y_predict)
            # print(y_result)
            # print(validate_label)
            # Caculate metrics
            acc = metrics.accuracy_score(validate_label, y_result)
            # print y_result
           
            # f1_score = metrics.f1_score(y_holdout, y_result,average="macro")
            # precision = metrics.precision_score(y_holdout, y_result,average="macro")
            # recall = metrics.recall_score(y_holdout, y_result,average="macro")
            report = metrics.classification_report(validate_label,y_result,digits=4)
            # print '\n clasification report:\n', report
            print 'Acc:', acc
            if acc > best_acc:
                best_iter = ep
                best_acc = acc
                best_model =  self.model
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 3:
                    break
        self.model = best_model
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)
