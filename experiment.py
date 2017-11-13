import json
from collections import deque

from dataset import Dataset, DatasetType
from nn.evolution import EvolutionaryStrategy
from nn.neural_network import ArtificialNeuralNetwork
from nn.trainer import NetworkTrainer
from util import QueuedCsvWriter
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score


class Experiment:

    def __init__(self,
                 results_file_name,
                 models_file_name,
                 epoch_patience=100):
        """

        :param network: The neural network to perform the experiment
        :param dataset: The dataset to train on
        :param results_file_name: The results file name
        :param models_file_name: The models file name
        :param learning_rate: Learning rate of the model
        :param epoch_patience: Number of epochs for which if the error
            doesn't change, assume we are in a minimum and stop training.
        """
        self.results_recorder = QueuedCsvWriter(results_file_name, ["epoch", "mse_train", "mse_validation", "precision", "recall", "accuracy"])
        self.models_recorder = QueuedCsvWriter(models_file_name, ["epoch", "model"], 1)

        # queues to store the last n epochs mean squared errors
        # to determine if we should continue training
        self.mse_validation_queue = deque(maxlen=epoch_patience)
        self.mse_train_queue = deque(maxlen=epoch_patience)
        self.epoch_patience = epoch_patience
        self.epoch = 0
        print(
            """\
=====
Starting experiment.
    epoch_patience: {patience}
    results_file  : {results}
    models_file   : {models}
=====
            """.format(patience=epoch_patience,
                       results=results_file_name,
                       models=models_file_name)
        )


    def run(self):
        """
        Needs to be implemeneted by subclass
        """
        raise NotImplementedError()

    def save_model(self, epoch):
        """
        Needs to be implemeneted by subclass
        """
        raise NotImplementedError()

    def should_stop_training(self):
        """
        Determines if we should continue training dependent upon
        any of the exit criterion have been met.
        """
        # the conditions that we have specified
        # to stop the training process
        conditions = [
            self.err_not_changing(self.mse_validation_queue),
            self.err_not_changing(self.mse_train_queue),
            self.average_validation_err_increasing() # TODO determine if we want this or not
        ]
        return any(conditions)

    def err_not_changing(self, err_queue: deque):
        if len(err_queue) == self.epoch_patience:
            # a set cannot contain duplicates
            # so if the length is 1, then we have
            # all duplicates, i.e. the error hasn't
            # changed
            if len(set(err_queue)) == 1:
                print("== Error hasn't changed for %d epochs, stopping training ==" % self.epoch_patience)
                return True
        else:
            return False

    def average_validation_err_increasing(self):
        """
        Determines if the average validation error is increasing
        on average for the last :patience: epochs, suggesting
        we are starting to overfit the model. We only determine
        this is a potential issue if we have elapsed at least
        1000 epochs, suggesting the model is still extremely
        volatile and will not converge.
        """
        if self.epoch > 1000: # we must run at least 1000 epochs before we check this
            sum_diff = 0
            for i in reversed(range(1, len(self.mse_validation_queue))):
                diff = self.mse_validation_queue[i] - self.mse_validation_queue[i-1]
                sum_diff += 1 if diff > 0 else -1
            if sum_diff > 0:
                print("== The average validation error has a positive increase on average in the last %d epochs, stopping training ==" % self.epoch_patience)
                return True
            else:
                return False

    def exit_handler(self):
        """
        Called when a termination signal is sent to the program,
        so we make the state of the model and the results writers
        are flushed before we terminate.
        """
        self.models_recorder.flush()
        self.results_recorder.flush()
        print("Exited prematurely...")



class BackpropExperiment(Experiment):

    def __init__(self,
                 network: ArtificialNeuralNetwork,
                 dataset: Dataset,
                 results_file_name,
                 models_file_name,
                 learning_rate=0.1,
                 epoch_patience=100,
                 classification=False,
                 num_classes=None):
        """

        :param network: The neural network to perform the experiment
        :param dataset: The dataset to train on
        :param results_file_name: The results file name
        :param models_file_name: The models file name
        :param learning_rate: Learning rate of the model
        :param epoch_patience: Number of epochs for which if the error
            doesn't change, assume we are in a minimum and stop training.
        """
        super().__init__(results_file_name, models_file_name, epoch_patience)
        training_set = dataset.get_train()
        validation_set = dataset.get_validation()
        self.dataset = dataset
        self.network = network
        self.trainer = NetworkTrainer(network, training_set, validation_set, learning_rate, classification=classification, num_classes=num_classes)

    def run(self):
        for [epoch, mse_train, mse_validation] in self.trainer.train_batch():
            print("epoch=%d, mse_train=%f, mse_validation=%f" % (epoch, mse_train, mse_validation))
            self.epoch = epoch
            self.mse_train = mse_train
            self.mse_validation = mse_validation
            self.mse_train_queue.append(round(mse_train, 6))
            self.mse_validation_queue.append(round(mse_validation, 6))

            if epoch % 50 == 0:
                self.save_model(epoch)

            if epoch % 50 == 0:
                self.save_stats()

            if self.should_stop_training():
                print("=== Training was completed. ===")
                self.save_stats()
                self.trainer.stop()

    def save_stats(self):
        if self.dataset.type == DatasetType.CLASSIFICATION:
            # print a classification report on the accuracy
            X, Y = self.dataset.X, self.dataset.CLASS_Y
            predicted_y = self.network.predict(X, True)
            precision = precision_score(Y, predicted_y, average="weighted")
            recall = recall_score(Y, predicted_y, average="weighted")
            accuracy = accuracy_score(Y, predicted_y)
            self.results_recorder.writerow(
                [str(self.epoch), str(self.mse_train), str(self.mse_validation), str(precision), str(recall), str(accuracy)]
            )
            print(classification_report(Y, predicted_y))

    def save_model(self, epoch):
        model = self.network.json()
        model_serialized = json.dumps(model)
        self.models_recorder.writerow([str(epoch), model_serialized])

    def exit_handler(self):
        self.save_stats()
        super().exit_handler()

class EvolutionaryExperiment(Experiment):

    def __init__(self,
                 trainer: EvolutionaryStrategy,
                 dataset: Dataset,
                 results_file_name,
                 models_file_name,
                 max_generations=10000
    ):
        super().__init__(results_file_name, models_file_name)
        self.trainer = trainer
        self.dataset = dataset
        self.max_generations = max_generations

    def run(self):

        for generation in range(self.max_generations):
            self.epoch = generation
            [train_fitness, validation_fitness] = self.trainer.run_generation()
            self.train_fitness = train_fitness
            self.validation_fitness = validation_fitness
            print("generation=%d, train_fitness=%f, valid_fitness=%f" % (generation, train_fitness, validation_fitness))
            self.mse_train_queue.append(round(train_fitness, 6))
            self.mse_validation_queue.append(round(validation_fitness, 6))

            self.results_recorder.writerow([str(generation), "%f" % train_fitness, "%f" % validation_fitness])

            # save our model progress over time
            if generation % 50 == 0:
                self.save_model(generation)

            # print statistics over time
            if generation % 50 == 0:
                self.results_recorder.writerow([str(generation), "%f" % train_fitness, "%f" % validation_fitness])
                self.save_stats()

            if self.should_stop_training():
                self.save_stats()
                print("=== Training was completed. ===")
                break

        self.save_stats()

    def save_model(self, epoch):
        model = self.trainer.get_fittest_individual().json()
        model_serialized = json.dumps(model)
        self.models_recorder.writerow([str(epoch), model_serialized])

    def save_stats(self):
        if self.dataset.type == DatasetType.CLASSIFICATION:
            # print a classification report on the accuracy
            X, Y = self.dataset.X, self.dataset.CLASS_Y
            network = self.trainer.get_fittest_individual()
            predicted_y = network.predict(X, True)
            precision = precision_score(Y, predicted_y, average="weighted")
            recall = recall_score(Y, predicted_y, average="weighted")
            accuracy = accuracy_score(Y, predicted_y)
            print(classification_report(Y, predicted_y))
            self.results_recorder.writerow(
                [str(self.epoch), str(self.train_fitness), str(self.validation_fitness), str(precision), str(recall), str(accuracy)]
            )


    def exit_handler(self):
        self.save_stats()
        super().exit_handler()