import json

from collections import deque
from nn.neural_network import ArtificialNeuralNetwork
from nn.trainer import NetworkTrainer
from util import QueuedCsvWriter

def partition(dataset):
    """
    Returns a 70/30 split of the dataset
    """
    train_index = int(0.7*len(dataset))
    return [dataset[:train_index], dataset[train_index:]]

class Experiment:

    def __init__(self, network: ArtificialNeuralNetwork,
                 dataset,
                 results_file_name,
                 models_file_name,
                 learning_rate=0.1,
                 epoch_patience=50):
        """

        :param network: The neural network to perform the experiment
        :param dataset: The dataset to train on
        :param results_file_name: The results file name
        :param models_file_name: The models file name
        :param learning_rate: Learning rate of the model
        :param epoch_patience: Number of epochs for which if the error
            doesn't change, assume we are in a minimum and stop training.
        """
        [training_set, validation_set] = partition(dataset)
        self.network = network
        self.trainer = NetworkTrainer(network, training_set, validation_set, learning_rate)
        self.results_recorder = QueuedCsvWriter(results_file_name, ["epoch", "mse_train", "mse_validation"])
        self.models_recorder = QueuedCsvWriter(models_file_name, ["epoch", "model"], 1)

        # queues to store the last n epochs mean squared errors
        # to determine if we should continue training
        self.mse_validation_queue = deque(maxlen=epoch_patience)
        self.mse_train_queue = deque(maxlen=epoch_patience)
        self.epoch_patience = epoch_patience
        print(
            """
            =====
            Starting experiment.
                learning_rate : {learning_rate}
                |D|           : {size}
                epoch_patience: {patience}
                results_file  : {results}
                models_file   : {models}
            =====
            """.format(learning_rate=learning_rate, size=len(dataset),
                       patience=epoch_patience, results=results_file_name,
                       models=models_file_name)
        )


    def run(self):
        for [epoch, mse_train, mse_validation] in self.trainer.train_regression_batch():
            self.mse_train_queue.append(mse_train)
            self.mse_validation_queue.append(mse_validation)

            self.results_recorder.writerow([str(epoch), "%f" % mse_train, "%f" % mse_validation])
            if epoch % 50 == 0:
                self.save_model(epoch)

            if self.should_stop_training():
                print("=== Training was completed. ===")
                self.trainer.stop()

    def save_model(self, epoch):
        model = self.network.json()
        model_serialized = json.dumps(model)
        self.models_recorder.writerow([str(epoch), model_serialized])

    def should_stop_training(self):
        """
        Determines if we should continue training dependent upon
        any of the exit criterion have been met.
        """
        # the conditions that we have specified
        # to stop the training process
        conditions = [
            self.err_not_changing(self.mse_validation_queue),
            self.err_not_changing(self.mse_train_queue)
        ]
        return any(conditions)

    def err_not_changing(self, err_queue: deque):
        if len(err_queue) == self.epoch_patience:
            # a set cannot contain duplicates
            # so if the length is 1, then we have
            # all duplicates, i.e. the error hasn't
            # changed
            if len(set(err_queue)) == 1:
                print("== Error hasn't changed for %d epochs, stopping training ==")
        else:
            return False

    def average_err_diff_less_than_threshold(self, mse_validation_queue):
        """
        Determines if the average change in the last :patience: epochs
        is less than the specified
        :param mse_validation_queue:
        :return:
        """
        pass

    def exit_handler(self):
        """
        Called when a termination signal is sent to the program,
        so we make the state of the model and the results writers
        are flushed before we terminate.
        """
        self.models_recorder.flush()
        self.results_recorder.flush()
        print("Exited prematurely...")

