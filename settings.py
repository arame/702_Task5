import datetime
import os
import sys
class Settings:
    sample_size_cut = 0.3   # 30% of images for a person will be neutral, and 30% will be the emotion
    pathOutput = "../output/"
    pathInputFile = "../files/ck_final.pickle"
    pathSaveNet = "../save/"
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    dateString = ""
    minibatch_size = 32
    epochs = 150
    lr = 0.005
    lr_d = 0.6
    schedule = 80
    momentum = 0
    seed = 42
    shuffle=True
    weights_init = "rand_norm"
    optimizer = "sgd"
    hidden_n =[300, 200]
    l2 = 0.0            # -L2 weight decay and dropout cannot be run at the same time (usually 0.0001)
    dropout = False     # The code will set it to True if dropoutRate > 0
    drop_prob1= 0
    drop_prob2= 0

    @staticmethod
    def start():
        Settings.dateString = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
        Settings.validateHyperparameters()
        Settings.pathOutput = Settings.setSettingPath(Settings.pathOutput)
        Settings.pathSaveNet = Settings.setSettingPath(Settings.pathSaveNet)
        Settings.printHyperparameters()

    @staticmethod
    def validateHyperparameters():
        if Settings.l2 > 0 and (Settings.drop_prob1 > 0 or Settings.drop_prob2 > 0):
            sys.exit("!! Cannot run both l2 and dropout at the same time")
        if Settings.dropout == True and Settings.l2 != 0:
            sys.exit("Select either dropout or l2 regularisation")
            
        if Settings.weights_init != "trunc_norm" and Settings.weights_init != "rand_norm":
            sys.exit("No an initilisation choice, select from either 'trunc_norm' or 'rand_norm'")

        if Settings.optimizer != "sgd_mo" and Settings.optimizer != "sgd" and Settings.optimizer != "adam":
            sys.exit("No an optimizer choice, select from 'sgd_mo', 'sgd' or 'adam'")

        if (Settings.optimizer == "sgd"and Settings.momentum != 0) or (Settings.optimizer == "adam"and Settings.momentum != 0):
            sys.exit("Zero momentum or change the optimizer to sgd_mo")

    @staticmethod
    # Create a unique folder for the output files to avoid file name clashes and 
    # make it easier to locate output files for each run
    def setSettingPath(path):
        path = path + Settings.dateString
        os.makedirs(path)
        path = path + "/"
        return path

    @staticmethod
    def printHyperparameters():
        print("*"*100)
        print("* Hyperparameters")
        print("* ---------------")
        print("Learning Rate  ", Settings.lr)
        print("Batch Size     ", Settings.minibatch_size)
        print("Epochs         ", Settings.epochs)
        print("Momentum       ", Settings.momentum)
        if Settings.drop_prob1 == 0 and Settings.drop_prob2 == 0:
            print("!! No Drop out")
            Settings.dropout = False
        else:
            print("Dropout Rates   ", Settings.drop_prob1, " and ", Settings.drop_prob2)
            Settings.dropout = True
        if Settings.l2 == 0:
            print("!! No L2 weight decay")
        else:
            print("L2 weight decay", Settings.l2)

        print("Output files are located in the folder ", Settings.pathOutput)

class ImageSize:
    width = 100
    height = 100

class Emotion:
    num_classes = 0
    class_dict = {}
    idx2class = {}