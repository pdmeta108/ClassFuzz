import classifier
from random import randint, random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Qt5
import easygui
from PyQt5 import QtWidgets
from gui import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import sys
import colorama
from termcolor import colored
# Scikit-Fuzzy
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#### VARIABLES GA ####
mutProb = .05
tournamentSize = 5
elitism = True
popSize = 20
GENERATION_NUMBER = 20
######################


class Indiv:
    def __init__(self, init=True):
        self.rules = []
        if init:
            for i in range(classifier.nbRules):
                self.rules.append(classifier.generateRule())

    def __str__(self):
        s = ""
        for i in range(classifier.nbRules):
            s += "Regla " + str(i) + ": " + str(self.rules[i]) + "\t"
            s += str(classifier.getConf(self.rules[i])) + "\n"
        return s

    def getFitness(self):
        acc = classifier.getAccuracy(self)
        goodRulesNb, badRulesNb = classifier.checkRules(self)
        complexity = classifier.calcComplexity(self)

        w1 = 0.6
        w2 = 0.4

        score = w1 * (1.0 - acc) + w2 * (float(complexity) / float(len(self.rules) * 4))
        # this maximizes the accuracy and minimizes "complexity"
        score = -1 * score

        return score

        # 1 - nbofaccuratelyclassified / number of cases find

        # inferences = classifier.infer(self.rules)
        # inferences = classifier.simple_infer(self.rules)
        # return classifier.computeFitness(inferences)


class Population:
    """ x = Indiv()
    x.getfit()"""

    def __init__(self, init, size=popSize):
        if init:
            self.listpop = [Indiv() for _ in range(size)]
        else:
            self.listpop = []

    def getFittest(self):
        nb_max = self.listpop[0].getFitness()
        index = 0

        for i in range(1, len(self.listpop)):
            nextFitness = self.listpop[i].getFitness()
            if nextFitness > nb_max:
                nb_max = nextFitness
                index = i
        return self.listpop[index]

class PlotView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

         # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that
        # displays the 'figure'it takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.canvas.updateGeometry()
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QtWidgets.QVBoxLayout()
        # adding tool bar to the layout
        layout.addWidget(self.toolbar)
        # adding canvas to the layout
        layout.addWidget(self.canvas)
        self.setLayout(layout)

class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    # main class that creates the GUI and its functions

    def __init__(self):
        self.train_ratio = 0.5  # the default ratio is 0,5
        self.file_location = "iris.data.txt"
        self.rules = []
        self.output = []
        self.w = None
        super(self.__class__, self).__init__()
        self.setupUi(self)  # create the GUI

        # GUI Connections
        self.browse_file_button.clicked.connect(self.browse_button_clicked)
        self.trainSlider.valueChanged.connect(self.train_to_test_ratio_slider_value_changed)
        self.train_spin_box.valueChanged.connect(self.train_to_test_ratio_spinbox_value_changed)
        self.train_button.clicked.connect(self.train_button_clicked)
        self.poblation_box.valueChanged.connect(self.poblation_value_changed)
        self.generation_box.valueChanged.connect(self.generation_value_changed)
        self.tourney_box.valueChanged.connect(self.tournament_value_changed)
        self.classify_button.clicked.connect(self.classify_button_clicked)
        self.rules_button.clicked.connect(self.rules_button_clicked)
        self.plot_button.clicked.connect(self.plot_button_clicked)
        if self.combo_box_parametros.currentTextChanged:
            self.combo_box_parametros.currentTextChanged.connect(self.combo_box_changed)

        colorama.init()

    # GUI Functions
    def browse_button_clicked(self):
        # open a filechooser and pick the dataset used for the training and testing
        path = easygui.fileopenbox()

        if path:
            self.browse_file_text_edit.setText(path)
            self.file_location = path
            self.train_button.setEnabled(True)
            self.train_spin_box.setEnabled(True)
            self.trainSlider.setEnabled(True)

    def train_button_clicked(self):
        # Dividir datos con el porcentaje de train / test
        try:
            df = pd.read_csv(self.file_location)

            # Normalizar los 4 parametros
            df.iloc[:, 0:4] = df.iloc[:, 0:4].apply(lambda x: x / np.max(x))

            X = df.iloc[:, 0:4]

            # Cambiar 'Iris-Setosa' a 0, 'Iris-versicolor' a 1 y 'Iris-virginica' a 2
            # y despues convertirlo a pandas Series
            y = pd.Series(pd.factorize(df.iloc[:, 4])[0])

            global X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.train_ratio)

            # Reindexar todos
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            # Mostrar datos
            print(colored("Conjunto de Entrenamiento: ", 'yellow'))
            print(X_train)
            print(X_train.shape)

            print(colored("Conjunto del Test: ", 'yellow'))
            print(X_test)
            print(X_test.shape)

            print(colored("Entrenamiento para el Objetivo: ", 'yellow'))
            print(y_train)

            print(colored("Test Objetivo: ", 'yellow'))
            print(y_test)
            print("------------------------")

            print(colored("Entrenamiento Exitoso: ", 'green'))

            # Activar botones
            self.classify_button.setEnabled(True)
            self.generation_box.setEnabled(True)
            self.poblation_box.setEnabled(True)
            self.tourney_box.setEnabled(True)
        except FileNotFoundError:
            print(colored("Un archivo no fue especificado.", 'yellow'))
        except:
            print(colored("Algo salio mal.", 'yellow'))

    def classify_button_clicked(self):
        # Crear clase poblacion para comenzar a generar reglas
        pop = Population(True)
        # Se realiza en cada generacion el proceso de torneo y mutacion
        for i in range(GENERATION_NUMBER):

            newpop = Population(False)
            print(colored("Generacion numero : {}".format(str(i)), 'cyan'))

            for j in range(popSize):
                # Proceso de torneo
                parent1 = tournament(pop)
                parent2 = tournament(pop)

                child = crossOver(parent1, parent2)
                newpop.listpop.append(child)

            for j in range(popSize):
                # Proceso de mutacion
                mutation(newpop.listpop[j])

            pop = newpop
            # Mostrar el mejor set de reglas en la poblacion
            thisFittest = pop.getFittest()
            print(colored("Mejor precision ajustada : {}".format(str(classifier.getAccuracy(thisFittest))), 'red'))
            print(thisFittest)

        # Mostrar la complejidad del set de reglas
        print(colored("Calculo de complejidad: {}".format(classifier.calcComplexity(thisFittest)), 'magenta'))
        # Obtener set de reglas con salida y guardar en Main()
        reglas, salidas = getRulesfromFittest(thisFittest)
        self.rules = reglas
        self.output = salidas

        self.rules_button.setEnabled(True)
        self.sample_button.setEnabled(True)
        self.combo_box_parametros.setEnabled(True)

    def combo_box_changed(self):
        self.plot_button.setEnabled(True)

    def rules_button_clicked(self):
        s = ""
        reglas = binRulestoClassRules(self.rules, self.output)
        # salidas = intOutputtoClassOutput(self.output)
        print(colored("Reglas del sistema", 'magenta'))
        for i in range(len(reglas)):
            s += "Regla " + str(i + 1) + ": " + str(reglas[i]) + "\n"
        print(s)
        # print(colored("Salida de cada regla", 'magenta'))
        # print(salidas)

    def plot_button_clicked(self):

        if self.w is None:
            self.w = PlotView()
            self.w.setWindowTitle("Gráfica de Funcion Miembro")
            self.w.show()
        else:
            self.w.close()  # Close window.
            self.w = None  # Discard reference.
            self.w = PlotView()
            self.w.setWindowTitle("Gráfica de Funcion Miembro")
            self.w.show()
        # clearing old figure
        self.w.figure.clear()
        plot_memberhip_function(graph=self.w, parametro=self.combo_box_parametros.currentIndex())





    def train_to_test_ratio_slider_value_changed(self):
        # change the ratio of train/test samples both in variable and in spinner
        self.train_spin_box.setValue(self.trainSlider.value())
        self.train_ratio = round(1 - self.trainSlider.value() / 100, 2)

    def train_to_test_ratio_spinbox_value_changed(self):
        # change the ratio of train/test samples both in variable and in slider
        self.trainSlider.setValue(self.train_spin_box.value())
        self.train_ratio = round(1 - self.trainSlider.value() / 100, 2)

    def poblation_value_changed(self):
        # change the poblation number value in variable
        global popSize
        popSize = self.poblation_box.value()
        print(popSize)

    def tournament_value_changed(self):
        # change the poblation number value in variable
        global tournamentSize
        tournamentSize = self.poblation_box.value()

    def generation_value_changed(self):
        # change the poblation number value in variable
        global GENERATION_NUMBER
        GENERATION_NUMBER = self.generation_box.value()


def tournament(pop):
    tourList = Population(False, tournamentSize)

    for j in range(tournamentSize):
        indexT = randint(0, popSize - 1)

        pop.listpop[indexT]
        tourList.listpop.append(pop.listpop[indexT])

    return tourList.getFittest()


def crossOver(Indiv1, Indiv2):
    newIndiv = Indiv(False)

    for i in range(classifier.nbRules):
        rule1 = Indiv1.rules[i]

        rule2 = Indiv2.rules[i]

        newIndiv.rules.append(crossoverRules(rule1, rule2))

    return newIndiv


def crossoverRules(rule1, rule2):
    newRule = []

    for i in range(len(rule1)):
        prob = random()
        if prob < 0.5:
            newRule.append(rule1[i])

        else:
            newRule.append(rule2[i])

    return newRule


def mutation(indiv):
    for i in range(classifier.nbRules):

        for j in range(classifier.nbRules):

            prob = random()

            if prob < mutProb:
                indiv.rules[i][j] = 1 - indiv.rules[i][j]


# Obtener set de reglas y salidas del resultado
def getRulesfromFittest(thisFittest):
    reglas = []
    salidas = []
    for i in range(len(thisFittest.rules)):
        reglas.append(thisFittest.rules[i])
        salidas.append(classifier.getConf(thisFittest.rules[i]))
    return reglas, salidas


# Transformar reglas binarias en reglas literales
def binRulestoClassRules (rules, outputs):
    soltext = []
    count = 0
    for rule in rules:
        rule_text = ""
        textosalida = ""
        for i in range(0, len(rule), 3):
            if i < 2:
                if not (rule[i] == 0) and not ((rule[i + 1] == 0) or (rule[i + 2] == 0)):
                    rule_text += "Low SepalLength or "
                elif not (rule[i] == 0) and (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Low SepalLength and "
                if not (rule[i + 1] == 0) and not (rule[i + 2] == 0):
                    rule_text += "Med SepalLength or "
                elif not (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Med SepalLength and "
                if not (rule[i+2] == 0):
                    rule_text += "High SepalLength and "
            elif 2 <= i < 5:
                if not (rule[i] == 0) and not ((rule[i + 1] == 0) or (rule[i + 2] == 0)):
                    rule_text += "Low SepalWidth or "
                elif not (rule[i] == 0) and (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Low SepalWidth and "
                if not (rule[i + 1] == 0) and not (rule[i + 2] == 0):
                    rule_text += "Med SepalWidth or "
                elif not (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Med SepalWidth and "
                if not (rule[i + 2] == 0):
                    rule_text += "High SepalWidth and "
            elif 5 <= i < 8:
                if not (rule[i] == 0) and not ((rule[i + 1] == 0) or (rule[i + 2] == 0)):
                    rule_text += "Low PetalLength or "
                elif not (rule[i] == 0) and (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Low PetalLength and "
                if not (rule[i + 1] == 0) and not (rule[i + 2] == 0):
                    rule_text += "Med PetalLength or "
                elif not (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Med PetalLength and "
                if not (rule[i + 2] == 0):
                    rule_text += "High PetalLength and "
            elif 8 <= i < 11:
                if not (rule[i] == 0) and not ((rule[i + 1] == 0) or (rule[i + 2] == 0)):
                    rule_text += "Low PetalWidth or "
                elif not (rule[i] == 0) and (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Low PetalWidth and "
                if not (rule[i + 1] == 0) and not (rule[i + 2] == 0):
                    rule_text += "Med PetalWidth or "
                elif not (rule[i + 1] == 0) and (rule[i + 2] == 0):
                    rule_text += "Med PetalWidth and "
                if not (rule[i + 2] == 0):
                    rule_text += "High PetalWidth and "
        rule_text_2 = rule_text.rstrip('and ')
        # Output process
        tuple = outputs[count]
        if tuple[0] == 0:
            textosalida += "Iris-Setosa"
        elif tuple[0] == 1:
            textosalida += "Iris-versicolor"
        elif tuple[0] == 2:
            textosalida += "Iris-virginica"
        else:
            textosalida += "Not Valid"
        rule_text_2 += " entonces " + textosalida
        soltext.append(rule_text_2)
        count += 1
    return soltext

# Transformar tupla de salidas a literal
def intOutputtoClassOutput (salidas):
    textosalida = ""
    for salida in salidas:
        value = int(salida[0])
        if value == 0:
            textosalida += "Iris-Setosa" + ", "
        elif value == 1:
            textosalida += "Iris-versicolor" + ", "
        elif value == 2:
            textosalida += "Iris-virginica" + ", "
        else:
            textosalida += "Not Valid" + ", "
    return textosalida.rstrip(', ')

def plot_memberhip_function(graph, parametro):

    titulo_parametro = ""
    x_data = None

    # Variables universales
    x_sepall = np.arange(0, 1.0001, 0.0001)
    x_sepalw = np.arange(0, 1.0001, 0.0001)
    x_petall = np.arange(0, 1.0001, 0.0001)
    x_petalw = np.arange(0, 1.0001, 0.0001)

    if parametro == 1:
        titulo_parametro += "Sepal Largo"
        x_data = x_sepall
    elif parametro == 2:
        titulo_parametro += "Sepal Ancho"
        x_data = x_sepalw
    elif parametro == 3:
        titulo_parametro += "Petal Largo"
        x_data = x_petall
    elif parametro == 4:
        titulo_parametro += "Petal Ancho"
        x_data = x_petalw
    else:
        print(colored("Parametro invalido escoja otro", 'yellow'))
        return

    #Generar funciones miembro
    pr_low = fuzz.trimf(x_data, [-0.5, 0, 0.5])
    pr_med = fuzz.trimf(x_data, [0, 0.5, 1])
    pr_hi = fuzz.trimf(x_data, [0.5, 1, 1.5])

    # Visualizar universos y funciones miembros
    ax0 = graph.figure.add_subplot(111)

    ax0.plot(x_data, pr_low, 'b', linewidth=1.5, label='Bajo')
    ax0.plot(x_data, pr_med, 'g', linewidth=1.5, label='Medio')
    ax0.plot(x_data, pr_hi, 'r', linewidth=1.5, label='Alto')
    ax0.set_title(titulo_parametro)
    ax0.legend()

    graph.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    rbfc_window = Main()
    rbfc_window.setWindowTitle("Sistema de Clasificador Difuso (Basado en Reglas)")
    rbfc_window.show()
    sys.exit(app.exec_())

