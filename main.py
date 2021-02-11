import classifier
from random import randint, random, uniform
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

# Parametros de entrada

# Parametros Plataforma Juego
parametro1_arcade = ["Hongo / Tortuga", "Planta", "Dragon", "None"]  # Enemigo
parametro2_arcade = ["Corto", "Medio", "Largo", "None"]  # Hueco
parametro3_arcade = ["Tubo", "Bloque", "Muro", "None"]  # Obstáculo
parametro4_arcade = ["Fuego", "Estrella", "Nube", "None"]  # Armas de Jugador

# Parametros calculo Juego
parametro1_calc = ["Suma / Resta", "Multiplicacion", "Division", "None"]  # Operacion
parametro2_calc = ["Baja", "Media", "Alta", "None"]  # Puntaje
parametro3_calc = ["Simple",  "Intermedio", "Complejo", "None"]  # Dificultad
parametro4_calc = [ "Casi Nulo", "Incompleto", "Completo", "None"]  # Progreso

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
        self.entrada = []
        self.salida = []
        self.w = None
        super(self.__class__, self).__init__()
        self.setupUi(self)  # create the GUI

        # GUI Connections
        self.browse_file_button.clicked.connect(self.browse_button_clicked)
        self.trainSlider.valueChanged.connect(self.train_to_test_ratio_slider_value_changed)
        self.train_spin_box.valueChanged.connect(self.train_to_test_ratio_spinbox_value_changed)
        self.train_button.clicked.connect(self.train_button_clicked)
        self.mutation_box.valueChanged.connect(self.mutation_value_changed)
        self.poblation_box.valueChanged.connect(self.poblation_value_changed)
        self.generation_box.valueChanged.connect(self.generation_value_changed)
        self.tourney_box.valueChanged.connect(self.tournament_value_changed)
        self.classify_button.clicked.connect(self.classify_button_clicked)
        self.rules_button.clicked.connect(self.rules_button_clicked)
        self.plot_button.clicked.connect(self.plot_button_clicked)
        self.act_button.clicked.connect(self.act_button_clicked)
        if self.genre_box.currentTextChanged:
            self.genre_box.currentTextChanged.connect(self.genre_box_changed)
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


            # Obtener nombres de parametros de entrada
            for col in df.columns:
                if col == 'Clase':
                    continue
                else:
                    self.entrada.append(col)

            # Obtener nombres de parametros de salida
            for class_name in df['Clase'].unique():
                self.salida.append(class_name)

            # Normalizar los 4 parametros
            df.iloc[:, 0:4] = df.iloc[:, 0:4].apply(lambda x: x / np.max(x))

            X = df.iloc[:, 0:4]

            # Cambiar Clase1 a 0, Clase2 a 1 y Clase3 a 2
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

            # Cambiar informacion de combo_box
            self.combo_box_parametros.clear()
            self.combo_box_parametros.addItems(self.entrada)

            # Activar botones
            self.classify_button.setEnabled(True)
            self.generation_box.setEnabled(True)
            self.poblation_box.setEnabled(True)
            self.tourney_box.setEnabled(True)
            self.mutation_box.setEnabled(True)
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
        self.act_button.setEnabled(True)
        self.combo_box_parametros.setEnabled(True)
        self.genre_box.setEnabled(True)

    def rules_button_clicked(self):
        s = ""
        subclase = []

        if self.genre_box.currentText() == "Plataforma":
            subclase.append(parametro1_arcade)
            subclase.append(parametro2_arcade)
            subclase.append(parametro3_arcade)
            subclase.append(parametro4_arcade)
            reglas = binRulestoClassRules(self.rules, self.output, self.entrada, self.salida, subclase)
        elif self.genre_box.currentText() == "Calculo":
            subclase.append(parametro1_calc)
            subclase.append(parametro2_calc)
            subclase.append(parametro3_calc)
            subclase.append(parametro4_calc)
            reglas = binRulestoClassRules(self.rules, self.output, self.entrada, self.salida, subclase)
        # salidas = intOutputtoClassOutput(self.output)
        print(colored("Reglas del sistema", 'magenta'))
        for i in range(len(reglas)):
            s += "Regla " + str(i + 1) + ": If " + str(reglas[i]) + "\n"
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
        if self.combo_box_parametros.currentText() == "Enemigo":
            plot_memberhip_function(graph=self.w, parametro=self.combo_box_parametros.currentText(), subclase=parametro1_arcade)
        elif self.combo_box_parametros.currentText() == "Hueco":
            plot_memberhip_function(graph=self.w, parametro=self.combo_box_parametros.currentText(), subclase=parametro2_arcade)
        elif self.combo_box_parametros.currentText() == "Obstaculo":
            plot_memberhip_function(graph=self.w, parametro=self.combo_box_parametros.currentText(), subclase=parametro3_arcade)
        elif self.combo_box_parametros.currentText() == "Arma":
            plot_memberhip_function(graph=self.w, parametro=self.combo_box_parametros.currentText(), subclase=parametro4_arcade)

    def act_button_clicked(self):
        valor_1 = None
        valor_2 = None
        valor_3 = None
        valor_4 = None
        if self.genre_box.currentText() == "Plataforma":
            valor_1 = getSubClassValue(self.parameter1_box.currentText(), parametro1_arcade)
            valor_2 = getSubClassValue(self.parameter2_box.currentText(), parametro2_arcade)
            valor_3 = getSubClassValue(self.parameter3_box.currentText(), parametro3_arcade)
            valor_4 = getSubClassValue(self.parameter4_box.currentText(), parametro4_arcade)

        elif self.genre_box.currentText() == "Calculo":
            valor_1 = getSubClassValue(self.parameter1_box.currentText(), parametro1_calc)
            valor_2 = getSubClassValue(self.parameter2_box.currentText(), parametro2_calc)
            valor_3 = getSubClassValue(self.parameter3_box.currentText(), parametro3_calc)
            valor_4 = getSubClassValue(self.parameter4_box.currentText(), parametro4_calc)
        elif self.genre_box.currentText() == "Escoja":
            pass
        else:
            print(colored("Por favor escoja el género del videojuego y los parametros de las subclase que pertenecen", "yellow"))

        print(colored("Valores de los parametros para activar las reglas", "red"))
        print(colored(valor_1, "red"))
        print(colored(valor_2, "red"))
        print(colored(valor_3, "red"))
        print(colored(valor_4, "red"))

    def genre_box_changed(self):

        if self.genre_box.currentText() == "Plataforma":
            self.parameter1_box.clear()
            self.parameter2_box.clear()
            self.parameter3_box.clear()
            self.parameter4_box.clear()
            self.combo_box_parametros.clear()
            self.parameter1_box.addItems(parametro1_arcade)
            self.parameter2_box.addItems(parametro2_arcade)
            self.parameter3_box.addItems(parametro3_arcade)
            self.parameter4_box.addItems(parametro4_arcade)
            self.combo_box_parametros.addItems(self.entrada)
        elif self.genre_box.currentText() == "Calculo":
            self.parameter1_box.clear()
            self.parameter2_box.clear()
            self.parameter3_box.clear()
            self.parameter4_box.clear()
            self.combo_box_parametros.clear()
            self.parameter1_box.addItems(parametro1_calc)
            self.parameter2_box.addItems(parametro2_calc)
            self.parameter3_box.addItems(parametro3_calc)
            self.parameter4_box.addItems(parametro4_calc)
            self.combo_box_parametros.addItems(self.entrada)
        elif self.genre_box.currentText() == "Escoja":
            pass
        else:
            print(colored("Opcion invalida, escoja otra", "yellow"))

        self.parameter1_box.setEnabled(True)
        self.parameter2_box.setEnabled(True)
        self.parameter3_box.setEnabled(True)
        self.parameter4_box.setEnabled(True)

    def combo_box_changed(self):
        self.plot_button.setEnabled(True)

    def train_to_test_ratio_slider_value_changed(self):
        # change the ratio of train/test samples both in variable and in spinner
        self.train_spin_box.setValue(self.trainSlider.value())
        self.train_ratio = round(1 - self.trainSlider.value() / 100, 2)

    def train_to_test_ratio_spinbox_value_changed(self):
        # change the ratio of train/test samples both in variable and in slider
        self.trainSlider.setValue(self.train_spin_box.value())
        self.train_ratio = round(1 - self.trainSlider.value() / 100, 2)

    def mutation_value_changed(self):
        # change the poblation number value in variable
        global mutProb
        mutProb = self.poblation_box.value()

    def poblation_value_changed(self):
        # change the poblation number value in variable
        global popSize
        popSize = self.poblation_box.value()

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


# Insertar datos de la sub clase correpondiente a la clase de la regla
def getRulesSubClass(rule, i, ruleclass, subclass):
    rule_text = ""
    if not (rule[i] == 0) and not ((rule[i + 1] == 0) or (rule[i + 2] == 0)):
        rule_text += ruleclass + " = " + subclass[0] + " or "
    elif not (rule[i] == 0) and (rule[i + 1] == 0) and (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass[0] + " and "
    if not (rule[i + 1] == 0) and not (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass[1] + " or "
    elif not (rule[i + 1] == 0) and (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass[1] + " and "
    if not (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass[2] + " and "
    return rule_text


# Transformar reglas binarias en reglas literales
def binRulestoClassRules (rules, outputs, entrada, salida, subclase):
    soltext = []
    count = 0
    for rule in rules:
        rule_text = ""
        textosalida = ""
        for i in range(0, len(rule), 3):
            if i < 2:
                rule_text += getRulesSubClass(rule, i, entrada[0], subclase[0])
            elif 2 <= i < 5:
                rule_text += getRulesSubClass(rule, i, entrada[1], subclase[1])
            elif 5 <= i < 8:
                rule_text += getRulesSubClass(rule, i, entrada[2], subclase[2])
            elif 8 <= i < 11:
                rule_text += getRulesSubClass(rule, i, entrada[3], subclase[3])
        rule_text_2 = rule_text.rstrip('and ')
        # Output process
        tupla = outputs[count]
        if tupla[0] == 0:
            textosalida += salida[0]
        elif tupla[0] == 1:
            textosalida += salida[1]
        elif tupla[0] == 2:
            textosalida += salida[2]
        else:
            textosalida += "Not Valid"
        rule_text_2 += " entonces " + textosalida
        soltext.append(rule_text_2)
        count += 1
    return soltext


# Obtener valor de parametros de subclase
def getSubClassValue(parametro_box, subclase):
    if parametro_box == subclase[0]:
        return round(uniform(0, 0.25), 3)
    elif parametro_box == subclase[1]:
        return round(uniform(0.25, 0.75), 3)
    elif parametro_box == subclase[2]:
        return round(uniform(0.75, 1), 3)
    else:
        return 0


def plot_memberhip_function(graph, parametro, subclase):

    # Variables universales
    x_data = np.arange(0, 1.0001, 0.0001)
    # Generar funciones miembro
    pr_low = fuzz.trimf(x_data, [-0.5, 0, 0.5])
    pr_med = fuzz.trimf(x_data, [0, 0.5, 1])
    pr_hi = fuzz.trimf(x_data, [0.5, 1, 1.5])

    # Visualizar universos y funciones miembros
    ax0 = graph.figure.add_subplot(111)

    ax0.plot(x_data, pr_low, 'b', linewidth=1.5, label=subclase[0])
    ax0.plot(x_data, pr_med, 'g', linewidth=1.5, label=subclase[1])
    ax0.plot(x_data, pr_hi, 'r', linewidth=1.5, label=subclase[2])
    ax0.set_title(parametro)
    ax0.legend()

    graph.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    rbfc_window = Main()
    rbfc_window.setWindowTitle("Sistema de Clasificador Difuso (Basado en Reglas)")
    rbfc_window.show()
    sys.exit(app.exec_())

