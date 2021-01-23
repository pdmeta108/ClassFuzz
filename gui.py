# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fuzzyui.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(620, 650)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(2000, 2000))
        MainWindow.setStyleSheet("background-color: rgb(211,211,211);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(50, 10, 511, 544))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.gridLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.browse_file_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.browse_file_label.setFont(font)
        self.browse_file_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.browse_file_label.setObjectName("browse_file_label")
        self.verticalLayout.addWidget(self.browse_file_label)
        self.horizontalLayout1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout1.setSpacing(1)
        self.horizontalLayout1.setObjectName("horizontalLayout1")
        self.browse_file_text_edit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browse_file_text_edit.sizePolicy().hasHeightForWidth())
        self.browse_file_text_edit.setSizePolicy(sizePolicy)
        self.browse_file_text_edit.setStyleSheet("background-color: rgb(255,255,255); margin-right: 20px")
        self.browse_file_text_edit.setObjectName("browse_file_text_edit")
        self.horizontalLayout1.addWidget(self.browse_file_text_edit)
        self.browse_file_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browse_file_button.sizePolicy().hasHeightForWidth())
        self.browse_file_button.setSizePolicy(sizePolicy)
        self.browse_file_button.setStyleSheet("background-color: rgb(161,161,161); padding: 4px 10px;")
        self.browse_file_button.setDefault(False)
        self.browse_file_button.setFlat(False)
        self.browse_file_button.setObjectName("browse_file_button")
        self.horizontalLayout1.addWidget(self.browse_file_button)
        self.verticalLayout.addLayout(self.horizontalLayout1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.train_percentage_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.train_percentage_label.setFont(font)
        self.train_percentage_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.train_percentage_label.setObjectName("train_percentage_label")
        self.verticalLayout.addWidget(self.train_percentage_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.trainSlider = QtWidgets.QSlider(self.gridLayoutWidget)
        self.trainSlider.setEnabled(False)
        self.trainSlider.setOrientation(QtCore.Qt.Horizontal)
        self.trainSlider.setObjectName("trainSlider")
        self.horizontalLayout.addWidget(self.trainSlider)
        self.train_spin_box = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.train_spin_box.setEnabled(False)
        self.train_spin_box.setStyleSheet("margin-left:20px")
        self.train_spin_box.setObjectName("train_spin_box")
        self.horizontalLayout.addWidget(self.train_spin_box)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem2)
        self.data_train_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.data_train_label.setFont(font)
        self.data_train_label.setStyleSheet("margin-bottom: 5px")
        self.data_train_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.data_train_label.setObjectName("data_train_label")
        self.verticalLayout.addWidget(self.data_train_label)
        self.train_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.train_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.train_button.sizePolicy().hasHeightForWidth())
        self.train_button.setSizePolicy(sizePolicy)
        self.train_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.train_button.setStyleSheet("background-color: rgb(161,161,161); padding: 4px 10px; margin: 0 auto;")
        self.train_button.setObjectName("train_button")
        self.verticalLayout.addWidget(self.train_button)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem3)
        self.classify_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.classify_label.setFont(font)
        self.classify_label.setStyleSheet("")
        self.classify_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.classify_label.setObjectName("classify_label")
        self.verticalLayout.addWidget(self.classify_label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.poblation_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.poblation_label.setStyleSheet("margin: 0 auto;")
        self.poblation_label.setObjectName("poblation_label")
        self.horizontalLayout_2.addWidget(self.poblation_label)
        self.generation_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.generation_label.setStyleSheet("margin: 0 auto;")
        self.generation_label.setObjectName("generation_label")
        self.horizontalLayout_2.addWidget(self.generation_label)
        self.tourney_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.tourney_label.setStyleSheet("margin: 0 auto;")
        self.tourney_label.setObjectName("tourney_label")
        self.horizontalLayout_2.addWidget(self.tourney_label)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.poblation_box = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.poblation_box.setEnabled(False)
        self.poblation_box.setAutoFillBackground(False)
        self.poblation_box.setStyleSheet("background-color: rgb(255); margin: 10 auto;")
        self.poblation_box.setMinimum(1)
        self.poblation_box.setProperty("value", 20)
        self.poblation_box.setObjectName("poblation_box")
        self.horizontalLayout_4.addWidget(self.poblation_box)
        self.generation_box = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.generation_box.setEnabled(False)
        self.generation_box.setAutoFillBackground(False)
        self.generation_box.setStyleSheet("background-color: rgb(255); margin: 10 auto;")
        self.generation_box.setMinimum(1)
        self.generation_box.setProperty("value", 20)
        self.generation_box.setObjectName("generation_box")
        self.horizontalLayout_4.addWidget(self.generation_box)
        self.tourney_box = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.tourney_box.setEnabled(False)
        self.tourney_box.setAutoFillBackground(False)
        self.tourney_box.setStyleSheet("background-color: rgb(255); margin: 10 auto;")
        self.tourney_box.setMinimum(2)
        self.tourney_box.setMaximum(10)
        self.tourney_box.setProperty("value", 5)
        self.tourney_box.setObjectName("tourney_box")
        self.horizontalLayout_4.addWidget(self.tourney_box)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.classify_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.classify_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classify_button.sizePolicy().hasHeightForWidth())
        self.classify_button.setSizePolicy(sizePolicy)
        self.classify_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.classify_button.setStyleSheet("background-color: rgb(161,161,161); padding: 4px 10px; margin: 0 auto;")
        self.classify_button.setObjectName("classify_button")
        self.verticalLayout.addWidget(self.classify_button)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.plot_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.plot_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_button.sizePolicy().hasHeightForWidth())
        self.plot_button.setSizePolicy(sizePolicy)
        self.plot_button.setStyleSheet("background-color: rgb(161,161,161); padding: 4px 10px;")
        self.plot_button.setObjectName("plot_button")
        self.horizontalLayout_3.addWidget(self.plot_button)
        self.parametros_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.parametros_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.parametros_label.setStyleSheet("margin: 0 auto;")
        self.parametros_label.setAlignment(QtCore.Qt.AlignCenter)
        self.parametros_label.setObjectName("parametros_label")
        self.horizontalLayout_3.addWidget(self.parametros_label)
        self.combo_box_parametros = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.combo_box_parametros.setEnabled(False)
        self.combo_box_parametros.setStyleSheet("background-color: rgb(255); margin: 10 auto;")
        self.combo_box_parametros.setObjectName("combo_box_parametros")
        self.combo_box_parametros.addItem("")
        self.combo_box_parametros.addItem("")
        self.combo_box_parametros.addItem("")
        self.combo_box_parametros.addItem("")
        self.combo_box_parametros.addItem("")
        self.horizontalLayout_3.addWidget(self.combo_box_parametros)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem5)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.rules_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.rules_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rules_button.sizePolicy().hasHeightForWidth())
        self.rules_button.setSizePolicy(sizePolicy)
        self.rules_button.setStyleSheet("background-color: rgb(161,161,161); padding: 4px 10px;")
        self.rules_button.setObjectName("rules_button")
        self.horizontalLayout_5.addWidget(self.rules_button)
        self.sample_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.sample_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sample_button.sizePolicy().hasHeightForWidth())
        self.sample_button.setSizePolicy(sizePolicy)
        self.sample_button.setStyleSheet("background-color: rgb(161,161,161); padding: 4px 10px;")
        self.sample_button.setObjectName("sample_button")
        self.horizontalLayout_5.addWidget(self.sample_button)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 620, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.browse_file_label.setText(_translate("MainWindow", "Introducir archivo"))
        self.browse_file_text_edit.setPlaceholderText(_translate("MainWindow", "Introduzca la ubicacion del archivo"))
        self.browse_file_button.setText(_translate("MainWindow", "Examinar"))
        self.train_percentage_label.setText(_translate("MainWindow", "Porcentaje Entrenar"))
        self.data_train_label.setText(_translate("MainWindow", "Entrenar Datos"))
        self.train_button.setText(_translate("MainWindow", "Comenzar Entrenamiento"))
        self.classify_label.setText(_translate("MainWindow", "Generar y Clasificar Reglas con Algoritmo Genetico"))
        self.poblation_label.setText(_translate("MainWindow", "Poblacion"))
        self.generation_label.setText(_translate("MainWindow", "Numero de Generaciones"))
        self.tourney_label.setText(_translate("MainWindow", "Tamaño de torneo"))
        self.classify_button.setText(_translate("MainWindow", "Generar y Clasificar Reglas"))
        self.plot_button.setText(_translate("MainWindow", "Grafica de Funciones Miembro"))
        self.parametros_label.setText(_translate("MainWindow", "Parametros"))
        self.combo_box_parametros.setItemText(0, _translate("MainWindow", "Escoja"))
        self.combo_box_parametros.setItemText(1, _translate("MainWindow", "SepalLargo"))
        self.combo_box_parametros.setItemText(2, _translate("MainWindow", "SepalAncho"))
        self.combo_box_parametros.setItemText(3, _translate("MainWindow", "PetalLargo"))
        self.combo_box_parametros.setItemText(4, _translate("MainWindow", "PetalAncho"))
        self.rules_button.setText(_translate("MainWindow", "Reglas de Sistema"))
        self.sample_button.setText(_translate("MainWindow", "Muestra"))