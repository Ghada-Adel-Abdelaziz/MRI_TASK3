# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Plot.ui'
#
# Created by: PyQt5 UI code generator 5.9.2

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PlotForm(object):
    def setupUi(self, PlotForm):
        PlotForm.setObjectName("PlotForm")
        PlotForm.resize(680, 638)
        self.decayMx = PlotWidget(PlotForm)
        self.decayMx.setGeometry(QtCore.QRect(10, 30, 661, 161))
        self.decayMx.setObjectName("decayMx")
        self.decayMy = PlotWidget(PlotForm)
        self.decayMy.setGeometry(QtCore.QRect(10, 220, 661, 171))
        self.decayMy.setObjectName("decayMy")
        self.recoveryMz = PlotWidget(PlotForm)
        self.recoveryMz.setGeometry(QtCore.QRect(10, 430, 661, 161))
        self.recoveryMz.setObjectName("recoveryMz")
        self.Plot = QtWidgets.QPushButton(PlotForm)
        self.Plot.setGeometry(QtCore.QRect(530, 600, 131, 31))
        self.Plot.setObjectName("Plot")
        self.rotationAngle = QtWidgets.QLineEdit(PlotForm)
        self.rotationAngle.setGeometry(QtCore.QRect(10, 600, 151, 31))
        self.rotationAngle.setObjectName("rotationAngle")
        self.DecayInMx = QtWidgets.QLabel(PlotForm)
        self.DecayInMx.setGeometry(QtCore.QRect(260, 10, 131, 20))
        self.DecayInMx.setAlignment(QtCore.Qt.AlignCenter)
        self.DecayInMx.setObjectName("DecayInMx")
        self.DecayInMx_2 = QtWidgets.QLabel(PlotForm)
        self.DecayInMx_2.setGeometry(QtCore.QRect(260, 200, 131, 20))
        self.DecayInMx_2.setAlignment(QtCore.Qt.AlignCenter)
        self.DecayInMx_2.setObjectName("DecayInMx_2")
        self.DecayInMx_3 = QtWidgets.QLabel(PlotForm)
        self.DecayInMx_3.setGeometry(QtCore.QRect(250, 400, 131, 20))
        self.DecayInMx_3.setAlignment(QtCore.Qt.AlignCenter)
        self.DecayInMx_3.setObjectName("DecayInMx_3")
        self.tr = QtWidgets.QLineEdit(PlotForm)
        self.tr.setGeometry(QtCore.QRect(350, 600, 151, 31))
        self.tr.setObjectName("tr")
        self.te = QtWidgets.QLineEdit(PlotForm)
        self.te.setGeometry(QtCore.QRect(180, 600, 151, 31))
        self.te.setObjectName("te")

        self.retranslateUi(PlotForm)
        QtCore.QMetaObject.connectSlotsByName(PlotForm)

    def retranslateUi(self, PlotForm):
        _translate = QtCore.QCoreApplication.translate
        PlotForm.setWindowTitle(_translate("PlotForm", "Form"))
        self.Plot.setText(_translate("PlotForm", "PushButton"))
        self.rotationAngle.setText(_translate("PlotForm", "90"))
        self.DecayInMx.setText(_translate("PlotForm", "Decay In Mx"))
        self.DecayInMx_2.setText(_translate("PlotForm", "Decay In My"))
        self.DecayInMx_3.setText(_translate("PlotForm", "Recovery In Z"))
        self.tr.setText(_translate("PlotForm", "1"))
        self.te.setText(_translate("PlotForm", "0.4"))

from pyqtgraph import PlotWidget
