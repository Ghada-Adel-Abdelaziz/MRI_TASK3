import sys
import math
from gui import Ui_PlotForm
import traceback
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QColor, QBrush, QPainter, QPen, QDragEnterEvent
from PyQt5.QtCore import Qt, QThread
from task3_gui import Ui_MainWindow
from PIL import Image, ImageEnhance
from imageio import imsave, imread
import scipy.io as sio
import io
from time import sleep
import pyqtgraph as pg


class window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.time = np.arange(0, 1, 0.001)  # in sec but step in 1 msec
        self.vector = np.matrix([0, 0, 1])  # da range sabt

        # self.DecayMx = self.ui.decayMx
        # self.RecoveryMz = self.ui.recoveryMz

        self.DecayMx = self.ui.decayMx
        self.RecoveryMz = self.ui.recoveryMz
        self.ui.Reconstruction.clicked.connect(self.Reconstruction)
        self.ui.browse.clicked.connect(self.init)
        self.show()
        self.paint = False
        self.points = QtGui.QPolygon()
        self.x = None
        self.y = None
        self.count = -1
        self.text = '512'
        self.text2 = 'Proton Density'
        self.T1 = np.zeros((512, 512))
        self.T2 = np.zeros((512, 512))
        self.error_dialog = None
        self.brit = 0
        self.left = False
        self.right = False
        self.coord = []
        self.inten = []
        self.plotCol = []

    def getText2(self, index):
        self.text2 = self.ui.ImageChange.itemText(index)
        self.changePic()

    def getText(self, index):
        self.text = self.ui.comboBox.itemText(index)
        self.changePic()

    def StartThred(self):
        self.thread = QThread()
        self.thread.started.connect(self.Reconstruction)
        self.thread.start()

    def changePic(self):
        print(self.text, self.text2)
        if self.text == '512' and self.text2 == 'Proton Density':
            self.pixmap = QtGui.QPixmap(self.fileName0)
            self.pixmap = self.pixmap.scaled(512, 512)
            self.siz = 512
        if self.text == '120' and self.text2 == 'Proton Density':
            self.pixmap = QtGui.QPixmap(self.fileName0)
            self.pixmap = self.pixmap.scaled(120, 120)
            self.siz = 120
        if self.text == '512' and self.text2 == 'T1':
            self.pixmap = QtGui.QPixmap(self.fileName2)
            self.pixmap = self.pixmap.scaled(512, 512)
            self.siz = 512
        if self.text == '120' and self.text2 == 'T1':
            self.pixmap = QtGui.QPixmap(self.fileName2)
            self.pixmap = self.pixmap.scaled(120, 120)
            self.siz = 120
        if self.text == '512' and self.text2 == 'T2':
            self.pixmap = QtGui.QPixmap(self.fileName3)
            self.pixmap = self.pixmap.scaled(512, 512)
            self.siz = 512
        if self.text == '120' and self.text2 == 'T2':
            self.pixmap = QtGui.QPixmap(self.fileName3)
            self.pixmap = self.pixmap.scaled(120, 120)
            self.siz = 120

    def clearGraphicView(self):
        self.ui.decayMx.clear()
        self.ui.recoveryMz.clear()

    def showError(self):
        self.error_dialog = QtWidgets.QErrorMessage()
        self.error_dialog.showMessage("Please select a proper phantom")

    def init(self):
        self.setImage()
        self.coord = []
        self.inten = []
        self.ui.image.point = []
        self.clearGraphicView()
        self.count = -1
        self.brit = 0
        self.ui.FourierMatrix.clear()
        self.ui.Constructed.clear()

    def setImage(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                                 "Image Files (*.png *.jpg *jpeg *.bmp *.mat)")  # Ask for file
        if self.fileName:  # If the user gives a file
            if self.fileName.split(".")[1] != "mat":
                return self.showError()
            print(self.fileName)
            output = sio.loadmat(self.fileName)
            self.img = output['Phantom']
            self.pahntomForLoop = self.img
            self.imgT1 = output['T1']
            self.imgT2 = output['T2']
            imsave("phantom.png", self.img)
            imsave("T1.png", self.imgT1)
            imsave("T2.png", self.imgT2)
            self.fileName0 = 'phantom.png'
            self.fileName2 = "T1.png"
            self.fileName3 = "T2.png"
            self.ui.image.setMouseTracking(False)
            self.ui.image.mousePressEvent = self.getPixel
            self.ui.image.mouseMoveEvent = self.changeCont
            self.ui.rotationAngle.textChanged.connect((self.angleChan))
            self.paint = True
            self.ui.comboBox.activated.connect(self.getText)
            self.ui.ImageChange.activated.connect(self.getText2)
            self.pixmap = QtGui.QPixmap(self.fileName0)
            self.pixmap = self.pixmap.scaled(512, 512)
            self.siz = 512
            self.ui.image.setPixmap(self.pixmap)

    def createPD(self, intensity):
        return (1 / 255) * intensity

    def angleChan(self):
        self.clearGraphicView()
        self.count = -1
        for i in self.inten:
            self.T1 = self.createT1(i)
            self.T2 = self.createT2(i)
            self.PD = self.createPD(i)
            self.count += 1
            self.plot()

    def changeCont(self, event):
        if self.ui.checkBox.isChecked():
            if self.left:
                image1 = Image.open(self.fileName0)
                enhancer = ImageEnhance.Brightness(image1)
                self.brit += 0.01
                enhanced_img = enhancer.enhance(self.brit)
                enhanced_img.save('enhanced_img.png')
                self.fileName4 = 'enhanced_img.png'
                self.pixmap5 = QtGui.QPixmap(self.fileName4)
                self.pixmap5 = self.pixmap5.scaled(self.siz, self.siz)
                self.ui.image.setPixmap(self.pixmap5)
                print('sa7')
            if self.right:
                image1 = Image.open(self.fileName0)
                enhancer = ImageEnhance.Brightness(image1)
                self.brit -= 0.05
                enhanced_img = enhancer.enhance(self.brit)
                enhanced_img.save('enhanced_img.png')
                self.fileName4 = 'enhanced_img.png'
                self.pixmap5 = QtGui.QPixmap(self.fileName4)
                self.pixmap5 = self.pixmap5.scaled(self.siz, self.siz)
                self.ui.image.setPixmap(self.pixmap5)

    def mousePressEvent(self, e):
        self.points << e.pos()
        self.update()

    def passCord(self):
        self.ui.image.point.append([self.x, self.y, self.pen[self.count]])
        self.ui.image.x = self.ui.image.frameGeometry().width()
        self.ui.image.y = self.ui.image.frameGeometry().height()
        print(self.ui.image.point, self.coord)

    def getPixel(self, event):
        if event.button() == Qt.LeftButton:
            self.left = True
            self.right = False
        if event.button() == Qt.RightButton:  # for contrast left and right button
            self.right = True
            self.left = False
        if event.button() == Qt.LeftButton:
            if not self.ui.checkBox.isChecked():
                if self.text == '512':
                    self.x = event.pos().x()
                    self.y = event.pos().y()
                    self.x2 = self.x * (self.img.shape[0] / 512)
                    self.y2 = self.y * (self.img.shape[0] / 512)
                    self.x2 = math.floor(self.x2)
                    self.y2 = math.floor(self.y2)
                    print(self.x2, self.y2)
                    self.T1 = self.createT1(self.imgT1[self.x2, self.y2])
                    self.T2 = self.createT2(self.imgT2[self.x2, self.y2])
                    self.PD = self.createPD(self.img[self.x2, self.y2])
                    self.coord.append([self.x, self.y])
                if self.text == '120':
                    self.x = event.pos().x()
                    self.y = event.pos().y()
                    self.x1 = self.x * (512 / self.x2)
                    self.x1 = math.floor(self.x1)
                    self.y1 = self.y * (512 / self.y2)
                    self.y1 = math.floor(self.y1)
                    self.x2 = self.x1 * (self.img.shape[0] / 512)
                    self.y2 = self.y1 * (self.img.shape[0] / 512)
                    self.x2 = math.floor(self.x2)
                    self.y2 = math.floor(self.y2)
                    self.T1 = self.createT1(self.imgT1[self.x2, self.y2])
                    self.T2 = self.createT2(self.imgT2[self.x2, self.y2])
                    self.PD = self.createPD(self.img[self.x2, self.y2])
                    self.coord.append([self.x1, self.y1])
                self.count += 1
                if self.count == 4:
                    self.count = -1
                self.pen = [QtGui.QPen(QtCore.Qt.green), QtGui.QPen(QtCore.Qt.red), QtGui.QPen(QtCore.Qt.yellow),
                            QtGui.QPen(QtCore.Qt.blue),
                            QtGui.QPen(QtCore.Qt.cyan)]
                self.plotCol = [pg.mkPen('g'), pg.mkPen('r'), pg.mkPen('y'), pg.mkPen('b'), pg.mkPen('b'),
                                pg.mkPen('c')]
                self.plot()
                self.passCord()
                self.inten.append(self.img[self.x2, self.y2])

    def optFrame(self):
        if self.ui.image.x != self.ui.image.frameGeometry().width() or self.ui.image.y != self.ui.image.frameGeometry().height():
            self.ui.image.x = self.ui.image.frameGeometry().width()
            self.ui.image.y = self.ui.image.frameGeometry().height()
            k = 0
            for i in self.coord:
                x = self.coord[k][0]
                y = self.coord[k][1]
                self.ui.image.point[k][0] = x / (512 / self.ui.image.frameGeometry().width())
                self.ui.image.point[k][1] = y / (512 / self.ui.image.frameGeometry().height())
                print(self.ui.image.point[k][0], self.ui.image.point[k][1], self.ui.image.frameGeometry().width(),
                      self.ui.image.frameGeometry().height())
                k += 1

    def paintEvent(self, event):
        if self.paint and self.count == -1 and not self.ui.checkBox.isChecked():
            self.x2 = self.ui.image.frameGeometry().width()
            self.y2 = self.ui.image.frameGeometry().height()
            self.pixmap0 = self.pixmap
            self.ui.image.setPixmap(self.pixmap0)  # Set the pixmap onto the label
            # self.ui.image.adjustSize()
            self.ui.image.setScaledContents(True)
            self.ui.image.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.image.show()

        if self.paint and self.count >= -1 and not self.ui.checkBox.isChecked():
            self.x2 = self.ui.image.frameGeometry().width()
            self.y2 = self.ui.image.frameGeometry().height()
            self.optFrame()
            self.ui.image.paint = True
            self.pixmap0 = self.pixmap
            self.ui.image.setPixmap(self.pixmap0)  # Set the pixmap onto the label
            self.ui.image.setScaledContents(True)
            self.ui.image.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.image.show()
            self.paint1 = True
            if self.right:
                self.count = -1
                self.ui.image.point = []
                self.coord = []
                self.inten = []
                if self.text == '120' and self.text2 == 'Proton Density':
                    self.pixmap = QtGui.QPixmap(self.fileName0)
                    self.pixmap = self.pixmap.scaled(120, 120)
                if self.text == '512' and self.text2 == 'Proton Density':
                    self.pixmap = QtGui.QPixmap(self.fileName0)
                    self.pixmap = self.pixmap.scaled(512, 512)
                self.ui.image.setPixmap(self.pixmap)  # Set the pixmap onto the label
                self.ui.image.setScaledContents(True)
                self.ui.image.setAlignment(QtCore.Qt.AlignCenter)
                self.ui.image.show()
                self.clearGraphicView()

            # self.paint4 = False

    def plot(self):

        self.DecayMx = self.ui.decayMx
        self.RecoveryMz = self.ui.recoveryMz

        self.theta = ((float)(self.ui.rotationAngle.text()))  # 5ly balk not global
        self.Tr = ((float)(self.ui.tr.text()))
        self.Te = ((float)(self.ui.te.text()))

        self.Mx = []
        self.Mz = []
        self.vector = np.matrix([0, 0, 1])  # da range sabt

        self.vector = self.rotationAroundYaxisMatrix(self.theta, self.vector)

        for i in range(len(self.time)):
            # self.vector = self.rotationAroundZaxisMatrixXY(self.Tr,speed,self.vector,self.time[i])
            self.vector = self.recoveryDecayEquation(self.T1, self.T2, 1, self.vector, self.time[i])

            self.Mx = np.append(self.Mx, self.vector.item(0))
            self.Mz = np.append(self.Mz, self.vector.item(2))

        self.DecayMx.plot(self.time, np.ravel(self.Mx), pen=self.plotCol[self.count])
        self.RecoveryMz.plot(self.time, np.ravel(self.Mz), pen=self.plotCol[self.count])

        self.RecoveryMz.addLine(x=self.Tr)
        self.RecoveryMz.addLine(x=self.Te)
        self.DecayMx.addLine(x=self.Tr)
        self.DecayMx.addLine(x=self.Te)

    def createT2(self, intensity):

        if intensity == 100:  # Gray matter
            T2 = 170
        elif intensity == 255:  # white matter
            T2 = 150
        elif intensity == 200:  # muscle
            T2 = 50
        elif intensity == 120:  # fat
            T2 = 100
        elif intensity == 25:  # protein
            T2 = 10
        elif intensity == 150:
            T2 = 255

        elif intensity > 1:  # Black => air
            T2 = (0.5 * intensity) + 20

        elif intensity <= 1:
            T2 = (intensity * 100) + 10

        return T2

    def createT1(self, intensity):

        if intensity == 100:  # Gray matter
            T1 = 255

        elif intensity == 255:  # white matter
            T1 = 100

        elif intensity == 200:  # muscle
            T1 = 180

        elif intensity == 120:  # fat
            T1 = 200

        elif intensity == 25:  # protein
            T1 = 255

        elif intensity == 0:  # Black => air
            T1 = 1

        elif intensity > 1:  # Black => air
            T1 = round((7.5 * intensity) + 50)

        else:
            T1 = (intensity * 100) + ((intensity * 100) + 50)
        return T1

    def returnIntensityfromProtonDensity(self, Pd):  # proton intensity vales from 0 till 1
        return 255 * Pd

    def rotationAroundYaxisMatrix(self, theta, vector):
        vector = vector.transpose()
        theta = (math.pi / 180) * theta
        R = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R = np.dot(R, vector)
        R = R.transpose()
        return np.matrix(R)

    def rotationAroundZaxisMatrixXY(self, TR, speed, vector, time):  # time = self.time
        vector = vector.transpose()
        theta = speed * (time / TR)
        theta = (math.pi / 180) * theta
        XY = np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        XY = np.dot(XY, vector)
        XY = XY.transpose()
        return np.matrix(XY)

    def recoveryDecayEquation(self, T1, T2, PD, vector, time):
        vector = vector.transpose()
        Decay = np.matrix([[np.exp(-time / T2), 0, 0], [0, np.exp(-time / T2), 0], [0, 0, np.exp(-time / T1)]])
        Decay = np.dot(Decay, vector)

        Rec = np.dot(np.matrix([[0, 0, (1 - (np.exp(-time / T1)))]]), PD)
        Rec = Rec.transpose()
        Decay = np.matrix(Decay)
        Rec = np.matrix(Rec)

        RD = Decay + Rec
        RD = RD.transpose()
        return RD

    def Reconstruction(self):
        theta = ((float)(self.ui.flipAngle.text()))  # 5ly balk not global
        Tr = ((float)(self.ui.TR.text()))
        Te = ((float)(self.ui.TE.text()))
        vector = np.matrix([0, 0, 1])
        Kspace = np.zeros((self.pahntomForLoop.shape[0], self.pahntomForLoop.shape[1]), dtype=np.complex_)
        # Kspace.fill(255)
        KspaceSave = abs(Kspace)
        # imsave("Kspace.png", KspaceSave)
        self.fileName5 = "Kspace_trajecrory.png"
        # self.ui.FourierMatrix.setPixmap(QtGui.QPixmap(self.fileName5).scaled(512,512))
        # self.ui.FourierMatrix.setPixmap(QtGui.QPixmap(self.fileName5))
        print(theta, Te, Tr)
        self.ForLoops(theta, Tr, Te, vector, Kspace)

    def ForLoops(self, theta, TR, TE, vector, Kspace):

        signal = [[[0 for k in range(3)] for j in range(self.pahntomForLoop.shape[0])] for i in
                  range(self.pahntomForLoop.shape[1])]
        print(signal)
        start = True

        for Ki in range(Kspace.shape[0]):
            print('Ki: ', Ki)
            # move in each image pixel
            if start:
                for i in range(self.pahntomForLoop.shape[0]):
                    for j in range(self.pahntomForLoop.shape[1]):
                        signal[i][j] = self.rotationAroundYaxisMatrix(theta, vector)
                        signal[i][j] = signal[i][j] * np.exp(-TE / self.createT2(self.pahntomForLoop[i, j]))
            else:
                for i in range(self.pahntomForLoop.shape[0]):
                    for j in range(self.pahntomForLoop.shape[1]):
                        signal[i][j] = self.rotationAroundYaxisMatrix(theta, np.matrix(signal[i][j]))
                        signal[i][j] = signal[i][j] * np.exp(-TE / self.createT2(self.pahntomForLoop[i, j]))

            # for kspace column
            for Kj in range(Kspace.shape[1]):
                print('Kj: ', Kj)
                GxStep = ((2 * math.pi) / Kspace.shape[0]) * Kj
                GyStep = ((2 * math.pi) / Kspace.shape[1]) * Ki

                for i in range(self.pahntomForLoop.shape[0]):
                    for j in range(self.pahntomForLoop.shape[1]):
                        totalTheta = (GxStep * j) + (GyStep * i)
                        z = abs(complex(np.ravel(signal[i][j])[0], np.ravel(signal[i][j])[1]))
                        Kspace[Ki, Kj] = Kspace[Ki, Kj] + (z * np.exp(1j * totalTheta))

            for i in range(self.pahntomForLoop.shape[0]):
                for j in range(self.pahntomForLoop.shape[1]):
                    signal[i][j] = self.rotationAroundYaxisMatrix(theta, vector)  # Trial
                    signal[i][j] = self.recoveryDecayEquation(self.createT1(self.pahntomForLoop[i, j]),
                                                              self.createT2(self.pahntomForLoop[i, j]), 1,
                                                              np.matrix(signal[i][j]), TR)
                    signal[i][j] = [[0, 0, np.ravel(signal[i][j])[2]]]
                    start = False

        print("DONE")
        KspaceSave = abs(Kspace)
        print('Hello')
        print(Kspace)
        imsave('kspace_trajectory.png', KspaceSave)
        print("DONE1")
        self.ui.FourierMatrix.setPixmap(QtGui.QPixmap('kspace_trajectory.png').scaled(512, 512))
        print("DONE2")
        Kspacefft = np.fft.fft2(Kspace)
        # Kspaceifft = np.fft.ifft2(Kspace)
        print("DONE3")
        Kspacefft = abs(Kspacefft)
        imsave("image.png", Kspacefft)
        print("DONE4")
        pixmap = QtGui.QPixmap("image.png")
        pixmap = pixmap.scaled(512, 512)
        self.ui.Constructed.setPixmap(pixmap)

    def ReconstructionImage(self, Kspace):
        KspaceSave = abs(Kspace)
        imsave(self.fileName5, KspaceSave)
        self.ui.FourierMatrix.setPixmap(QtGui.QPixmap(self.fileName5).scaled(512, 512))
        Kspacefft = np.fft.ifft2(Kspace)
        # Kspaceifft = np.fft.ifft2(Kspace)
        Kspacefft = abs(Kspacefft)
        imsave("image.png", Kspacefft)
        pixmap = QtGui.QPixmap("image.png")
        pixmap = pixmap.scaled(512, 512)
        self.ui.Constructed.setPixmap(pixmap)

    def Error_mess(self):
        self.error_dialog = QtWidgets.QErrorMessage()
        self.error_dialog.showMessage('Oh no!')
        self.label.setPixmap(QtGui.QPixmap(None))
        self.label_2.setPixmap(QtGui.QPixmap(None))


app = QtWidgets.QApplication(sys.argv)
window = window()
sys.exit(app.exec_())