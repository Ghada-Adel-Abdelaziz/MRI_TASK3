import numpy as np
import math
from matplotlib import pyplot as plt

img = np.zeros((4,4))

#create white box an array [value] is the no. of values and determine the intensity, reshape it into rows and columns

box1 = np.array([255]*1*1).reshape(1,1)
box2 = np.array([150]*2*2).reshape(2,2)
box3 = np.array([125]*2*2).reshape(2,2)
box4 = np.array([25]*4*4).reshape(4,4)
box5 = np.array([200]*1*1).reshape(1,1)

#Generate specific x-y coordinates 

x1=1
y1=2
x2=1
y2=0

#Replace part of original image with white box

img[x1:x1+2, y1:y1+2] = box1
img[x2:x2+2, y2:y2+2] = box2
#img[x3:x3+1, y3:y3+1] = box3
# a function that returns T1 ( recovery time ) based on the intensity

def createT2(intensity):
        
        if intensity == 100: #Gray matter
                T2 =170
        elif intensity == 255: #white matter       
                T2 =150
        elif intensity == 200: #muscle        
                T2 = 50
        elif intensity == 120 : #fat        
                T2 = 100
        elif intensity == 25: #protein       
                T2 = 10
        elif intensity == 150:
                T2 = 255

        elif intensity > 1: #Black => air
                T2 = (0.5*intensity) + 20
        
        elif intensity <= 1: 
            T2 = (intensity*100) + 10

        return T2

def createT1(intensity):

        if intensity == 100: #Gray matter
            T1=255
            
        elif intensity == 255: #white matter
            T1= 100
        
        elif intensity == 200: #muscle
            T1=180
        
        elif intensity == 120 : #fat
            T1=200
            
        elif intensity == 25: #protein
            T1=255
            
        elif intensity == 0: #Black => air
            T1=1
            
        elif intensity > 1: #Black => air
            T1 = round((7.5*intensity) + 50)
        
        elif intensity <= 1: 
            T1 = (intensity*100) + ((intensity * 100) +50)
			
      

        return T1



T1 = np.zeros((img.shape[0],img.shape[1]))
T2= np.zeros((img.shape[0],img.shape[1]))


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        T1[i,j]=createT1(img[i,j])
        T2[i,j]=createT2(img[i,j])


import scipy.io

output = {
		"Phantom" : img,
        "T1": T1,
        "T2":T2,
	}
scipy.io.savemat('Phantom4x4.mat', output)

plt.imshow(img, cmap="gray")
plt.show()
plt.imshow(T1, cmap="gray")
plt.show()
plt.imshow(T2, cmap="gray")
plt.show()



#self.pixmap5 = QtGui.QPixmap(self.fileName4)
#                self.ui.image.setPixmap(self.pixmap5)
# self.img[self.x, self.y]

def rotationAroundYaxisMatrix(theta,vector):
            vector = vector.transpose()
            theta = (math.pi / 180) * theta
            R = np.matrix ([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]] )
            R = np.dot(R, vector)
            R = R.transpose()
            return np.matrix(R)


def rotationAroundZaxisMatrixXY(theta,vector): #time = self.time
            vector = vector.transpose()
            theta = (math.pi / 180) * theta
            XY = np.matrix([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0, 0, 1]])
            XY = np.dot(XY,vector)
            XY = XY.transpose()
            return np.matrix(XY) 


def recoveryDecayEquation(T1,T2,PD,vector,time):
            vector = vector.transpose()
            Decay =np.matrix([[np.exp(-time/T2),0,0],[0,np.exp(-time/T2),0],[0,0,np.exp(-time/T1)]])
            Decay = np.dot(Decay,vector)
        
            Rec= np.dot(np.matrix([[0,0,(1-(np.exp(-time/T1)))]]),PD)
            Rec = Rec.transpose()
            Decay = np.matrix(Decay)
            Rec =  np.matrix(Rec)    
        
            RD  = Decay + Rec
            RD = RD.transpose()
            return RD

#Generate random array of zeros 512 rows and 512 columns



box1 = np.array([25]*70*50).reshape(70,50)
box2 = np.array([200]*25*150).reshape(25,150)
box3 = np.array([120]*100*25).reshape(100,25)
box4 = np.array([255]*100*50).reshape(100,50)
box5 = np.array([100]*50*70).reshape(50,70)

#Generate specific x-y coordinates 
x1=250
y1=250
x2=290
y2=290
x3=240
y3=285
x4=210
y4=220
x5=310
y5=305

#Replace part of original image with white box

img[x1:x1+70, y1:y1+50] = box1
img[x2:x2+25, y2:y2+150] = box2
img[x3:x3+100, y3:y3+25] = box3
img[x4:x4+100, y4:y4+50] = box4
img[x5:x5+50, y5:y5+70] = box5


import pprint
TE = 50
vector= np.matrix ([0,0,1]) #da range sabt
theta = 90 
TR = 3000

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrum = np.angle(fshift)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

img = output['Phantom']
self.pahntomForLoop = img
self.imgT1 = output['T1']
self.imgT2 = output['T2']

#imsave("phantom2.jpg", img)

plt.show()