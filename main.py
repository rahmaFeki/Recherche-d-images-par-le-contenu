from PyQt5 import QtWidgets
import sys
import cv2 as cv
import numpy as np
import design
import functions as fn
import os


class designWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):

    def __init__(self):
        super(designWindow, self).__init__()
        self.setupUi(self)
        self.figImg = fn.makeFigure(self.horizontalLayout)
        self.figImg2 = fn.makeFigure(self.horizontalLayout_2)
        self.horizontalLayout = None
        self.histN = None
        self.shape = None
        self.contrast = None
        self.pushButton.clicked.connect(self.getImage)
        self.pushButton_2.clicked.connect(self.showResultats)

    # charger l'image requête et calculer ces caractéristiques
    def getImage(self):
        self.figImg2.clf()
        file = QtWidgets.QFileDialog.getOpenFileName(self, "choose image", "", "image files(*jpg)")
        if file[0]:
            print(file[0])
            self.horizontalLayout = cv.imread(file[0])
            self.horizontalLayout = cv.cvtColor(self.horizontalLayout, cv.COLOR_BGR2RGB)
            self.figImg.clf()
            ax = self.figImg.add_subplot(111)
            ax.imshow(self.horizontalLayout)
            ax.axis("off")
            self.figImg.canvas.draw()
            self.histN = fn.imHist(self.horizontalLayout)
            self.horizontalLayout = cv.cvtColor(self.horizontalLayout, cv.COLOR_RGB2GRAY)
            param = fn.calcText(self.horizontalLayout)
            self.contrast = param[0]
            self.enrgy = param[1]
            self.correlation = param[2]
            self.homo = param[3]
            _, binary = cv.threshold(self.horizontalLayout, 255, 255, cv.THRESH_BINARY_INV)

            contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            self.shape = contours[0]

    # calculer distance(img requete et image de la base) avec descripteur de couleur
    def compHist(self):
        dicti = {}
        histFN = None
        pathname = os.path.dirname('bases_images/')
        files = os.listdir(pathname)
        for file in files:
            f = cv.imread('bases_images/' + file)
            f = cv.cvtColor(f, cv.COLOR_BGR2RGB)
            histFN = fn.imHist(f)
            d = cv.compareHist(self.histN, histFN, cv.HISTCMP_INTERSECT)
            dicti[file] = d
        distance = sorted(dicti.items(), key=lambda x: x[1], reverse=True)
        di = {}
        for file in distance:
            di[file[0]] = file[1]
        return di

    # calculer distance(img requete et image de la base) avec descripteur de texture
    def cmpText(self):
        pathname = os.path.dirname('bases_images/')
        files = os.listdir(pathname)
        dicti = {}
        for file in files:
            f = cv.imread('bases_images/' + file)
            f = cv.cvtColor(f, cv.COLOR_BGR2RGB)
            f1 = cv.cvtColor(f, cv.COLOR_RGB2GRAY)
            param = fn.calcText(f1)
            contraste = param[0]
            energy = param[1]
            correlation = param[2]
            homo = param[3]
            distCont = np.abs((self.contrast - contraste) / self.contrast)
            distEner = np.abs((self.enrgy - energy) / self.enrgy)
            distCorr = np.abs((self.correlation - correlation) / self.correlation)
            distHomo = np.abs((self.homo - homo) / self.homo)
            distText = (distCont + distEner + distCorr + distHomo) / 4
            dicti[file] = distText
        return dicti

    # calculer distance(img requete et image de la base) avec descripteur de la forme avec cv.findContours et cv.matchshapes
    def calcShape(self):
        dict = {}
        pathname = os.path.dirname('bases_images/')
        files = os.listdir(pathname)
        for file in files:
            f1 = cv.imread('bases_images/' + file)
            f1 = cv.cvtColor(f1, cv.COLOR_BGR2RGB)
            f1 = cv.cvtColor(f1, cv.COLOR_RGB2GRAY)
            _, binary = cv.threshold(f1, 255, 255, cv.THRESH_BINARY_INV)

            shape, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            shape = shape[0]

            d = cv.matchShapes(shape, self.shape, cv.CONTOURS_MATCH_I2, 0)
            dict[file] = d

        return dict

    # chercher les images similaires avec l'img requete et afficher les 9 premiers
    def showResultats(self):
        self.figImg2.clf()
        distance = {}
        distanceCouleur = self.compHist()
        distanceTexture = self.cmpText()
        distanceForme = self.calcShape()
        minCouleur = fn.minimum(distanceCouleur)
        maxCouleur = fn.maximum(distanceCouleur)
        minTexture = fn.minimum(distanceTexture)
        maxTexture = fn.maximum(distanceTexture)
        minForme = fn.minimum(distanceForme)
        maxForme = fn.maximum(distanceForme)
        # calculer distance globale(0.25*couleur+0.25*texture+0.5*forme):dict distance{nomfic,distance}
        for file in distanceCouleur.keys():
            # distanceCouleur:la distance la plus élevée correspond à l'image la plus semblable
            # distanceTexture:la distance la plus faible correspond à l'image la plus semblable
            # distanceForme:la distance la plus faible correspond à l'image la plus semblable
            distance[file] = 0.5 * (1 - fn.normaliser(distanceCouleur[file], minCouleur, maxCouleur)) + \
                             0.5* fn.normaliser(distanceTexture[file], minTexture, maxTexture) + \
                             0* (fn.normaliser(distanceForme[file], minForme, maxForme))
        # tri croissant des distances globales(plus distance petite, plus les images sont similaires)
        distance = sorted(distance.items(), key=lambda x: x[1], reverse=False)
        d = distance[:9]
        x = 1

        for file in d:
            f = cv.imread('bases_images/' + file[0])
            f = cv.cvtColor(f, cv.COLOR_BGR2RGB)
            ax = self.figImg2.add_subplot(3, 3, x)
            q = file[1]
            q = round(q, 4)
            ax.set_title(q)
            ax.imshow(f)
            ax.axis("off")
            self.figImg2.canvas.draw()
            x = x + 1


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = designWindow()
    form.show()
    app.exec_()
