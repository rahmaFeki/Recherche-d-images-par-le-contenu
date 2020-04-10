from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as figureCanvas
import cv2 as cv
from skimage.feature import greycomatrix
import numpy as np
from skimage.feature import greycoprops


def makeFigure(layout):
    fig = Figure()
    canvas = figureCanvas(fig)
    layout.addWidget(canvas)
    canvas.draw()
    return fig

#normaliser une distance entre 0 et 1
def normaliser(val, min, max):
    return (val - min) / (max - min)

#calcul histogramme
def imHist(img):
    hist = None
    hist = cv.calcHist([img],  [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist)
    return hist

#retourne la distance la plus élevée
def maximum(dict):
    m = 0
    for k in dict.keys():
        if (dict[k] >= m):
            m = dict[k]
    return m

#retourne la distance la plus faible
def minimum(dict):
    m = 0
    for k in dict.keys():
        if (dict[k] <= m):
            m = dict[k]
    return m

#retourne les matrices de cooccurence
def calcomatrix(img):
    glcm = greycomatrix(img, [1], [0, np.pi / 4, np.pi/2, (np.pi * 3 / 4)], levels=256, normed=True, symmetric=True)
    return glcm

#retourne les paramétres
def calcText(img):
    comatrix = calcomatrix(img)
    contrast = np.mean(greycoprops(comatrix, 'contrast'))
    energy = np.mean(greycoprops(comatrix, 'energy'))
    correlation = np.mean(greycoprops(comatrix, 'correlation'))
    homo = np.mean(greycoprops(comatrix, 'homogeneity'))
    return [contrast, energy, correlation, homo]
