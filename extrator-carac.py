import numpy as np
import cv2
import skimage
import pandas as pd
from skimage import img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from skimage.morphology import skeletonize, disk, convex_hull_image, medial_axis
from skimage import io, color, img_as_ubyte, data
from skimage.filters.rank import entropy
from skimage.segmentation import flood, flood_fill
from skimage.util import invert, img_as_float
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
#from resizeimage import resizeimage
import os, os.path
import json, csv, sys
from pathlib import Path
from pprint import pprint

# inicializando as variaveis
nomes=[] #vetor com os nomes das imagens
path = r'baseTreinamento/' #pasta a ser usada no processamento
#print(os.getcwd()) #checar se o projeto esta no diretorio correto
os.chdir(path) #mudar para o diretorio desejado
#print(os.getcwd())
imagens_validas = [".jpg"] #extensao valida das imagens a serem usadas no processamento
listaResult = []#array com os resultados do processamento

#inicio a extracao de caracteristica da imagem
def get_textural_features(img, isMultidirectional=False, distance=1):
    img = np.array(img.convert('L', colors=8))
    glcm = greycomatrix(img, [distance], [0], 256, symmetric=False, normed=False)
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    feature = np.array([dissimilarity, correlation, homogeneity, energy])
    return feature

#pegando todas as imagens do diretorio
for im in os.listdir(os.getcwd()):
    filename = os.path.splitext(im)#separa o nome da extensao
    ext = filename[1]#pega a extensao
    if ext.lower() not in imagens_validas:#checa se esta dentro das extensoes validas
        continue
    nomeOrigem = filename[0] + ext #nome original da imagem
    imgOriginal = cv2.imread(nomeOrigem, cv2.IMREAD_ANYCOLOR)#le a imagem
    cortada = imgOriginal[30:190, 10:190]#cortando imagens para deixar (160, 180)
    cv2.imshow("cortada",cortada)
    imgHSV = cv2.cvtColor(cortada, cv2.COLOR_BGR2HSV)#HSV
    cv2.imshow("HSV",imgHSV)
    imgHSV[:,:,2] = cv2.equalizeHist(imgHSV[:,:,2])#equalizar
    imgBlur = cv2.medianBlur(imgHSV,11)#blur
    cv2.imshow("Median Blur",imgBlur)
    mascara = cv2.inRange(imgBlur,(0,20,0),(180,170,85))#pegar a mão thresh, inRange UTF = (imgHSV,(0,20,0),(180,170,60))
    cv2.imshow("Mascara",mascara)
    btws = cv2.bitwise_and(imgBlur, imgBlur, mask=mascara)#pegar a mão colorida
    cv2.imshow("Bitwise",btws)
    maoGray = cv2.cvtColor(btws, cv2.COLOR_BGR2GRAY)#mao em gray
    cv2.imshow("Gray",maoGray)
    _,maoThresh = cv2.threshold(maoGray,0,255,0)#thresh, otsu deixa espaços em branco na mão
    contours, hierarchy = cv2.findContours(maoThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#encontrando contornos list
    maxContour = max(contours, key = cv2.contourArea)#pegando o maior contorno da lista
    imgContornos = np.zeros([160,180,3],dtype=np.uint8)#imagem com mesmo formato
    cv2.drawContours(imgContornos, maxContour, -1, (255,0,0), 3)#desenha o contorno da mão em azul
    cv2.imshow("Contornos",imgContornos)
    #pegar centro dos contornos
    for c in maxContour:
        M = cv2.moments(maxContour)
        cX = int(M["m10"] / M["m00"])#centro X
        cY = int(M["m01"] / M["m00"])#centro Y
    #floodfill a partir do centro do contorno
    cv2.floodFill(imgContornos, None, seedPoint=(cX,cY), newVal=(255,0,0))#preenchendo a imagem
    cv2.imshow("FloodFill",imgContornos)
    mascaraContorno = cv2.inRange(imgContornos,(250,0,0),(255,0,0))#criando a máscara para a imagem
    btwsContorno = cv2.bitwise_and(cortada, cortada, mask=mascaraContorno)#pegar a mão colorida
    cv2.imshow("Bitwise and",btwsContorno)
    btwsGray = cv2.cvtColor(btwsContorno, cv2.COLOR_BGR2GRAY)#mao em gray novamente
    _,btwsThresh = cv2.threshold(btwsGray,0,255,0)#thresh, otsu deixa espaços em branco na mão
    kernel = np.ones((3,3),np.uint8)#elemento estruturante
    imgAbertura = cv2.morphologyEx(btwsThresh, cv2.MORPH_OPEN, kernel)#opening na img thresh
    imgFechamento = cv2.morphologyEx(imgAbertura, cv2.MORPH_CLOSE, kernel)#closing na img thresh
    cv2.imshow("Morfológicas",imgFechamento)
    _,btwsThresh2 = cv2.threshold(imgFechamento,127,1,cv2.THRESH_OTSU)#thresh novamente
    #cv2.imshow("Thresh", imgFechamento)
    skel, distance = medial_axis(btwsThresh2, return_distance=True)
    medial_skeleton = distance * skel#medial skeleton
    zhang_skeleton = skeletonize(btwsThresh2, method='zhang')#esqueleton zhang
    lee_skeleton = skeletonize(btwsThresh2, method='lee')#esqueleton lee
    mdarray = np.array(medial_skeleton)
    zgarray = np.array(zhang_skeleton)
    learray = np.array(lee_skeleton)
        #cv_medial = img_as_float(medial_skeleton)
        #cv_zhang = img_as_float(zhang_skeleton)
        #cv_lee = img_as_float(lee_skeleton)
    img_medial = Image.fromarray(mdarray)
    img_zhang = Image.fromarray(zgarray)
    img_lee = Image.fromarray(learray)
    #pprint(cv_lee)
    #extrair características dos 3 skeletons
    medial_feature = get_textural_features(img_medial)#extraindo da img medial skeleton
    zhang_feature = get_textural_features(img_zhang)#extraindo da img zhang skeleton
    lee_feature = get_textural_features(img_lee)#extraindo da img lee skeleton
    #criar lista dessas caracteristicas
    #pprint(mdarray)
    #pprint("agr flatenned")
    #pprint(mdarray.flatten())
    #print(np.histogram(mdarray.flatten(),bins='auto'))
    listaResult.append({
        "Imagem" : nomeOrigem,#nome da imagem com extensao
        "MD_Dissimilaridade" : medial_feature[0],#GLCM dissimilaridade
        "MD_Correlação" : medial_feature[1],#GLCM correlação
        "MD_Homogeneidade" : medial_feature[2],#GLCM homogeneidade
        "MD_Energia" : medial_feature[3],#GLCM energia
        "MD_DesvioPadrao" : np.std(mdarray.flatten()),#Estatísticas desvio padrão
        "MD_Variancia" : np.var(mdarray.flatten()),#Estatísticas variância
        "MD_Media" : np.mean(mdarray.flatten()),#Estatísticas média
        "MD_Uniformidade" : 1 - 0.5*sum(abs(mdarray.flatten() - np.average(mdarray.flatten())))/(len(mdarray.flatten())*np.average(mdarray.flatten())),#Estatísticas uniformidade
        "ZG_Dissimilaridade" : zhang_feature[0],#GLCM dissimilaridade
        "ZG_Correlação" : zhang_feature[1],#GLCM correlação
        "ZG_Homogeneidade" : zhang_feature[2],#GLCM homogeneidade
        "ZG_Energia" : zhang_feature[3],#GLCM energia
        "ZG_DesvioPadrao" : np.std(zgarray.flatten()),#Estatísticas desvio padrão
        "ZG_Variancia" : np.var(zgarray.flatten()),#Estatísticas variância
        "ZG_Media" : np.mean(zgarray.flatten()),#Estatísticas média
        "ZG_Uniformidade" : 1 - 0.5*sum(abs(zgarray.flatten() - np.average(zgarray.flatten())))/(len(zgarray.flatten())*np.average(zgarray.flatten())),#Estatísticas uniformidade
        "LE_Dissimilaridade" : lee_feature[0],#GLCM dissimilaridade
        "LE_Correlação" : lee_feature[1],#GLCM correlação
        "LE_Homogeneidade" : lee_feature[2],#GLCM homogeneidade
        "LE_Energia" : lee_feature[3],#GLCM energia
        "LE_DesvioPadrao" : np.std(learray.flatten()),#Estatísticas desvio padrão
        "LE_Variancia" : np.var(learray.flatten()),#Estatísticas variância
        "LE_Media" : np.mean(learray.flatten()),#Estatísticas média
        "LE_Uniformidade" : 1 - 0.5*sum(abs(learray.flatten() - np.average(learray.flatten())))/(len(learray.flatten())*np.average(learray.flatten())),#Estatísticas uniformidade
        "Classe" : nomeOrigem[0]#nome da classe
    })
    # plotar imagens
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8, 4),
                            sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(btwsThresh2, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title(nomeOrigem, fontsize=20)
    ax[1].imshow(img_medial, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Zhang', fontsize=20)
    ax[2].imshow(img_zhang, cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[2].set_title('Lee', fontsize=20)
    ax[3].imshow(img_lee, cmap='magma')#plt.cm.gray)
    ax[3].axis('off')
    ax[3].set_title('Medial-axis', fontsize=20)
    fig.tight_layout()
    plt.show()
#gerando dataframe
dataframe = pd.DataFrame(listaResult)
classes = dataframe.iloc[:,-1]#pegando as classes(26) (-1 por ser o ultimo elemento)
somente_dados = dataframe.iloc[:,1:25]#tirando nome (0) e classe (26)
#normalizando
normalizar = StandardScaler()
dados_normalizados = normalizar.fit_transform(somente_dados)
base = pd.DataFrame(dados_normalizados)
base['z'] = classes#adicionando uma coluna a base com as classes
#saida em CSV
os.chdir("..")#subir um diretorio
base.to_csv('saidaCSV.csv',header=False,index=False,sep=',',encoding='utf-8-sig')
cv2.waitKey()