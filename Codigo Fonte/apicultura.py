# importing matplotlib modules 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

import numpy as np

import cv2 as cv

import random

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

'''
obs: Estou usando python 3.5.0

'''

def readImage():
    # Read Images 
    img = mpimg.imread('1.jpg') 
    return img


#Fundamentos de imagens digitais
#Ilustração dos efeitos de amostragem e quantização
#Diminui e aumenta a imagem, além de almentar o nível de cinza
def IluEfeitosAmostragem():
    image = color.rgb2gray(readImage())                #seleciona a imagem do arquivo
    image_rescaled = rescale(image, 0.50, anti_aliasing=False) 
    
    image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                                     anti_aliasing=True)
    
    image_downscaled = downscale_local_mean(image, (4, 3))
   
    fig, axes = plt.subplots(nrows=3, ncols=2)          #plota imagens em 3 linhas e 2 colunas
    ax = axes.ravel()

    
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Imagem Original")

    ax[1].imshow(image_rescaled, cmap='gray')
    ax[1].set_title("Rescaled image (aliasing)")

    ax[2].imshow(image_resized, cmap='gray')
    ax[2].set_title("Resized image (no aliasing)")

    ax[3].imshow(image_downscaled, cmap='gray')
    ax[3].set_title("Downscaled image (no aliasing)")


    ax[4].imshow(image, cmap='gray')
    ax[4].set_title("Quantização")

    #ax[0].set_xlim(0, 512)
    #ax[0].set_ylim(512, 0)
    plt.tight_layout()
    plt.show()

def abrirJanela():
    cv.waitKey(0) 
    cv.destroyAllWindows() 

def readImageOpenCV(valor, contador):

    x = str(contador[0])
    string = x+".jpg"
    if valor == 0: #se for 0, imagem em cor cinza
        img = cv.imread(string, 0)
        img = redimemsionarImagem(img)
    else:
        img = cv.imread(string)
        img = redimemsionarImagem(img)
    return img

def writeImageOpenCV(img, contador):
    contador[0] += 1 #variável usada para salvar as segmentadas em disco
    x = str(contador)
    string = x+".jpg"
    print(contador)
    cv.imwrite(string, img)
    #return contador

#Operações aritméticas sobre imagens
def operacoesArtImg(img, contador):

    if type(img) == int:
        img = readImageOpenCV(1, contador)
        img = redimemsionarImagem(img)
        add  =  cv.add (img ,10)   #adiciona os pixels da imagem com algum valor escalar
        writeImageOpenCV(add, contador)
    else:
    	add = cv.add(img, 10)
    	writeImageOpenCV(add, contador)
    
    return add
   
def redimemsionarImagem(img):
    largura = img.shape[1]
    altura = img.shape[0]
    proporcao = float(altura/largura)
    largura_nova = 800 #em pixels
    altura_nova = int(largura_nova*proporcao)
    tamanho_novo = (largura_nova, altura_nova)
    img_redimensionada = cv.resize(img,tamanho_novo, interpolation = cv.INTER_AREA)
    return img_redimensionada

def negativoDeUmaImagem(img, contador):
    if type(img)==int:
        img = redimemsionarImagem(readImageOpenCV(0, contador))
        img_not = cv.bitwise_not(img)
        writeImageOpenCV(img_not, contador)
    else:
        img_not = cv.bitwise_not(img)
        writeImageOpenCV(img_not, contador)

    return img_not


def equalizacaoHistograma(img, contador):    #almenta o contraste de uma imagem

    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(0, contador))
        #cv.imshow("Sem equalizacao", img)

        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h_eq = cv.equalizeHist(img)
        contador = writeImageOpenCV(h_eq, contador)
    else: 
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h_eq = cv.equalizeHist(img)
        contador = writeImageOpenCV(h_eq, contador)

    return h_eq

def suavizacaoCalculoDaMedia(img, contador): #borra a imagem
    if type(img)==int:
        img = redimemsionarImagem(readImageOpenCV(1, contador))
        suave = cv.blur(img,(3,3))
        writeImageOpenCV(suave, contador)
    else:
        suave = cv.blur(img, (3,3))
        writeImageOpenCV(suave, contador)
    return suave

def filtroLaplaciano(img, contador):
    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(1, contador))
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lap = cv.Laplacian(img, cv.CV_64F)
        #lap = np.uint8(np.absolute(lap))
        #resultado = np.vstack([img, lap])
        writeImageOpenCV(lap, contador)
    else:
        lap = cv.Laplacian(img, cv.CV_64F)
        writeImageOpenCV(lap, contador)
    return lap

'''
Gradiente de uma imagem
    -Usado para reaçar os defeitos de uma imagem
    -se usa em indústrias
    -Usa-se a mascará de Sobel para encontrar o gradiente
'''
def gradienteImagem(img, contador):  
    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(0, contador))
        #cv.imshow("Sem Gradiente", img)
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #img = cv.imread('Desert.jpg')

        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        edges_x = cv.filter2D(img,cv.CV_8U,kernelx)
        edges_y = cv.filter2D(img,cv.CV_8U,kernely)

        ed = edges_x + edges_y

        writeImageOpenCV(ed, contador)
    else:
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        edges_x = cv.filter2D(img,cv.CV_8U,kernelx)
        edges_y = cv.filter2D(img,cv.CV_8U,kernely)

        ed = edges_x + edges_y

        writeImageOpenCV(ed, contador)
    return ed

def suavizacaoPelaGaussiana(img, contador):  #suavizacao pela Gaussiana gera menos borrão na imagem
    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(1, contador))
        suave = cv.GaussianBlur(img, ( 3, 3), 0)
        writeImageOpenCV(suave, contador)
    else:
        suave = cv.GaussianBlur(img, ( 3, 3), 0)
        writeImageOpenCV(suave, contador)

    return suave

def segmentacaoDeteccaoDeBordas(img, contador):# é usado para realçar as bordas da imagem
    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(1, contador))
        sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
        sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv.bitwise_or(sobelX, sobelY)
        writeImageOpenCV(sobel, contador)
    else:
        sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
        sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv.bitwise_or(sobelX, sobelY)
        writeImageOpenCV(sobel, contador)

    return sobel

def efeitosDalimiarizacaoDetecBordas(img, contador):
    img = limiarizacaoImagemDetcBordas(img, contador)
    img = segmentacaoDeteccaoDeBordas(img, contador)
    return img
    
#Primeiro foi feito a segmentação para detecção de bordas e apos isso a limiarizacao
def limiarizacaoImagemDetcBordas(img, contador): #converter imagem em tons de cinza para preto e branco
    #img = redimemsionarImagem(readImageOpenCV(1))
    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(1, contador))
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        suave = cv.GaussianBlur(img, (7, 7), 0) # aplica blur
        (T, binI) = cv.threshold(suave, 160, 255,  #coloca a imagem em preto e branco
        cv.THRESH_BINARY_INV)
        writeImageOpenCV(binI, contador)
    else:
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        suave = cv.GaussianBlur(img, (7, 7), 0) # aplica blur
        (T, binI) = cv.threshold(suave, 160, 255,  #coloca a imagem em preto e branco
        cv.THRESH_BINARY_INV)
        writeImageOpenCV(binI, contador)
    
    return binI

def limiarizacaoUsandoOtsu(img, contador):
    if type(img) == int:
        #img = redimemsionarImagem(readImageOpenCV(0))  #pass 0 to convert into gray level 
        img = redimemsionarImagem(readImageOpenCV(0, contador))
        ret,thr = cv.threshold(img, 0, 255, cv.THRESH_OTSU) #Só aceita imagem preto e branco
        writeImageOpenCV(thr, contador)   
    else:
        ret,thr = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
        writeImageOpenCV(thr, contador)   
    return thr

def mascaraDeNitidezHigh_Boost(img, contador):

    if type(img) == int:
        '''
        -MASCARA
        Passo 1: Suaviza a imagem original
        Passo 2: Subtrai a imagem suavizada da original
        Passo 3: Aplica o filtro
        '''
        img = redimemsionarImagem(readImageOpenCV(1, contador))

        original = img
        suavizada = cv.GaussianBlur(img, (7,7), 0)

        resultado = suavizada - original
        #onde o 2 é o valor do meu k
        filtroHB = original + 3*resultado
        writeImageOpenCV(filtroHB, contador)
    else:
        original = img
        suavizada = cv.GaussianBlur(img, (7,7), 0)

        resultado = suavizada - original
        #onde o 2 é o valor do meu k
        filtroHB = original + 3*resultado
        writeImageOpenCV(filtroHB, contador)

    return filtroHB

def limiarizacaoLocal(img, contador):
    if type(img) == int:
        img = redimemsionarImagem(readImageOpenCV(0, contador))
        img = cv.medianBlur(img,5)
        ret,th1 = cv.threshold(img,32,186,cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(img,186,cv.ADAPTIVE_THRESH_MEAN_C,\
                    cv.THRESH_BINARY,11,2) # O limiarLocal é a media dos pixels vizinhos menos uma constante C
        th3 = cv.adaptiveThreshold(img,186,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,11,2) #Usando a Gaussiana fica uma imagem sem ruídos
                                        #Sendo o limiar a média ponderada da gaussiana dos valores dos pixels vizinhos menos uma constante C
        writeImageOpenCV(th2, contador)                                
    else:
        img = cv.medianBlur(img,5)
        ret,th1 = cv.threshold(img,174,110,cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                    cv.THRESH_BINARY,11,2) # O limiarLocal é a media dos pixels vizinhos menos uma constante C
        th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,11,2) #Usando a Gaussiana fica uma imagem sem ruídos
                                        #Sendo o limiar a média ponderada da gaussiana dos valores dos pixels vizinhos menos uma constante C    
        writeImageOpenCV(th2, contador)

    return th2

if __name__ == "__main__": 
    img = 0
    case = 1
    contador = [0]

    '''
        Obs:
        -O método equalizacaoHistograma precisa de uma imagem na cor cinza para seu funcionamento,
        então em alguns casos não foi possível usá-lo, pois outros métodos usam imagens coloridas
        -Para modificar a cor da imagem no momento da leitura é só trocar o parametro (dentro dos métodos) onde:
            *0 cor cinza
            *1 colorido
    '''
    #1 teste

    
    img = equalizacaoHistograma(img, contador)
    img = limiarizacaoLocal(img, contador)
    img = mascaraDeNitidezHigh_Boost(img, contador)
    img = operacoesArtImg(img, contador)
    img = negativoDeUmaImagem(img, contador)
    img = negativoDeUmaImagem(img, contador)
    

    #2 teste
    '''
    img = limiarizacaoLocal(img, contador)
    img = equalizacaoHistograma(img, contador)
    img = operacoesArtImg(img, contador)
    img = mascaraDeNitidezHigh_Boost(img, contador)
    img = negativoDeUmaImagem(img, contador)
    '''

    #3 teste
    '''
    img = mascaraDeNitidezHigh_Boost(img, contador)
    img = limiarizacaoLocal(img, contador)
    img = equalizacaoHistograma(img, contador)
    img = operacoesArtImg(img, contador)
    img = negativoDeUmaImagem(img, contador)
    '''

    #4 teste
    '''
    img = mascaraDeNitidezHigh_Boost(img, contador)
    img = segmentacaoDeteccaoDeBordas(img, contador)
    '''
