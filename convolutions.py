# linha de comando python convolutions.py --image 3d_pokemon.png

# pacotes necessários
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import pdb
import time
from scipy import stats

# método de convulsão
def convolve(image, kernel):
        # dimensões espaciais da imagem, e do kernel
        # retorna a altura e largura da imagem em matriz
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]

        # alocar memória para a imagem de saída, tendo o cuidado de
        # "acoplar" as bordas da imagem de entrada para que o espaço
        # tamanho (ou seja, largura e altura) não sejam reduzidos
        # replicando os pixels para que a img de saída tenha o mesmo tamanho
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        # faça um loop sobre a imagem de entrada, "deslizando" o kernel
        # cada (x, y) - coordena da esquerda para a direita e de cima para
        # inferior


        # Parte Utilizada na Analise Quantitativa
        #==================================================================================#
        for y in np.arange(pad, iH + pad): # Complexidade n
                for x in np.arange(pad, iW + pad): # Complexidade n
                        # extrai a area de interesse da imagem ROI, pegando o
                        # *centro* das coordenadas (x, y)atuais
                        roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1] # 6 Operações

                        #cálculo para convolução
                        k = (roi * kernel).sum() # 9 + 8 + 1 Operações
                        #print(k)
                        #armazena o valor envolvido na saída (x, y) -
                        #coordenada da imagem de saída
                        output[y - pad, x - pad] = k # 3 Operaçõoes
        # ==================================================================================#

        #Contagem FInal: 3O(n² . (9³ + 18))

        # redimensionar a imagem de saída para estar no intervalo [0, 255]
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        # return a imagem de sáida
        
        
        return output

# argumento de imagem
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
args = vars(ap.parse_args())

# kernels
# parametros: ksize, sigma, theta, lambda
gabor = cv2.getGaborKernel((21,21), 5, 1, 10, 1, 0, cv2.CV_32F)
mauricio = cv2.getGaborKernel((21,21), 15, 1, 10, 1, 0, cv2.CV_32F)
marcelo = cv2.getGaborKernel((21,21), 25, 2, 20, 1, 0, cv2.CV_32F)
tamires = cv2.getGaborKernel((21,21), 35, 2, 15, 1, 0, cv2.CV_32F)
ricardo = cv2.getGaborKernel((21,21), 8, 1, 4, 3, 0, cv2.CV_32F)
gilmar = cv2.getGaborKernel((21,21), 3, 3, 6, 3, 0, cv2.CV_32F)


# construir o banco de kernel, uma lista de kernels que vamos
# aplicar usando a função `convole` e a Função filter2D do OpenCV

kernelBank = (
        ("maricio", mauricio),
        ("ricardo", ricardo),
        ("tamires", tamires),
        ("marcelo", marcelo),
        ("gilmar", gilmar)        
)

# carrega a imagem de entrada e convert para escala de cinza.
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, kernel) in kernelBank:
        print("[INFO] aplicando {} kernel".format(kernelName))
        tempoInicial = time.time()
        # aplicando função convolve
        convoleOutput = convolve(gray, kernel)
        tempoFinal = time.time()
        tempoExecucao = str(tempoFinal - tempoInicial)
        # aplicando função2D do openCVopencvOutput = cv2.filter2D(gray, -1, kernel)       
        tempoFinal = time.time()
        tempoExecucao = str(tempoFinal - tempoInicial)
        print(tempoExecucao)
        print("\n")
        print("Media")
        print(convoleOutput.mean())
        print("Variancia")
        print(convoleOutput.var())
        print("Desvio padrao")
        print(convoleOutput.std())
        
        print("\n")
        

        # mostrar imagens de saída
        cv2.imshow("original", gray)
        cv2.imshow("{} - convole".format(kernelName), convoleOutput)
        
        #cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
