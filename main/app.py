import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import datetime as dt

#------------------------------------------------------------------------------------
# DEFS
def predict():          
       imagemVerificar = image.load_img('../temp/img.png', target_size=(64, 64))
       imagemVerificar = image.img_to_array(imagemVerificar)
       imagemVerificar = np.expand_dims(imagemVerificar, axis = 0)
       result = classifier.predict(imagemVerificar)
       maior, class_index = -1, -1

       for x in range(classes):      
           
           if result[0][x] > maior:
              maior = result[0][x]
              class_index = x
       
       return [result, letters[str(class_index)]]

def writeLog(word):
    arq = open("logText.txt", "r")
    content = arq.readlines()
    content.append("-------------------------------------------------\n")
    content.append("Moment: " + str(dt.datetime.now()) + "\n\n")
    content.append(word)
    content.append("\n-------------------------------------------------\n")
    arq = open("logText.txt", "w")
    arq.writelines(content)
    arq.close()

#------------------------------------------------------------------------------------
#Variables

image_x, image_y = 64,64

classifier = load_model('../models/model-angelo_99.h5')
classes = 21
letters = {'0' : 'A', '1' : 'B', '2' : 'C', '3': 'D', '4': 'E', '5':'F', '6':'G', '7': 'I', '8':'L', '9':'M', '10':'N', '11': 'O', '12':'P', '13':'Q', '14':'R', '15':'S', '16':'T', '17':'U', '18':'V', '19':'W','20':'Y'}
word = ''
cam = cv2.VideoCapture(0)

img_counter = 0
gestures = cv2.imread("../dataset/alfabeto-libras.png")
gestures = cv2.resize(gestures, (480,640))

img_text = ['','']
#------------------------------------------------------------------------------------
# Main

while True:

    frase = cv2.imread("../dataset/fundo branco.png")
    frase = cv2.resize(frase, (900,100))

    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    checker = str(img_text[0])
    checker = checker[2:15]
    print(checker)

    if "e" not in checker:
        cv2.putText(frame, str(img_text[1]), (100, 230), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0)) # ESCREVE AS LETRAS
    
    img = cv2.rectangle(frame, (1250,600),(850,200), (255,0,127), thickness=2, lineType=8, shift=0)
    cv2.putText(frame, str(img_text[1]), (100, 230), cv2.FONT_HERSHEY_TRIPLEX, 6, (0, 0, 0)) # ESCREVE AS LETRAS
    #cv2.putText(frame, word, (100, 700), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255)) # escreve a palavra escrita
    cv2.putText(frase, word, (0, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0)) # escreve a palavra escrita
    imcrop = img[102*2:298*2, 427*2:623*2]

    cv2.imshow("SINAIS", gestures) # mapa de gestos
    cv2.imshow("FRASE", frase) # Printa a frase escrita
    cv2.imshow("FRAME", frame) # Imagem da camera com os objetos criados

    save_img = cv2.resize(imcrop, (image_x, image_y))
    cv2.imwrite("../temp/img.png", save_img)
    img_text = predict()    
    
    #Guarda a letra informada na camera
    if cv2.waitKey(1) == 32: # Tecla espaço
        word += str(img_text[1])
    
    # Retira a ultima letra da word
    if cv2.waitKey(1) == 97: # Tecla a
        word = word[:-1]

    # Limpa a word escrita
    if cv2.waitKey(1) == 98: # Tecla b
        word = ''

    # Finaliza o sistema    
    if cv2.waitKey(1) == 27: # Tecla ESC
        if(word):
            writeLog()
        break


cam.release()
cv2.destroyAllWindows()

