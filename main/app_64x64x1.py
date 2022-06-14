import cv2
import numpy as np
from keras.models import load_model
from PIL import Image 
from keras.preprocessing import image
import time

image_x, image_y = 64,64

classifier = load_model('../models/other_models/model_epoch_48_98.6_final.h5')

classes = 21
letras = {'0' : 'A', '1' : 'B', '2' : 'C', '3': 'D', '4': 'E', '5':'F', '6':'G', '7': 'I', '8':'L', '9':'M', '10':'N', '11': 'O', '12':'P', '13':'Q', '14':'R', '15':'S', '16':'T', '17':'U', '18':'V', '19':'W','20':'Y'}
palavra = ''

def predictor():          
       test_image = Image.open('../temp/img.png').convert('L')
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       maior, class_index = -1, -1

       for x in range(classes):      
           
           if result[0][x] > maior:
              maior = result[0][x]
              class_index = x
       
       return [result, letras[str(class_index)]]
       
    
cam = cv2.VideoCapture(0)

img_counter = 0
sinais = cv2.imread("../dataset/alfabeto-libras.png")
sinais = cv2.resize(sinais, (480,640))
frase = cv2.imread("../dataset/fundo branco.png")
frase = cv2.resize(frase, (900,100))
img_text = ['','']
while True:
    ret, frame = cam.read()
    
    frame = cv2.flip(frame,1)
    img = cv2.rectangle(frame, (1250,600),(850,200), (255,0,127), thickness=2, lineType=8, shift=0)
    cv2.putText(frame, str(img_text[1]), (100, 230), cv2.FONT_HERSHEY_TRIPLEX, 6, (0, 0, 0)) # ESCREVE AS LETRAS
    cv2.putText(frame, palavra, (100, 700), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255)) # escreve a palavra escrita
    #cv2.putText(frase, palavra, (0, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0)) # escreve a palavra escrita
    imcrop = img[102*2:298*2, 427*2:623*2]


    cv2.imshow("SINAIS", sinais)
    cv2.imshow("FRASE", frase)
    cv2.imshow("FRAME", frame)
    
    imggray = cv2.cvtColor(imcrop,cv2.COLOR_BGR2GRAY)

    img_name = "../temp/img.png"
    save_img = cv2.resize(imggray, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()
    print(palavra)
    
    #Guarda a letra informada na camera
    if cv2.waitKey(2) == 32: # Tecla espaço
        palavra += str(img_text[1])
    
    # Retira a ultima letra da palavra
    if cv2.waitKey(2) == 97: # Tecla a
        palavra = palavra[:-1]
        time.sleep(1)

    # Limpa a palavra escrita
    if cv2.waitKey(2) == 98: # Tecla b
        palavra = ''

    # Finaliza o sistema    
    if cv2.waitKey(2) == 27: # Tecla ESC
        break


cam.release()
cv2.destroyAllWindows()