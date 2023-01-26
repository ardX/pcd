import numpy as np
import cv2

def counter(img):
    # membaca citra
    #img = cv2.imread('water_coins.jpg')

    # konversi ke grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height, width = gray.shape 

    # ubah brighness contrast
    brightness = -200
    contrast = 100
    tmp = np.int16(gray)
    tmp = tmp * (contrast/127+1) - contrast + brightness
    tmp = np.clip(tmp, 0, 255)
    adjusted = np.uint8(tmp)

    # konversi ke citra hitam putih/binary
    ret, thresh = cv2.threshold(adjusted,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # kernel yang digunakan
    kernel = np.ones((3,3),np.uint8)

    # proses opening
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # bagian pasti background
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # bagian pasti foreground/obyek
    sure_fg = cv2.erode(opening,kernel,iterations=3)

    # bagian yang bukan pasti background dan pasti obyek
    unknown = cv2.subtract(sure_bg,sure_fg)

    #buat marker untuk menandai komponen yang saling tersambung
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1

    # set bagian bukan pasti background dan pasti obyek dengan angka/marker 0
    markers[unknown==255] = 0

    # marker dijalankan pada fungsi watershed
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # hitung obyek dan beri label obyek
    new_img = img.copy()
    conts,h=cv2.findContours(sure_fg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    num_obj= 0
    for i in range(len(conts)):
        if cv2.contourArea(conts[i])>width:
            x,y,w,h=cv2.boundingRect(conts[i])
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,0,255), 2)
            cv2.putText(new_img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255))
            num_obj+=1
    
    
    # return hasil
    return new_img, num_obj