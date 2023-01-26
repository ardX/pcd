import numpy as np
import cv2

def counter(img):
    # membaca file citra
    #img = cv2.imread('fruits.jpg')

    # konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # filter penghalusan dengan gaussian blur
    blur = cv2.GaussianBlur(gray,(25,25),0)

    # perbaikan brightness dan contrast
    brightness = -140
    contrast = 90
    tmp = np.int16(blur)
    tmp = tmp * (contrast/127+1) - contrast + brightness
    tmp = np.clip(tmp, 0, 255)
    adjusted = np.uint8(tmp)

    # merubah ke citra biner(hitam putih)
    ret, thresh = cv2.threshold(adjusted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # penghapusan noise menggunakan operasi morfologi opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # bagian yang pasti background
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # bagian yang pasti foreground/obyek
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # bagian unknown (bukan pasti background dan pasti foreground)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # pemberian marker pada tiap obyek
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    # bagian unknown diset 0
    markers[unknown==255] = 0

    # proses watershed
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # perhitungan jumlah obyek dan pelabelan obyek
    new_img = img.copy()
    conts,h=cv2.findContours(sure_fg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    num_obj= len(conts)
    #print("jumlah buah: "+str(num_obj))
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.putText(new_img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255))
    
    return new_img, num_obj

    # simpan dalam file
    #cv2.imwrite("buah.png", new_img)