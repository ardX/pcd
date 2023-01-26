import numpy as np
import cv2
import glob

def counter(img):
    # membaca file citra
    # img = cv2.imread('buah/fruits.jpg')

    # konversi dari warna ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian filter untuk penghalusan
    blur = cv2.GaussianBlur(gray,(25,25),0)

    # perbaikan brightness dan kontras
    brightness = -140
    contrast = 90
    tmp = np.int16(blur)
    tmp = tmp * (contrast/127+1) - contrast + brightness
    tmp = np.clip(tmp, 0, 255)
    adjusted = np.uint8(tmp)

    # konversi ke citra biner
    ret, thresh = cv2.threshold(adjusted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    # lakukan watershed
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # perhitungan jumlah obyek dan pelabelan
    new_img = img.copy()
    conts,h=cv2.findContours(sure_fg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    number_of_objects_in_image= len(conts)
    print("jumlah buah: "+str(number_of_objects_in_image))
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.putText(new_img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255))

    # show the image
    # cv2.imshow("nama window", new_img)
    # cv2.waitKey(0)

    # cv2.imwrite("ouput.png", new_img)
    return new_img

filenames = [img for img in glob.glob("buah/*.*")]
for imgfile in filenames:
    print(imgfile)
    img = cv2.imread(imgfile)
    img = counter(img)
    cv2.imwrite(imgfile.replace("buah", "output"), img)


