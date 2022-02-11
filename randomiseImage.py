import cv2, math, hashlib
import numpy as np
from pylab import array
from statistics import mean

MINSTRETCH = 15
MAXSTRETCH = 255 - MINSTRETCH
STRETCHRANGE = MAXSTRETCH - MINSTRETCH
lockImage = None
height = 0
width = 0


def createLockImage(keyString = "AUT"):
    Key1 = hashlib.md5(keyString.encode())
    Key2 = hashlib.md5(keyString.encode()[::-1]) #Reverse the key
    keyList1 = list(Key1.digest())
    keyList2 = list(Key2.digest())
    R1 = mean(keyList1)
    R2 = mean(keyList2)
    DIAMETER = ((R1+R2)/10)
    img = np.zeros((height, width), np.uint8)  # with 3 chanels # img2 = img1.copy()
    for h in range(0, height):
        for w in range(0, width):   
            wc = (w - R1) % DIAMETER - DIAMETER / 2
            hc = (h - R2) % DIAMETER - DIAMETER / 2            
            img[h, w] = (wc ** 2 + hc ** 2) + w + h
    #cv2.imshow("lock", img)
    #cv2.imwrite("lock.png", img)
    return img

def encodeRandom(originalImage):
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    originalImage = STRETCHRANGE * (originalImage / 255.0) + MINSTRETCH
    originalImage = array(originalImage, np.uint8)
    return lockImage - originalImage

def decodeRandom(originalImage):
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    originalImage = lockImage - originalImage
    originalImage[originalImage > MAXSTRETCH] = MAXSTRETCH
    originalImage[originalImage < MINSTRETCH] = MINSTRETCH
    originalImage = 255.0 * ((originalImage - MINSTRETCH) / STRETCHRANGE)
    return array(originalImage, np.uint8)

encodeURL = "encoded.jpg"
#Main function to test here.
def main():
    global height, width, lockImage, R1, R2, DIAMETER
    image = cv2.imread("high-st-image.jpg")

    height, width = image.shape[0:2]  # Total pixel number: img.size

    lockImage = createLockImage()
    cv2.imshow("lockImage", lockImage)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original image", image)    
    encodedImage = encodeRandom(image)
 
    cv2.imwrite(encodeURL, encodedImage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])   
    cv2.imshow("encoded", encodedImage)

    encodedImage = cv2.imread(encodeURL)   
    lockImage = createLockImage()
   
    decodedImage = decodeRandom(encodedImage)
    cv2.imwrite("decodedImage.png", decodedImage)
    cv2.imshow("decodedImage", decodedImage)

    diff = cv2.subtract(decodedImage, gray_image) + cv2.subtract(gray_image, decodedImage)
    diff[diff < 20] = 0
    cv2.imshow("different", diff)

    print(100.0 * cv2.countNonZero(diff)/(diff.shape[0] * diff.shape[1]))

    cv2.waitKey(0)

if __name__ == '__main__':
    main()


