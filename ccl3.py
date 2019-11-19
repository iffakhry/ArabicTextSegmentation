import cv2
import imutils
import numpy as np
from imutils import contours

img = cv2.imread('bismillah2.png', 0)
widthImg, heightImg = img.shape
#img_gray = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
ret, threshA = cv2.threshold(img, 127, 255, 0)
imgA = cv2.bitwise_not(threshA)
imgA = cv2.resize(imgA, (imgA.shape[1]*2, imgA.shape[0]*2))
imgACopy = imgA.copy()
widthImgResize, heightImgResize = imgA.shape

#---- Original ----
ret, thresh1 = cv2.threshold(img, 127, 255, 0)
#---- Invert image ----
img1 = cv2.bitwise_not(thresh1)

#---- Langsung Threshold binary invert ----
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
img2 = thresh2

#---- Threshold menggunakan Gaussian filtering dan Otsu Thresholding ----
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,thresh3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img3 = cv2.bitwise_not(thresh3)

template = cv2.imread('harokat.png',0)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

# found = None
# # loop over the scales of the image
# for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#     # resize the image according to the scale, and keep track
#     # of the ratio of the resizing
#     resized = imutils.resize(img3, width=int(img3.shape[1] * scale))
#     r = img3.shape[1] / float(resized.shape[1])
#
#     # if the resized image is smaller than the template, then break
#     # from the loop
#     if resized.shape[0] < tH or resized.shape[1] < tW:
#         break
#
#     edged = cv2.Canny(resized, 50, 200)
#     result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
#     (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
#
#     # check to see if the iteration should be visualized
#     # if args.get("visualize", False):
#         # draw a bounding box around the detected region
#     clone = np.dstack([edged, edged, edged])
#     cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
#                   (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#     cv2.imshow("Visualize", clone)
#     cv2.waitKey(0)

    # # if we have found a new maximum correlation value, then ipdate
    # # the bookkeeping variable
    # if found is None or maxVal > found[0]:
    #     found = (maxVal, maxLoc, r)
    #
    # # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # # of the bounding box based on the resized ratio
    # (_, maxLoc, r) = found
    # (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    # (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    #
    # # draw a bounding box around the detected result and display the image
    # cv2.rectangle(img3, (startX, startY), (endX, endY), (0, 0, 255), 2)

# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img3,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img3, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)

# harokat matching
# template = cv2.imread('harokatBig.png', 0)
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(imgA, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(imgA, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 2)

#_, markers = cv2.connectedComponents(img)
#print np.amax(markers)

#--------------------
#RETR_EXTERNAL or RETR_TREE

# image, cnts , hierarchy = cv2.findContours(imgA,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts , hierarchy = cv2.findContours(imgA,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
(cnts, _) = contours.sort_contours(cnts)

#cnt = cnts[5]
#x,y,w,h = cv2.boundingRect(cnt)
#imgA = cv2.rectangle(imgA,(x,y),(x+w,y+h),(255,255,255),2)
print ("width",widthImg, " height: ", heightImg)

#counting
number = 0
numberA=0

#nilai tengah objek ke 1
cy1 = 0

#nilai tengah objek
cX=0
cY=0

#lokasi titik - titik objek
x=0
y=0
w=0
h=0

#height terbesar dari harokat
heightMaxHarokat = 0

for c in cnts:
    number += 1
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #-- location centroid
    print ("loc",number, " x: ", cX ,"  y: ", cY)

    #garis mendatar
    if number == 1:
        cy1 = cY
        cv2.line(imgA, (0, cY-35), (heightImgResize, cY-30), (255, 255, 255), 1)
        cv2.line(imgA, (0, cY), (heightImgResize, cY), (255, 255, 255), 1)
        cv2.line(imgA, (0, cY + 20), (heightImgResize, cY + 25), (255, 255, 255), 1)

    (x,y),radius = cv2.minEnclosingCircle(c)
    center = (int(x),int(y))
    radius = int(radius)
    #imgA = cv2.circle(imgA,center,radius,(255,255,255),1)
    #cv2.circle(imgA, center, radius, (255, 255, 255), 1)
    cv2.putText(imgA, "#{}".format(number), (cX - 1, cY -1 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #-- location x, y, w, h
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(imgA,(x,y),(x+w,y+h),(255,255,255),2)
    print ("pos",number, " x: ", x ,"  y: ", y ,"  w: ", w ,"  h: ", h)
    cv2.line(imgA,(x,0),(x,heightImg),(255,255,255),1)
    cv2.line(imgA,(x + w,0),(x + w ,heightImg),(255,255,255),1)

    #crop function untuk harokat
    if(cY <= cy1-35 or cY >= cy1+20):
        #ukur tinggi maksimal harokat
        if (h > heightMaxHarokat):
            heightMaxHarokat = h
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        crop_img = imgA[y:y+h, x:x+w] # Crop from x, y, w, h -> 100, 200, 300, 400
        # res = cv2.matchTemplate(crop_img, template, cv2.TM_CCOEFF_NORMED)
        # threshold = 0.8
        # loc = np.where(res >= threshold)
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(crop_img, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 2)
        #cv2.imshow("cropped{}".format(number), crop_img)

    #memilih objek huruf saja (belum dipisah per huruf)
    if(cY >= cy1-35 and cY <= cy1+20):
        if(h > heightMaxHarokat):
            crop_imgHuruf = imgACopy[0:heightImgResize, x:x + w]
            # cv2.imshow("cropped{}".format(number), crop_imgHuruf)
            yStart = y
            yEnd = y + h
            # xStart = x
            # xEnd = x + w

            xStart = 0
            xEnd = 0
            # imageQ, cntsQ, hierarchyQ = cv2.findContours(crop_imgHuruf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntsQ, hierarchyQ = cv2.findContours(crop_imgHuruf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (cntsQ, _) = contours.sort_contours(cntsQ)

            for c1 in cntsQ:

                Ma = cv2.moments(c1)
                cXa = int(Ma["m10"] / Ma["m00"])
                cYa = int(Ma["m01"] / Ma["m00"])
                xa, ya, wa, ha = cv2.boundingRect(c1)
                numberA += 1

                if(cYa <= cy1-35 or cYa >= cy1+20):
                    # if(xa >= xStart and xa+wa <= xEnd):
                    cv2.rectangle(crop_imgHuruf, (xa, ya), (xa + wa, ya + ha), (255, 255, 255), 2)
                    print ("posCek", numberA, " x: ", xa, "  y: ", ya, "  w: ", wa, "  h: ", ha)



                    if ((xa > xEnd) or ((xa < xEnd) and (xa + wa) > xEnd)) :
                        cv2.line(crop_imgHuruf, (xa - 5, 0), (xa - 5, heightImgResize), (255, 255, 255), 2)
                        cv2.line(crop_imgHuruf, ((xa + wa) + 5, 0), ((xa + wa) + 5, heightImgResize), (255, 255, 255), 2)

                    xStart = xa - 5
                    xEnd = (xa + wa) + 5
                    # crop_imgQ = crop_imgHuruf[ya:ya + ha, xa:xa + wa]
                    # cv2.imshow("croppedImgQ{}".format(numberA), crop_imgQ)
                    # if(ya>yStart):
                    #     yStart = ya
                    # if(ya + ha < yEnd):
                    #     yEnd = ya + ha
                    # if(xa > xStart):
                    #     xStart = xa
                    # if(xa + wa < xEnd):
                    #     xEnd = xa + wa


                    print ("xStart: ",xStart, " xEnd: ",xEnd," yStart: ",yStart," yENd: ",yEnd)
                    # crop_imgHuruf2 = crop_imgHuruf[0: heightImgResize, xStart-5:xEnd+5]

                    # if(numberA ==1):
                    #     xEnd = xa + wa
                    #     xStart = xEnd

            cv2.imshow("croppedQ{}".format(numberA), crop_imgHuruf)






#leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
#rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
#topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
#bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
#img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('imageSegment',imgA)
cv2.imshow('image',img1)
cv2.imshow('image2',img2)
cv2.imshow('image3',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()