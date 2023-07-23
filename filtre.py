from PIL import Image,ImageFilter,ImageDraw,ImageFont
from PIL.ImageFilter import(
    BLUR,CONTOUR,DETAIL,EDGE_ENHANCE,EDGE_ENHANCE_MORE,EMBOSS,FIND_EDGES,SMOOTH,SMOOTH_MORE,SHARPEN
)
import face_recognition
import cv2
import numpy as np
from collections import deque
import os
from pyfiglet import Figlet
def print_cool(text):
    cool_text = Figlet(font="slant")
    os.system('mode con : cols = 75 lines = 30')
    return str(cool_text.renderText(text))
print(print_cool("          Görüntü İşleme            "))
print("1-Thumbnail")
print("2-Filigran Ekle")
print("3-PNG ye çevirme")
print("4-Makyaj")
print("5-Noel Baba")
print("6-Fes")
print("7-Çiz")
a = int(input("Seçeneği Gir"))
if a == 1:
    resim = Image.open("a.jpg",mode='r')
    resim.thumbnail((150,150))
    resim.save("thumbnail.jpg")
    resim2 = Image.open("thumbnail.jpg")
    resim2.show()
if a == 2:
    resim = Image.open("a.jpg")
    resim_genislik , resim_yukseklik = resim.size
    filigran = ImageDraw.Draw(resim)
    metin = input("Filigran Yazısı")
    font = ImageFont.truetype('verdana.ttf',30)
    metin_genislik , metin_yukseklik = filigran.textsize(metin,font)
    margin = 20
    x = resim_genislik - metin_genislik - margin
    y = resim_yukseklik - metin_yukseklik - margin
    filigran.text((x,y),metin,font=font)
    resim.show()
    resim.save("filigran.jpg")
if a == 3:
    resim = Image.open("a.jpg",mode = 'r')
    resim.save("png.png")
    resim2 = Image.open("png.png")
    resim2.show()
if a == 4:
    image = face_recognition.load_image_file("a.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

        pil_image.show()
if a == 5 :

    shiftValueW =25
    shiftValueH = 35
    shiftValueX = 0
    shiftValueY = -10


    bgColorThresholdValue = 230
    haarCascadeForFace = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(haarCascadeForFace)
    image_model = cv2.imread('b.jpg')
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (600, 300))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )
        for (x, y, w, h) in faces:
            x = x + shiftValueX
            y = y + shiftValueY
            model_width = w + shiftValueW
            model_height = int(0.35 * h) + shiftValueH
            image_model = cv2.resize(image_model,(model_width, model_height))
            for i in range(model_height):
                for j in range(model_width):
                    for k in range(3):
                        if image_model[i][j][k] < bgColorThresholdValue:
                            frame[y+i-int(0.25*h)][x+j][k] = image_model[i][j][k]
        cv2.imshow('Sonuc', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if a == 6 :

    shiftValueW =25
    shiftValueH = 35
    shiftValueX = 0
    shiftValueY = -10


    bgColorThresholdValue = 230
    haarCascadeForFace = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(haarCascadeForFace)
    image_model = cv2.imread('a.jpg')
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (600, 300))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )
        for (x, y, w, h) in faces:
            x = x + shiftValueX
            y = y + shiftValueY
            model_width = w + shiftValueW
            model_height = int(0.35 * h) + shiftValueH
            image_model = cv2.resize(image_model,(model_width, model_height))
            for i in range(model_height):
                for j in range(model_width):
                    for k in range(3):
                        if image_model[i][j][k] < bgColorThresholdValue:
                            frame[y+i-int(0.25*h)][x+j][k] = image_model[i][j][k]
        cv2.imshow('Sonuc', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
if a == 8:
    def setValues(x):
        print("")



    cv2.namedWindow("Color detectors")
    cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180,setValues)
    cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255,setValues)
    cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,setValues)
    cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180,setValues)
    cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255,setValues)
    cv2.createTrackbar("Lower Value", "Color detectors", 49, 255,setValues)



    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]


    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0


    kernel = np.ones((5,5),np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0


    paintWindow = np.zeros((471,636,3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
    paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
    paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
    paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)



    cap = cv2.VideoCapture(0)


    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
        Upper_hsv = np.array([u_hue,u_saturation,u_value])
        Lower_hsv = np.array([l_hue,l_saturation,l_value])



        frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
        frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
        frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
        frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
        frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)



        Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        Mask = cv2.erode(Mask, kernel, iterations=1)
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
        Mask = cv2.dilate(Mask, kernel, iterations=1)


        cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        center = None


        if len(cnts) > 0:

            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))


            if center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    paintWindow[67:,:,:] = 255
                elif 160 <= center[0] <= 255:
                        colorIndex = 0
                elif 275 <= center[0] <= 370:
                        colorIndex = 1
                elif 390 <= center[0] <= 485:
                        colorIndex = 2
                elif 505 <= center[0] <= 600:
                        colorIndex = 3
            else :
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1


        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)


        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)
        cv2.imshow("mask",Mask)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()
