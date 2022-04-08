from tkinter import *
import pyautogui
from PIL import Image, ImageTk
from mss import mss
import datetime
import numpy as np
import cv2
import torch 
from torchvision import transforms

model = torch.jit.load("model.pt")

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

tranform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

def predict(img_np):

    input_tensor = tranform(img_np[None:])
    out = model(input_tensor)[0]
    sroce = torch.argmax(out)[1]

    return (1-sroce)*100, sroce*100

class Application():
    def __init__(self, master):
        self.master = master
        self.rect = None
        self.x = self.y = 0
        self.start_x = None
        self.start_y = None
        self.curX = None
        self.curY = None

        # root.configure(background = 'red')
        # root.attributes("-transparentcolor","red")

        root.attributes("-transparent", "blue")
        root.geometry('600x600+200+200')  # set new geometry
        root.title('LIAR DETECTION')

        
        self.menu_frame = Frame(master, bg="blue")
        self.menu_frame.pack(fill=BOTH, expand=YES)

        

        self.buttonBar = Frame(self.menu_frame,bg="")
        self.buttonBar.pack(fill=BOTH,)

        self.snipButton = Button(self.buttonBar, width=3, command=self.createScreenCanvas, background="green")
        self.snipButton.pack(expand=YES)

        self.master_screen = Toplevel(root)
        self.master_screen.withdraw()
        self.master_screen.attributes("-transparent", "blue")
        self.picture_frame = Frame(self.master_screen, background = "blue")
        self.picture_frame.pack(fill=BOTH, expand=YES)

        self.label =Label(self.buttonBar)
        self.label.pack(expand=YES)

    def takeBoundedScreenShot(self, x1, y1, x2, y2):
        im = pyautogui.screenshot(region=(x1, y1, x2, y2))
        x = datetime.datetime.now()
        fileName = x.strftime("%f")
        im.save("snips/" + fileName + ".png")
        return x1, y1, x2, y2

    def createScreenCanvas(self):
        self.master_screen.deiconify()
        root.withdraw()

        self.screenCanvas = Canvas(self.picture_frame, cursor="cross", bg="grey11")
        self.screenCanvas.pack(fill=BOTH, expand=YES)

        self.screenCanvas.bind("<ButtonPress-1>", self.on_button_press)
        self.screenCanvas.bind("<B1-Motion>", self.on_move_press)
        self.screenCanvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.master_screen.attributes('-fullscreen', True)
        self.master_screen.attributes('-alpha', .3)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)

    def on_button_release(self, event):
        self.recPosition()

        if self.start_x <= self.curX and self.start_y <= self.curY:
            print("right down")
            pos = self.takeBoundedScreenShot(self.start_x, self.start_y, self.curX - self.start_x, self.curY - self.start_y)

        elif self.start_x >= self.curX and self.start_y <= self.curY:
            print("left down")
            pos = self.takeBoundedScreenShot(self.curX, self.start_y, self.start_x - self.curX, self.curY - self.start_y)

        elif self.start_x <= self.curX and self.start_y >= self.curY:
            print("right up")
            pos = self.takeBoundedScreenShot(self.start_x, self.curY, self.curX - self.start_x, self.start_y - self.curY)

        elif self.start_x >= self.curX and self.start_y >= self.curY:
            print("left up")
            pos = self.takeBoundedScreenShot(self.curX, self.curY, self.start_x - self.curX, self.start_y - self.curY)

        self.exitScreenshotMode()

        self.pos = pos

        self.update_image()


        # imgtk = ImageTk.PhotoImage(image = im)

        # self.label.imgtk = imgtk
        # self.label.configure(image=imgtk)
        # return event

    
    def update_image(self):

        left = int(self.pos[0])
        top = int(self.pos[1])
        width = int(self.pos[2])
        height = int(self.pos[3])

        mon = {'left': left, 'top': top, 'width': width, 'height': height}

        with mss() as sct:
            # while True:
            screenShot = sct.grab(mon)
            img = Image.frombytes(
                'RGB', 
                (screenShot.width, screenShot.height), 
                screenShot.rgb, 
            )
            src = np.asarray(img)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=5)
            # cv2.imshow('test', np.array(img))
            for i, (x, y, w, h) in enumerate(faces):
                faces = src[y:y + h, x:x + w]
                cv2.rectangle(src, (x, y), (x+w, y+h), (1, 1, 255), 2)
                # break
                
                break
            # cv2.imwrite(f'FrameVideo/{mode}/{label}/face_idx_{idx}_{j}_{mode}.jpg', faces)
            conf_liar, conf_true = predict(faces)

            pad = np.zeros((50, src.shape[1], 3), dtype=np.uint8)*255
            fontScale = 1
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)
            pad = cv2.putText(pad, f'LIAR: {conf_liar}%', (20,25), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            pad = cv2.putText(pad, f'TRUE: {conf_true}%', (300,25), font, 
                   fontScale, color, thickness, cv2.LINE_AA)

            src = np.concatenate((src, pad), 0)


            img = Image.fromarray(src)

            imgtk = ImageTk.PhotoImage(image = img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        root.after(20 , self.update_image)


    def exitScreenshotMode(self):
        print("Screenshot mode exited")
        self.screenCanvas.destroy()
        self.master_screen.withdraw()
        root.deiconify()

    def exit_application(self):
        print("Application exit")
        root.quit()

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.screenCanvas.canvasx(event.x)
        self.start_y = self.screenCanvas.canvasy(event.y)

        self.rect = self.screenCanvas.create_rectangle(self.x, self.y, 1, 1, outline='red', width=3, fill="blue")

    def on_move_press(self, event):
        self.curX, self.curY = (event.x, event.y)
        # expand rectangle as you drag the mouse
        self.screenCanvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)

    def recPosition(self):
        print(self.start_x)
        print(self.start_y)
        print(self.curX)
        print(self.curY)

if __name__ == '__main__':
    root = Tk()
    app = Application(root)
    root.mainloop()