from tkinter import *
from PIL import ImageTk, Image
import cv2
from tkinter import filedialog, PhotoImage


root = Tk()

label_name_file = Label(root)
label_in = Label(root, text='in')
label_out = Label(root, text='out')

label_name_file.grid(row=0, column=1)
label_in.grid(row=0, column=2)
label_out.grid(row=0, column=3)
flag = True


def video_stream(cap, lmain):
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream, cap, lmain) 


def open_run_movie():
    open_file = filedialog.askopenfilename()
    global lmain, flag, label_name_file
    label_name_file['text'] = f"file name: {open_file.split('/')[-1]}"
    if flag:
        lmain = Label(root)
        lmain.grid(row=1, column=0, columnspan=4)
        flag = False
    else:
        lmain.destroy()
        lmain = Label(root)
        lmain.grid(row=1, column=0, columnspan=4)
    cap = cv2.VideoCapture(open_file)
    video_stream(cap, lmain)

    
button = Button(root, text='open file', command=open_run_movie)
button.grid(row=0, column=0)

root.mainloop()