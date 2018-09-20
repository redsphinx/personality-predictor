from Tkinter import Button, Label, PhotoImage, Tk
from gui_var import GuiVar
import cv2 as cv
from PIL import ImageTk, Image
import numpy as np
from utils import load_model, predict_frame
import plotting


def initialize(b):
    b.camera = Label(b.root, image=b.camera_image)
    b.camera.image = b.camera_image

    b.trait_all = Label(b.root, image=b.trait_image_all)
    b.trait_all.image = b.trait_image_all

    b.run_button = Button(b.root, text='Run', state='active', command=lambda: run(b))
    b.model = load_model()

    b.label_e = Label(b.root, text='friendly')
    b.label_a = Label(b.root, text='authentic')
    b.label_c = Label(b.root, text='organized')
    b.label_s = Label(b.root, text='comfortable')
    b.label_o = Label(b.root, text='imaginative')


def create_window(gv):
    gv.root.title(gv.window_title)
    gv.root.geometry('%dx%d+%d+%d' % (gv.window_width, gv.window_height, gv.window_position, gv.window_position))
    gv.trait_all.place(x=-50, y=0)
    gv.run_button.place(x=0, y=300)
    gv.camera.place(x=0, y=300)
    dx = 380
    dy = 60
    gv.label_e.place(x=dx, y=dy * 0.9)
    gv.label_a.place(x=dx, y=dy * 1.6)
    gv.label_c.place(x=dx, y=dy * 2.3)
    gv.label_s.place(x=dx, y=dy * 3)
    gv.label_o.place(x=dx, y=dy * 3.7)


def main_loop(gv):
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    img1 = Image.fromarray(frame, 'RGB')
    img2 = ImageTk.PhotoImage(img1)
    gv.camera_image = img2
    gv.camera.config(image=gv.camera_image)

    f = frame.astype(np.float32)
    f = np.transpose(f, (2, 0, 1))
    f = np.expand_dims(f, 0)

    y = predict_frame(f, gv.model)

    gv.series_e = plotting.update_series(gv.series_e, y[0].data[0])
    gv.series_a = plotting.update_series(gv.series_a, y[0].data[1])
    gv.series_c = plotting.update_series(gv.series_c, y[0].data[2])
    gv.series_s = plotting.update_series(gv.series_s, y[0].data[3])
    gv.series_o = plotting.update_series(gv.series_o, y[0].data[4])

    loc = plotting.make_plot(gv)
    gv.trait_image_all = PhotoImage(file=loc)
    gv.trait_all.config(image=gv.trait_image_all)



def start_everything(gv):
    main_loop(gv)
    interval = 20

    if gv.keep_going:
        gv.root.after(interval, lambda: start_everything(gv))


def run(gv):
    gv.keep_going = True
    gv.run_button.config(text='Run', state='disabled')
    start_everything(gv)


def main():
    b = GuiVar()
    initialize(b)

    create_window(b)
    root = b.root

    root.mainloop()


main()
