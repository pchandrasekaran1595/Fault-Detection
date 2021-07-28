import os
import sys
import cv2
import shutil
import ctypes
import torch
import platform
import tkinter as tk
from torchvision import ops
from tkinter import messagebox
from PIL import Image, ImageTk
from time import time 

import Models
import utils as u
from MakeData import make_data 
from Train import trainer

_, batch_size, lr, wd = Models.build_siamese_model()
screen_resolution = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))

# ******************************************************************************************************************** #

# Inference Helper
def __help__(frame=None, model=None, show_prob=True, pt1=None, pt2=None, fea_extractor=None):
    disp_frame = frame.copy()

    frame = u.preprocess(frame, change_color_space=False)
    with torch.no_grad():
        features = u.normalize(fea_extractor(u.FEA_TRANSFORM(frame).to(u.DEVICE).unsqueeze(dim=0)))
        y_pred = torch.sigmoid(model(features))[0][0].item()

    if show_prob:
        if y_pred >= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_GREEN, thickness=2)
            if pt1[0] != 'None' and pt1[1] != 'None' and pt2[0] != 'None' and pt2[1] != 'None':
                cv2.rectangle(img=disp_frame, 
                              pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                              color=u.GUI_GREEN, thickness=2)
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_ORANGE, thickness=2)
            if pt1[0] != 'None' and pt1[1] != 'None' and pt2[0] != 'None' and pt2[1] != 'None':
                cv2.rectangle(img=disp_frame, 
                              pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                              color=u.GUI_ORANGE, thickness=2)
        else:
            cv2.putText(img=disp_frame, text="No Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_RED, thickness=2)
            if pt1[0] != 'None' and pt1[1] != 'None' and pt2[0] != 'None' and pt2[1] != 'None':
                cv2.rectangle(img=disp_frame, 
                              pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                              color=u.GUI_RED, thickness=2)
    else:
        if y_pred >= u.lower_bound_confidence:
            if pt1[0] != 'None' and pt1[1] != 'None' and pt2[0] != 'None' and pt2[1] != 'None':
                cv2.rectangle(img=disp_frame, 
                              pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                              color=u.GUI_GREEN, thickness=2)
            else:
                cv2.putText(img=disp_frame, text="Match", org=(25, 75),
                            fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=(0, 255, 0), thickness=2)
            
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            if pt1[0] != 'None' and pt1[1] != 'None' and pt2[0] != 'None' and pt2[1] != 'None':
                cv2.rectangle(img=disp_frame, 
                              pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                              color=u.GUI_ORANGE, thickness=2)
            else:
                cv2.putText(img=disp_frame, text="Possible Match", org=(25, 75),
                            fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=u.GUI_ORANGE, thickness=2)
        
        else:
            if pt1[0] != 'None' and pt1[1] != 'None' and pt2[0] != 'None' and pt2[1] != 'None':
                cv2.rectangle(img=disp_frame, 
                              pt1=(int(pt1[0]) - u.RELIEF, int(pt1[1]) - u.RELIEF), pt2=(int(pt2[0]) + u.RELIEF, int(pt2[1]) + u.RELIEF), 
                              color=u.GUI_RED, thickness=2)
            else:
                cv2.putText(img=disp_frame, text="No Match", org=(25, 75),
                            fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=u.GUI_RED, thickness=2)
    return disp_frame


class Video(object):
    def __init__(self, id=None, width=None, height=None, fps=None):
        self.id = id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self):
        if platform.system() != 'Windows':
            self.cap = cv2.VideoCapture(self.id)
        else:
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, model=None, part_name=None, isResult=False, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.V = V
        self.image = None
        self.isResult = isResult
        self.part_name = part_name
        self.model = model

        # If VideoFrame is in Result Mode; Default is None
        if self.isResult:
            self.model_path = os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints"), "State.pt")
            self.model.load_state_dict(torch.load(self.model_path, map_location=u.DEVICE)["model_state_dict"])
            self.model.eval()
            self.model.to(u.DEVICE)
            file = open(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Box.txt"), "r")
            self.data = file.read().split(",")
            file.close()

        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        self.delay = 15
        self.id = None
    
    def start(self):
        self.V.start()
        self.update()
    
    def update(self):
        ret, frame = self.V.get_frame()

        if not self.isResult:
            frame = u.clahe_equ(frame)
            temp_frame = frame.copy()
            x1, y1, x2, y2 = u.get_box_coordinates(Models.roi_extractor, u.ROI_TRANSFORM, temp_frame)
            frame = u.process(frame, x1, y1, x2, y2)
            if ret:
                self.image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.id = self.after(self.delay, self.update)
            else:
                return
        else:
            if ret:
                frame = u.clahe_equ(frame)
                temp_frame = frame.copy()
                frame = __help__(frame=frame, model=self.model, 
                                 pt1=(self.data[0], self.data[1]), pt2=(self.data[2], self.data[3]),
                                 show_prob=True, fea_extractor=Models.fea_extractor)
                self.image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.id = self.after(self.delay, self.update)
            else:
                return
        
    def stop(self):
        if self.id:
            self.after_cancel(self.id)
            self.id = None
            self.V.stop()
    
# ******************************************************************************************************************** #

class ImageFrame(tk.Frame):
    def __init__(self, master, imgfilepath, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.image = None

        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        if imgfilepath:
            self.image = cv2.cvtColor(src=cv2.imread(imgfilepath, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)
            self.image = cv2.resize(src=self.image, dsize=(u.CAM_WIDTH, u.CAM_HEIGHT), interpolation=cv2.INTER_AREA)
            self.image = ImageTk.PhotoImage(Image.fromarray(self.image))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

# ******************************************************************************************************************** #

class ButtonFrame(tk.Frame):
    def __init__(self, master, VideoWidget=None, ImageWidget=None, model=None, part_name=None, adderstate=False, *args, **kwargs):
        tk.Frame.__init__(self, master, width=150, background="#2C40D1", *args, **kwargs)

        self.master = master
        self.VideoWidget = VideoWidget
        self.ImageWidget = ImageWidget
        self.widget_height = 3
        self.widget_width = 25
        self.mdoel = model

        self.part_name = part_name
        if self.part_name:
            path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive")
            Len = len(os.listdir(path))
            if Len > 0:
                self.countp = Len
            else:
                self.countp = 1
            
            path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Negative")
            Len = len(os.listdir(path))
            if Len > 0:
                self.countn = Len + 1
            else:
                self.countn = 1
    
        self.adderstate = adderstate
        if self.adderstate:
            self.buttonState = "normal"
        else:
            self.buttonState = "disabled"
        self.model = model

        # Label
        self.label = tk.Label(self, text="Component/Part Name", 
                              background="gray", foreground="black", 
                              width=self.widget_width, height=self.widget_height,
                              relief="raised")
        self.label.grid(row=0, column=0)

        # Entry
        self.entry = tk.Entry(self, background="white", foreground="black",
                              selectbackground="blue", selectforeground="white", 
                              width=self.widget_width, relief="sunken")
        self.entry.grid(row=0, column=1)

        # Add
        self.addButton = tk.Button(self, text="Add Object",
                                   width=self.widget_width, height=self.widget_height, 
                                   background="#0FAFFF", activebackground="#8ED9FF", foreground="black",
                                   relief="raised", command=self.do_add)
        self.addButton.grid(row=1, column=0)

        # Train
        self.trainButton = tk.Button(self, text="Train",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#74FF00", activebackground="#B1FF70", foreground="black",
                                     relief="raised", command=self.do_train)
        self.trainButton.grid(row=1, column=1)

        # Realtime Application
        self.rtAppButton = tk.Button(self, text="Application",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#00FFD4", activebackground="#8AFFEB", foreground="black",
                                     relief="raised", command=self.do_rtapp)
        self.rtAppButton.grid(row=1, column=2)

        # Add to Positive
        self.posButton = tk.Button(self, text="Add to Positive",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#69E14D", activebackground="#A8E39A", foreground="black",
                                     relief="raised", state=self.buttonState, command=self.do_pos)
        self.posButton.grid(row=2, column=0)

        # Add to Negative
        self.negButton = tk.Button(self, text="Add to Negative",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#7F827E", activebackground="#BAC0B8", foreground="black",
                                     relief="raised", state=self.buttonState, command=self.do_neg)
        self.negButton.grid(row=2, column=2)

        # Reset
        self.resetButton = tk.Button(self, text="Reset",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#FFC500", activebackground="#FFDF84", foreground="black",
                                     relief="raised", command=self.do_reset)
        self.resetButton.grid(row=3, column=0)

        # Quit
        self.quitButton = tk.Button(self, text="Quit",
                                    width=self.widget_width, height=self.widget_height, 
                                    background="red", activebackground="#FCAEAE", foreground="black",
                                    relief="raised", command=self.do_quit)
        self.quitButton.grid(row=3, column=2)
    
    # Callback handling Adding of new Objects
    def do_add(self):
        self.part_name = self.entry.get()
        if self.part_name:
            self.path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive")
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            shutil.rmtree(os.path.join(u.DATASET_PATH, self.part_name))
            os.makedirs(self.path)

        file = open(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Box.txt"), "w+")
        ret, frame = self.VideoWidget.V.get_frame()
        frame = u.clahe_equ(frame)
        if ret:
            cv2.imwrite(os.path.join(self.path, "Snapshot_1.png"), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            x1, y1, x2, y2 = u.get_box_coordinates(Models.roi_extractor, u.ROI_TRANSFORM, frame)
            file.write(repr(x1) + "," + repr(y1) + "," +repr(x2) + "," + repr(y2))
        file.close()
        
        self.VideoWidget.stop()
        self.master.iconify()

        u.breaker()
        u.myprint("Generating Feature Vector Data ...", "green")
        start_time = time()
        make_data(part_name=self.part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
        make_data(part_name=self.part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
        u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")
        trainer(part_name=self.part_name, model=self.model, epochs=u.epochs, lr=lr, wd=wd, batch_size=batch_size, early_stopping=u.early_stopping_step, fea_extractor=Models.fea_extractor)

        self.master.state("zoomed")
        self.master.destroy()
        setup(part_name=self.part_name, model=self.model, imgfilepath=os.path.join(self.path, "Snapshot_1.png"), adderstate=True, isResult=True)

    # Callback handling training of objects
    def do_train(self):
        if not self.part_name:
            self.part_name = self.entry.get()
            self.path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive")

        if self.part_name:
            u.breaker()

            self.VideoWidget.stop()
            self.master.iconify()

            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=self.part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            make_data(part_name=self.part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")
            trainer(part_name=self.part_name, model=self.model, epochs=u.epochs, lr=lr, wd=wd, batch_size=batch_size, early_stopping=u.early_stopping_step, fea_extractor=Models.fea_extractor)

            self.VideoWidget.start()
            self.master.state("zoomed")

            u.breaker()
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
    
    # Callback handling realtime application start
    def do_rtapp(self):
        self.part_name = self.entry.get()
        if self.part_name:
            self.master.destroy()
            setup(part_name=self.part_name, model=self.model, imgfilepath=os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive"), "Snapshot_1.png"), adderstate=True, isResult=True)
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
    
    # Callback handling adding images to Positive Diretory
    def do_pos(self):
        ret, frame = self.VideoWidget.V.get_frame()
        frame = u.clahe_equ(frame)
        if ret:
            cv2.imwrite(os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive"), "Extra_{}.png".format(self.countp)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            self.countp += 1

    # Callback handling adding images to Negative Diretory
    def do_neg(self):
        ret, frame = self.VideoWidget.V.get_frame()
        frame = u.clahe_equ(frame)
        if ret:
            cv2.imwrite(os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Negative"), "Extra_{}.png".format(self.countn)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            self.countn += 1
    
    # Callback resetting the application
    def do_reset(self):
        self.VideoWidget.V.stop()
        self.master.destroy()
        setup(model=self.model)
    
    # Callback quitting the application
    def do_quit(self):
        self.VideoWidget.V.stop()
        self.master.master.destroy()

# ******************************************************************************************************************** #

# Wrapper
class Application():
    def __init__(self, master, V=None, part_name=None, model=None, imgfilepath=None, adderstate=False, isResult=False):

        VideoWidget = VideoFrame(master, V=V, model=model, part_name=part_name, isResult=isResult)
        VideoWidget.pack(side="left")
        VideoWidget.start()
        ImageWidget = ImageFrame(master, imgfilepath=imgfilepath)
        ImageWidget.pack(side="right")
        ButtonWidget = ButtonFrame(master, VideoWidget=VideoWidget, ImageWidget=ImageWidget, model=model, part_name=part_name, adderstate=adderstate)
        ButtonWidget.pack(side="bottom")

# ******************************************************************************************************************** #

# Top level window setup and Application start
def setup(part_name=None, model=None, imgfilepath=None, adderstate=False, isResult=False):
    window = tk.Toplevel()
    window.title("Application")
    window.geometry("{}x{}".format(screen_resolution[0], screen_resolution[1]))
    window.state("zoomed")
    w_canvas = tk.Canvas(window, width=screen_resolution[0], height=screen_resolution[1], bg="#40048C")
    w_canvas.place(x=0, y=0)
    Application(window, V=Video(id=u.device_id, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, fps=u.FPS), 
                part_name=part_name, model=model, imgfilepath=imgfilepath, adderstate=adderstate, isResult=isResult)


# ******************************************************************************************************************** #

# Building the GUI Application; contains basic CLI arguments
def app():
    args_1 = "--num-samples"
    args_2 = "--embed"
    args_3 = "--epochs"
    args_4 = "--lower"
    args_5 = "--upper"
    args_6 = "--early"

    if args_1 in sys.argv:
        u.num_samples = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        u.embed_layer_size = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        u.epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        u.lower_bound_confidence = float(sys.argv[sys.argv.index(args_4) + 1])        
    if args_5 in sys.argv:
        u.upper_bound_confidence = float(sys.argv[sys.argv.index(args_5) + 1]) 
    if args_6 in sys.argv:
        u.early_stopping_step = int(sys.argv[sys.argv.index(args_6) + 1]) 

    root = tk.Tk()
    rw, rh = 256, 256
    root.geometry("{}x{}".format(rw, rh))
    root.title("Root Window")
    root.iconify()

    model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)
    setup(model=model)
    
    root.mainloop()

# ******************************************************************************************************************** #