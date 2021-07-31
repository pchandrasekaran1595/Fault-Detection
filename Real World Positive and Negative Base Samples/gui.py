"""
    GUI Application
"""

import os
import sys
import cv2
import ctypes
import torch
import platform
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from time import time 

import Models
import utils as u
from MakeData import make_data 
from Train import trainer

# Initialize Siamese Network Hyperparameters
_, batch_size, lr, wd = Models.build_siamese_model()

# Get the resolution of the screen
screen_resolution = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))

# ******************************************************************************************************************** #

# Inference Helper
def __help__(frame=None, anchor=None, model=None, show_prob=True, fea_extractor=None):
    """
        frame         : Current frame being processed
        anchor        : Anchor Image
        model         : Siamese Network Model
        show_prob     : Flag to control whether to display the similarity score
        fea_extractor : Feature Extraction Model
    """
    disp_frame = frame.copy()
    h, w, _ = frame.shape

    # Alpha Blend Anchor Image if it is passed
    if anchor is not None:
        disp_frame = u.alpha_blend(anchor, disp_frame, 0.15)
    frame = u.preprocess(frame, False)
    
    # Perform Inference on current frame
    with torch.no_grad():
        features = u.normalize(fea_extractor(u.FEA_TRANSFORM(frame).to(u.DEVICE).unsqueeze(dim=0)))
        y_pred = torch.sigmoid(model(features))[0][0].item()

    # Prediction > Upper Bound                 -----> Match
    # Lower Bound <= Prediction <= Upper Bound -----> Possible Match
    # Prediction < Lower Bound                 -----> Defective
    if show_prob:
        if y_pred >= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_GREEN, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(w/2) - 100, int(h/2) - 100), 
                          pt2=(int(w/2) + 100, int(h/2) + 100), 
                          color=u.GUI_GREEN, thickness=2)
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Match, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_ORANGE, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(w/2) - 100, int(h/2) - 100), 
                          pt2=(int(w/2) + 100, int(h/2) + 100), 
                          color=u.GUI_ORANGE, thickness=2)
        else:
            cv2.putText(img=disp_frame, text="Defective, {:.5f}".format(y_pred), org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_RED, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(w/2) - 100, int(h/2) - 100), 
                          pt2=(int(w/2) + 100, int(h/2) + 100), 
                          color=u.GUI_RED, thickness=2)
    else:
        if y_pred >= u.lower_bound_confidence:
            cv2.putText(img=disp_frame, text="Match", org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=(0, 255, 0), thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(w/2) - 100, int(h/2) - 100), 
                          pt2=(int(w/2) + 100, int(h/2) + 100), 
                          color=u.GUI_GREEN, thickness=2) 
        elif u.lower_bound_confidence <= y_pred <= u.upper_bound_confidence:
            cv2.putText(img=disp_frame, text="Possible Match", org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_ORANGE, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(w/2) - 100, int(h/2) - 100), 
                          pt2=(int(w/2) + 100, int(h/2) + 100), 
                          color=u.GUI_ORANGE, thickness=2)
        else:
            cv2.putText(img=disp_frame, text="Defective", org=(25, 75),
                        fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=u.GUI_RED, thickness=2)
            cv2.rectangle(img=disp_frame, 
                          pt1=(int(w/2) - 100, int(h/2) - 100), 
                          pt2=(int(w/2) + 100, int(h/2) + 100), 
                          color=u.GUI_RED, thickness=2)
    return disp_frame

# ******************************************************************************************************************** #

# Capture Object Class
class Video(object):
    def __init__(self, id=None, width=None, height=None, fps=None):
        """
            id     : Device ID of the capture object
            width  : Width of the capture frame
            height : Height of the capture frame
            fps    : FPS of the capture object
        """
        self.id = id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self):
        """
            Initialize the capture object
        """
        if platform.system() != 'Windows':
            self.cap = cv2.VideoCapture(self.id)
        else:
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def get_frame(self):
        """
            Read a frame from the capture object
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    def stop(self):
        """
            Stop the capture object
        """
        if self.cap.isOpened():
            self.cap.release()

# tkinter Video Display
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
            # Load the Model
            self.model_path = os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints"), "State.pt")
            self.model.load_state_dict(torch.load(self.model_path, map_location=u.DEVICE)["model_state_dict"])
            self.model.eval()
            self.model.to(u.DEVICE)
            self.anchor = cv2.imread(os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive"), "Snapshot_1.png"), cv2.IMREAD_COLOR)

        # Setup the canvas and pack it into the frame
        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        # Delay after which frame will be updated (in ms)
        self.delay = 15
        self.id = None
    
    def start(self):
        """
            Start Updating the canvas
        """
        self.V.start()
        self.update()
    
    def update(self):
        """
            - Handles how the canvas is updated
            - Has 2 modes: Normal Mode and Result Mode
            - Normal Mode is used during frame capture, Result Mode is used during inference
        """
        ret, frame = self.V.get_frame()

        if not self.isResult:
            # Apply CLAHE (2, 2) Preprocessing. May not be required once lighting issue is fixed
            frame = u.clahe_equ(frame)
            if ret:
                h, w, _ = frame.shape
                frame = cv2.rectangle(img=frame, pt1=(int(w/2) - 100, int(h/2) - 100), pt2=(int(w/2) + 100, int(h/2) + 100), color=(255, 255, 255), thickness=2)

                # Convert image from np.ndarray format into tkinter canvas compatible format and update
                self.image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.id = self.after(self.delay, self.update)
            else:
                return
        else:
            if ret:
                # Apply CLAHE (2, 2) Preprocessing. May not be required once lighting issue is fixed
                frame = u.clahe_equ(frame)

                # Process frame in during inference
                frame = __help__(frame=frame, model=self.model, anchor=None,
                                 show_prob=True, fea_extractor=Models.fea_extractor)

                # Convert image from np.ndarray format into tkinter canvas compatible format
                self.image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.id = self.after(self.delay, self.update)
            else:
                return

    def stop(self):
        """
            Stop updating the canvas
        """
        if self.id:
            self.after_cancel(self.id)
            self.id = None
            self.V.stop()
    
# ******************************************************************************************************************** #

# tkinter Image Display
class ImageFrame(tk.Frame):
    def __init__(self, master, imgfilepath, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.image = None

        # Setup the canvas and pack it into the frame
        self.canvas = tk.Canvas(self, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, background="black")
        self.canvas.pack()

        # Display image if the filepath argument is passed
        if imgfilepath:
            # Read the image
            self.image = cv2.cvtColor(src=cv2.imread(imgfilepath, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)

            # Resize image to the shape of the canvas
            self.image = cv2.resize(src=self.image, dsize=(u.CAM_WIDTH, u.CAM_HEIGHT), interpolation=cv2.INTER_AREA)

            # Convert image from np.ndarray format into tkinter canvas compatible format
            self.image = ImageTk.PhotoImage(Image.fromarray(self.image))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

# ******************************************************************************************************************** #

# tkinter Button Handling
class ButtonFrame(tk.Frame):
    def __init__(self, master, VideoWidget=None, ImageWidget=None, part_name=None, isFirstTimeRun=None, *args, **kwargs):
        tk.Frame.__init__(self, master, width=150, background="#2C40D1", *args, **kwargs)

        self.master = master
        self.VideoWidget = VideoWidget
        self.ImageWidget = ImageWidget

        self.widget_height = 3
        self.widget_width = 25

        self.isFirstTimeRun = isFirstTimeRun

        self.part_name = part_name
        self.countn, self.countp = 1, 1

        # Label Widget
        self.label = tk.Label(self, text="Component/Part Name", 
                              background="gray", foreground="black", 
                              width=self.widget_width, height=self.widget_height,
                              relief="raised")
        self.label.grid(row=0, column=0)

        # Entry Widget
        self.entry = tk.Entry(self, background="white", foreground="black",
                              selectbackground="blue", selectforeground="white", 
                              width=self.widget_width, relief="sunken")
        self.entry.grid(row=0, column=1)

        # Button : Add to Positive
        self.posButton = tk.Button(self, text="Add to Positive",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#DFDFDC", activebackground="#F6F6F4", foreground="black",
                                     relief="raised", command=self.do_pos)
        self.posButton.grid(row=1, column=0)

        # Button : Add to Negative
        self.negButton = tk.Button(self, text="Add to Negative",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#505050", activebackground="#717171", foreground="black",
                                     relief="raised", command=self.do_neg)
        self.negButton.grid(row=1, column=1)

        # Button : Train
        self.trainButton = tk.Button(self, text="Train",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#45FE33", activebackground="#8EFF83", foreground="black",
                                     relief="raised", command=self.do_train)
        self.trainButton.grid(row=2, column=0)

        # Button : Realtime Application
        self.rtAppButton = tk.Button(self, text="Application",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#00FFD4", activebackground="#8AFFEB", foreground="black",
                                     relief="raised", command=self.do_rtapp)
        self.rtAppButton.grid(row=2, column=1)

        # Button : Reset
        self.resetButton = tk.Button(self, text="Reset",
                                     width=self.widget_width, height=self.widget_height, 
                                     background="#FFC500", activebackground="#FFDF84", foreground="black",
                                     relief="raised", command=self.do_reset)
        self.resetButton.grid(row=3, column=0)

        # Button : Quit
        self.quitButton = tk.Button(self, text="Quit",
                                    width=self.widget_width, height=self.widget_height, 
                                    background="red", activebackground="#FCAEAE", foreground="black",
                                    relief="raised", command=self.do_quit)
        self.quitButton.grid(row=3, column=1)

    # Callback handling adding images to the Positive Diretory
    def do_pos(self):

        # Get the part name from the entry field
        self.part_name = self.entry.get()

        # Check that user has entered a name; only then proceed, else display error message
        if self.part_name:
            self.path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive")
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
        
        # If the dataset path doesn't exist, create it
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.countp = len(os.listdir(self.path)) + 1

        # Read the current frame from the capture object
        ret, frame = self.VideoWidget.V.get_frame()

        # Apply CLAHE (2, 2) Preprocessing. May not be required once lighting issue is fixed
        frame = u.clahe_equ(frame)

        # Save the frame and update counter
        if ret:
            cv2.imwrite(os.path.join(self.path, "Snapshot_{}.png".format(self.countp)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            self.countp += 1


    # Callback handling adding images to the Negative Diretory
    def do_neg(self):

        # Get the part name from the entry field
        self.part_name = self.entry.get()

        # Check that user has entered a name; only then proceed, else display error message
        if self.part_name:
            self.path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Negative")
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
        
        # If the dataset path doesn't exist, create it
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.countn = len(os.listdir(self.path)) + 1

        # Read the current frame from the capture object
        ret, frame = self.VideoWidget.V.get_frame()

        # Apply CLAHE (2, 2) Preprocessing. May not be required once lighting issue is fixed
        frame = u.clahe_equ(frame)

        # Save the frame and update counter
        if ret:
            cv2.imwrite(os.path.join(self.path, "Snapshot_{}.png".format(self.countn)), cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))
            self.countn += 1
    
    # Callback handling the Training
    def do_train(self):

        # Get the part name from the entry field
        self.part_name = self.entry.get()

        # Check that user has entered a name; only then proceed, else display error message
        if self.part_name:
            u.breaker()

            # path = os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints")
            # names = os.listdir(path)
            # if "State.pt" in names:
            #     self.model_path = os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Checkpoints"), "State.pt")
            #     self.model.load_state_dict(torch.load(self.model_path, map_location=u.DEVICE)["model_state_dict"])

            # Release the capture object; Minimize the Application
            self.VideoWidget.stop()
            self.master.iconify()

            # Generate the Feature Vector Dataset
            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=self.part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor)
            make_data(part_name=self.part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")

            # Initialize Siamese Network
            model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)

            # Train the Model
            trainer(part_name=self.part_name, model=model, epochs=u.epochs, lr=lr, wd=wd, batch_size=batch_size, early_stopping=u.early_stopping_step, fea_extractor=Models.fea_extractor)

            # Start the capture object; Maximize the Application
            self.VideoWidget.start()
            self.master.state("zoomed")

            u.breaker()
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
    
    # Callback handling the Inference
    def do_rtapp(self):

        # Get the part name from the entry field
        self.part_name = self.entry.get()

        # Check that user has entered a name; only then proceed, else display error message
        if self.part_name:
            # Destroy the current application window
            self.master.destroy()

            # Initialize Siamese Network
            model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)

            # Start a new application window
            setup(part_name=self.part_name, model=model, imgfilepath=os.path.join(os.path.join(os.path.join(u.DATASET_PATH, self.part_name), "Positive"), "Snapshot_1.png"), isResult=True)
        else:
            messagebox.showerror(title="Value Error", message="Enter a valid input")
            return
    
    # Callback handling reset
    def do_reset(self):
        # Release the capture object
        self.VideoWidget.V.stop()

        # Destory the current application window
        self.master.destroy()

        # Initialize Siamese Network
        model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)

        # Start a new application window
        setup(model=model)
    
    # Callback handling quit
    def do_quit(self):
        # Release the capture object
        self.VideoWidget.V.stop()

        # Destoy the root window; also destroys the application window.
        self.master.master.destroy()

# ******************************************************************************************************************** #

# Wrapper around all the tkinter frames
class Application(object):
    def __init__(self, master, V=None, part_name=None, model=None, imgfilepath=None, isResult=False):

        VideoWidget = VideoFrame(master, V=V, model=model, part_name=part_name, isResult=isResult)
        VideoWidget.pack(side="left")
        VideoWidget.start()
        ImageWidget = ImageFrame(master, imgfilepath=imgfilepath)
        ImageWidget.pack(side="right")
        ButtonWidget = ButtonFrame(master, VideoWidget=VideoWidget, ImageWidget=ImageWidget, part_name=part_name)
        ButtonWidget.pack(side="bottom")

# ******************************************************************************************************************** #

# Top level window setup and Application start
def setup(part_name=None, model=None, imgfilepath=None, isResult=False):
    # Setup a toplevel window
    window = tk.Toplevel()
    window.title("Application")
    window.geometry("{}x{}".format(screen_resolution[0], screen_resolution[1]))
    window.state("zoomed")
    w_canvas = tk.Canvas(window, width=screen_resolution[0], height=screen_resolution[1], bg="#40048C")
    w_canvas.place(x=0, y=0)

    # Initialize Application Wrapper
    Application(window, V=Video(id=u.device_id, width=u.CAM_WIDTH, height=u.CAM_HEIGHT, fps=u.FPS), 
                part_name=part_name, model=model, imgfilepath=imgfilepath, isResult=isResult)


# ******************************************************************************************************************** #

# Building the GUI Application; contains basic CLI arguments
def app():
    args_1 = "--num-samples"
    args_2 = "--embed"
    args_3 = "--epochs"
    args_4 = "--lower"
    args_5 = "--upper"
    args_6 = "--early"

    # CLI Argument Handling
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

    # Root Window Setup
    root = tk.Tk()
    rw, rh = 256, 256
    root.geometry("{}x{}".format(rw, rh))
    root.title("Root Window")
    root.iconify()

    # Initialize Siamese Network
    model, _, _, _ = Models.build_siamese_model(embed=u.embed_layer_size)

    # Start a new application window
    setup(model=model)
    
    # Start
    root.mainloop()

# ******************************************************************************************************************** #