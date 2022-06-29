import time
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import arduino
import classifier
from ttkthemes import *

class App(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        master.title("ウェアラブルデバイスの画面")
        master.geometry("600x300")
        
        style = ttk.Style()
        style.theme_use('black')
        
        # 変数
        
        self.progress_brush = tk.IntVar(0)
        self.progress_drink = tk.IntVar(0)
        self.progress_senobi = tk.IntVar(0)
        self.progress_walk = tk.IntVar(0)
        self.progress_face = tk.IntVar(0)
        self.progress_all = tk.IntVar(0)
        
        frame = ttk.Frame(
            master,
            padding=10,
            width=600,
            height=300,
        )
        frame.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # frame1
        
        frame1 = ttk.Frame(
            frame,
            padding=10,
            width=400,
            height=250
        )
        frame1.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # 進捗バー
        
        label_all = ttk.Label(
            frame1,
            text="進捗"
        )
        label_all.grid(column=0, row=0)
        
        style.configure("red.Horizontal.TProgressbar", background='green2')

        pbar_all = ttk.Progressbar(
            frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_all,
            maximum=20,
            length=200,
            mode="determinate",
            style="red.Horizontal.TProgressbar"
        )
        pbar_all.grid(column=1, row=0)
        
        # frame2
        
        frame2 = ttk.Frame(
            frame,
            padding=10,
            width=400,
            height=400
        )
        frame2.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # brush
        
        button_brush = ttk.Button(
            frame2,
            text="brush",
            command=self.update_brush
        )
        button_brush.grid(column=0, row=1)
        
        self.label_brush_completed = ttk.Label(
            frame2,
            text="Washroom: Brush Teeth",
            foreground="white",
        )
        self.label_brush_completed.grid(column=1, row=1)
        
        # drink
        
        button_drink = ttk.Button(
            frame2,
            text="drink",
            command=self.update_drink
        )
        button_drink.grid(column=0, row=2)
    
        self.label_drink_completed = ttk.Label(
            frame2,
            text="Kitchen: Drink Water",
            foreground="white",
        )
        self.label_drink_completed.grid(column=1, row=2)
        
        # senobi
        
        button_senobi = ttk.Button(
            frame2,
            text="senobi",
            command=self.update_senobi
        )
        button_senobi.grid(column=0, row=3)
        
        self.label_senobi_completed = ttk.Label(
            frame2,
            text="Bedroom: Senobi",
            foreground="white",
        )
        self.label_senobi_completed.grid(column=1, row=3)
        
        # walk
        
        button_walk = ttk.Button(
            frame2,
            text="walk",
            command=self.update_walk
        )
        button_walk.grid(column=0, row=4)
        
        self.label_walk_completed = ttk.Label(
            frame2,
            text="EveryWhere: Walk",
            foreground="white",
        )
        self.label_walk_completed.grid(column=1, row=4)
        
        # face
        
        button_face = ttk.Button(
            frame2,
            text="face",
            command=self.update_face
        )
        button_face.grid(column=0, row=5)
        
        self.label_face_completed = ttk.Label(
            frame2,
            text="Washroom: Wash Face",
            foreground="white",
        )
        self.label_face_completed.grid(column=1, row=5)
        
        # frame3
        
        frame3 = ttk.Frame(
            frame,
            padding=10,
            width=400,
            height=400
        )
        frame3.grid(column=1, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_start = ttk.Button(
            frame3,
            text="start",
            command=self.start
        )
        button_start.grid(column=0, row=0)
        
        button_stop = ttk.Button(
            frame3,
            text="stop",
            command=self.stop
        )
        button_stop.grid(column=1, row=0)
        
        button_report = ttk.Button(
            frame3,
            text="report",
            command=self.create_dialog_graph
        )
        button_report.grid(column=2, row=0)
        
        # frame4
        
        frame4 = ttk.Frame(
            frame,
            padding=10,
            width=400,
            height=400
        )
        frame4.grid(column=1, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.canvas = tk.Canvas(
            frame4,
            background='gray35'
        )
        self.canvas.grid(column=0, row=0)
        self.img_nothing = ImageTk.PhotoImage(Image.open('images/sleep.png').resize((200,200)))
        self.img_brush = ImageTk.PhotoImage(Image.open('images/brush.png').resize((200,200)))
        self.img_drink = ImageTk.PhotoImage(Image.open('images/drink.png').resize((200,200)))
        self.img_senobi = ImageTk.PhotoImage(Image.open('images/senobi.png').resize((200,200)))
        self.img_walk = ImageTk.PhotoImage(Image.open('images/walk.png').resize((200,200)))
        self.img_face = ImageTk.PhotoImage(Image.open('images/washface.png').resize((200,200)))
        self.img_finish = ImageTk.PhotoImage(Image.open('images/finish.png').resize((200,200)))
        self.update()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.show_image = self.canvas.create_image(
            w / 2,
            h / 2,                   
            image=self.img_nothing
        )

    def update_brush(self):
        if self.progress_brush.get() >= 4:
            self.label_brush_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_brush.set(self.progress_brush.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_brush)
            self.label_brush_completed["foreground"] = "yellow"
            
    def update_drink(self):
        if self.progress_drink.get() >= 4:
            self.label_drink_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_drink.set(self.progress_drink.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_drink)
            self.label_drink_completed["foreground"] = "yellow"
            
    def update_senobi(self):
        if self.progress_senobi.get() >= 4:
            self.label_senobi_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_senobi.set(self.progress_senobi.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_senobi),
            self.label_senobi_completed["foreground"] = "yellow"
            
    def update_walk(self):
        if self.progress_walk.get() >= 4:
            self.label_walk_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_walk.set(self.progress_walk.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
            self.label_walk_completed["foreground"] = "yellow"
            
    def update_face(self):
        if self.progress_face.get() >= 4:
            self.label_face_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_face.set(self.progress_face.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_face)
            self.label_face_completed["foreground"] = "yellow"
            
    def update_nothing(self):
        self.canvas.itemconfig(self.show_image, image=self.img_nothing)
            
    def create_dialog_graph(self):
        dlg = tk.Toplevel(self)
        dlg.title("今日の結果")
        dlg.geometry("1000x800")
        
        fig = plt.figure()
        ax = fig.add_subplot()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([self.progress_brush.get()*25,
                        self.progress_drink.get()*25,
                        self.progress_senobi.get()*25,
                        self.progress_walk.get()*25,
                        self.progress_face.get()*25])
        label = np.array(['brush', 'drink', 'senobi', 'walk', 'face'])
        ax.bar(x, y, tick_label=label)
        ax.set_ylabel("scores")
        canvas = FigureCanvasTkAgg(fig, master=dlg)
        canvas.draw()
        canvas.get_tk_widget().grid(column = 0, row = 0)
            
    def loop(self):
        data = []
        for _ in range(120):
            acc = arduino.get()
            data.append(acc)
            time.sleep(1/120)

        x = torch.tensor(data, dtype=torch.float)
        result = classifier.classificate(model, x)[-1]
        
        if result == 0:
            self.update_brush()
        elif result == 1:
            self.update_drink()
        elif result == 2:
            self.update_face()
        elif result == 3:
            self.update_walk()
        elif result == 4:
            self.update_senobi()
        else:
            self.update_nothing()
            
        self.jobID = self.after(1, self.loop)
        
    def start(self):
        self.after(1000, self.loop)
        
    def stop(self):
        self.after_cancel(self.jobID)

if __name__ == "__main__":
    root = ThemedTk()
    model = classifier.load_model(
        model_path='model_9freedom.pth',
        input_dim=9,
        hidden_dim=128,
        target_dim=5
    )
    app = App(master = root)
    app.mainloop()