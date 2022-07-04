import time
import tkinter as tk
from tkinter import ttk
from ttkthemes import *
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
from pygame import mixer
import arduino
import classifier
import kakugen

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
        
        self.done_brush = False
        self.done_drink = False
        self.done_senobi = False
        self.done_walk = False
        self.done_face = False
        
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
            if self.done_brush == False:
                self.done_brush = True
                mixer.init()
                mixer.music.load("sounds/complete.mp3")
                mixer.music.play(1)
        else:
            self.progress_brush.set(self.progress_brush.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_brush)
            self.label_brush_completed["foreground"] = "yellow"
            
    def update_drink(self):
        if self.progress_drink.get() >= 4:
            self.label_drink_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
            if self.done_drink == False:
                self.done_drink = True
                mixer.init()
                mixer.music.load("sounds/complete.mp3")
                mixer.music.play(1)
        else:
            self.progress_drink.set(self.progress_drink.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_drink)
            self.label_drink_completed["foreground"] = "yellow"
            
    def update_senobi(self):
        if self.progress_senobi.get() >= 4:
            self.label_senobi_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
            if self.done_senobi == False:
                self.done_senobi = True
                mixer.init()
                mixer.music.load("sounds/complete.mp3")
                mixer.music.play(1)
        else:
            self.progress_senobi.set(self.progress_senobi.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_senobi),
            self.label_senobi_completed["foreground"] = "yellow"
            
    def update_walk(self):
        if self.progress_walk.get() >= 4:
            self.label_walk_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
            if self.done_walk == False:
                self.done_walk = True
                mixer.init()
                mixer.music.load("sounds/complete.mp3")
                mixer.music.play(1)
        else:
            self.progress_walk.set(self.progress_walk.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
            self.label_walk_completed["foreground"] = "yellow"
            
    def update_face(self):
        if self.progress_face.get() >= 4:
            self.label_face_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
            if self.done_face == False:
                self.done_face = True
                mixer.init()
                mixer.music.load("sounds/complete.mp3")
                mixer.music.play(1)
        else:
            self.progress_face.set(self.progress_face.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_face)
            self.label_face_completed["foreground"] = "yellow"
            
    def update_nothing(self):
        self.canvas.itemconfig(self.show_image, image=self.img_nothing)
            
    def create_dialog_graph(self):
        dlg = tk.Toplevel(self)
        dlg.title("分析レポート")
        dlg.geometry("700x650")
        
        notebook = ttk.Notebook(dlg)
        notebook.pack(expand=True, fill='both', padx=20, pady=20)
        tab1 = ttk.Frame(
            notebook,
            width=600,
            height=600
        )
        tab2 = ttk.Frame(
            notebook,
            width=600,
            height=600
        )
        
        notebook.add(tab1, text="今日の結果")
        notebook.add(tab2, text="3日間の結果")
        
        fig_today = self.graph_today()
        canvas_today = FigureCanvasTkAgg(fig_today, master=tab1)
        canvas_today.draw()
        canvas_today.get_tk_widget().grid(column=0, row=0)
        
        fig_3days = self.graph_3days()
        canvas_3days = FigureCanvasTkAgg(fig_3days, master=tab2)
        canvas_3days.draw()
        canvas_3days.get_tk_widget().grid(column=0, row=0)
        
        label_kakugen = ttk.Label(
            tab1,
            text="【今日の格言】",
            foreground='white',
            font=("游明朝", 30)
        )
        label_kakugen.grid(column=0, row=1)
        
        msg = kakugen.random_generate()
        label_content = ttk.Label(
            tab1,
            text=msg,
            foreground='white',
            font=("游明朝", 30)
        )
        label_content.grid(column=0, row=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
    def graph_today(self):
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([
            self.progress_brush.get()*25,
            self.progress_drink.get()*25,
            self.progress_senobi.get()*25,
            self.progress_walk.get()*25,
            self.progress_face.get()*25
        ])
        label = np.array(['Brush', 'Drink', 'Senobi', 'Walk', 'Face'])
        ax.bar(x, y, tick_label=label)
        ax.set_ylim([0, 100])
        ax.set_ylabel("scores")
        return fig
    
    def graph_3days(self):
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot()
        x = np.array([1, 2, 3, 4])
        y_brush = np.array([100, 50, self.progress_brush.get()*25, 0])
        y_drink = np.array([0, 50, self.progress_drink.get()*25, 0])
        y_senobi = np.array([50, 100, self.progress_senobi.get()*25, 0])
        y_walk = np.array([25, 25, self.progress_walk.get()*25, 0])
        y_face = np.array([100, 25, self.progress_face.get()*25, 0])
        label = np.array(['7/2', '7/3', '7/4', ''])
        ax.bar(x, y_brush, tick_label=label, label='brush', color='lawngreen', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_drink, tick_label=label, label='drink', bottom=y_brush, color='lightskyblue', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_senobi, tick_label=label, label='senobi', bottom=y_drink+y_brush, color='lightsalmon', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_walk, tick_label=label, label='walk', bottom=y_senobi+y_drink+y_brush, color='yellow', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_face, tick_label=label, label='face', bottom=y_walk+y_senobi+y_drink+y_brush, color='aqua', edgecolor='black', linewidth=0.3)
        ax.legend(fontsize=7)
        ax.set_ylabel('scores')
        return fig
            
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
        mixer.init()
        mixer.music.load("sounds/start.mp3")
        mixer.music.play(1)
        time.sleep(0.5)
        mixer.music.load("sounds/csikos.mp3")
        mixer.music.play(1)
        
    def stop(self):
        self.after_cancel(self.jobID)
        mixer.init()
        mixer.music.load("sounds/finish.mp3")
        mixer.music.play(1)

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