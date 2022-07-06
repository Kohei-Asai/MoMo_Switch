import threading
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
import os
import arduino.arduino as arduino
import machine_learning.classifier as classifier
import machine_learning.data_readmake as data_readmake
import kakugen

class App(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        self.master.title("main window (wearable device)")
        self.master.geometry("600x300")
        
        self.progress_brush = tk.IntVar(value=0)
        self.progress_drink = tk.IntVar(value=0)
        self.progress_senobi = tk.IntVar(value=0)
        self.progress_walk = tk.IntVar(value=0)
        self.progress_face = tk.IntVar(value=0)
        self.progress_all = tk.IntVar(value=0)
        
        self.done_brush = False
        self.done_drink = False
        self.done_senobi = False
        self.done_walk = False
        self.done_face = False
        
        self.create_widgets()
        
    def create_widgets(self):
        self.style = ttk.Style()
        self.style.theme_use('black')
        
        self.frame_main = ttk.Frame(
            self.master,
            padding=10,
            width=600,
            height=300,
        )
        self.frame_main.pack(expand=True, fill='both')
        
        self.create_frame_pbar()
        self.create_frame_actions()
        self.create_frame_measure_report()
        self.create_frame_start_stop()
        self.create_frame_image()
        
    def create_frame_pbar(self):
        frame_pbar = ttk.Frame(
            self.frame_main,
            padding=10,
            width=400,
            height=250
        )
        frame_pbar.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_all = ttk.Label(
            frame_pbar,
            text="score"
        )
        label_all.grid(column=0, row=0)
        
        self.style.configure("red.Horizontal.TProgressbar", background='green2')

        pbar_all = ttk.Progressbar(
            frame_pbar,
            orient=tk.HORIZONTAL,
            variable=self.progress_all,
            maximum=20,
            length=200,
            mode="determinate",
            style="red.Horizontal.TProgressbar"
        )
        pbar_all.grid(column=1, row=0)
        
    def create_frame_actions(self):
        frame_actions = ttk.Frame(
            self.frame_main,
            padding=10,
            width=400,
            height=400
        )
        frame_actions.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # brush
        
        button_brush = ttk.Button(
            frame_actions,
            text="brush",
            command=self.update_brush
        )
        button_brush.grid(column=0, row=1)
        
        self.label_brush_completed = ttk.Label(
            frame_actions,
            text="Washroom: Brush Teeth",
            foreground="white",
        )
        self.label_brush_completed.grid(column=1, row=1)
        
        # drink
        
        button_drink = ttk.Button(
            frame_actions,
            text="drink",
            command=self.update_drink
        )
        button_drink.grid(column=0, row=2)
    
        self.label_drink_completed = ttk.Label(
            frame_actions,
            text="Kitchen: Drink Water",
            foreground="white",
        )
        self.label_drink_completed.grid(column=1, row=2)
        
        # senobi
        
        button_senobi = ttk.Button(
            frame_actions,
            text="senobi",
            command=self.update_senobi
        )
        button_senobi.grid(column=0, row=3)
        
        self.label_senobi_completed = ttk.Label(
            frame_actions,
            text="Bedroom: Senobi",
            foreground="white",
        )
        self.label_senobi_completed.grid(column=1, row=3)
        
        # walk
        
        button_walk = ttk.Button(
            frame_actions,
            text="walk",
            command=self.update_walk
        )
        button_walk.grid(column=0, row=4)
        
        self.label_walk_completed = ttk.Label(
            frame_actions,
            text="EveryWhere: Walk",
            foreground="white",
        )
        self.label_walk_completed.grid(column=1, row=4)
        
        # face
        
        button_face = ttk.Button(
            frame_actions,
            text="face",
            command=self.update_face
        )
        button_face.grid(column=0, row=5)
        
        self.label_face_completed = ttk.Label(
            frame_actions,
            text="Washroom: Wash Face",
            foreground="white",
        )
        self.label_face_completed.grid(column=1, row=5)
        
    def create_frame_measure_report(self):
        frame_measure_report = ttk.Frame(
            self.frame_main,
            padding=10,
            width=400,
            height=400
        )
        frame_measure_report.grid(column=0, row=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_measure = ttk.Button(
            frame_measure_report,
            text="measure",
            command=self.create_dialog_measure
        )
        button_measure.grid(column=0, row=0)
        
        button_report = ttk.Button(
            frame_measure_report,
            text="report",
            command=self.create_dialog_graph
        )
        button_report.grid(column=1, row=0)
        
    def create_frame_start_stop(self):
        frame_start_stop = ttk.Frame(
            self.frame_main,
            padding=10,
            width=400,
            height=400
        )
        frame_start_stop.grid(column=1, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_start = ttk.Button(
            frame_start_stop,
            text="start",
            command=self.start
        )
        button_start.grid(column=0, row=0)
        
        button_stop = ttk.Button(
            frame_start_stop,
            text="stop",
            command=self.stop
        )
        button_stop.grid(column=1, row=0)
        
    def create_frame_image(self):
        frame_image = ttk.Frame(
            self.frame_main,
            padding=10,
            width=400,
            height=400
        )
        frame_image.grid(column=1, row=1, rowspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.canvas = tk.Canvas(
            frame_image,
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
        
    def loop(self):
        data = []
        for _ in range(100):
            acc = [float(arduino.ble.accx),
                    float(arduino.ble.accy),
                    float(arduino.ble.accz),
                    float(arduino.ble.gyrx),
                    float(arduino.ble.gyry),
                    float(arduino.ble.gyrz),
                    float(arduino.ble.magx),
                    float(arduino.ble.magy),
                    float(arduino.ble.magz)]
            data.append(acc)
            time.sleep(1/100)

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
        t1 = threading.Thread(target=arduino.ArduinoRun, args=("t1",))
        t1.start()
        self.after(10, self.loop)
        mixer.init()
        mixer.music.load("sounds/start.mp3")
        mixer.music.play(1)
        time.sleep(0.7)
        mixer.music.load("sounds/PeerGynt.mp3")
        mixer.music.play(1)
        
    def stop(self):
        self.after_cancel(self.jobID)
        arduino.ble.stop()
        mixer.init()
        mixer.music.load("sounds/finish.mp3")
        mixer.music.play(1)

    def update_brush(self):
        if self.progress_brush.get() >= 4:
            self.label_brush_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
            if self.done_brush == False:
                self.done_brush = True
                mixer.init()
                mixer.music.load("sounds/shining.mp3")
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
                mixer.music.load("sounds/shining.mp3")
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
                mixer.music.load("sounds/shining.mp3")
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
                mixer.music.load("sounds/shining.mp3")
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
                mixer.music.load("sounds/shining.mp3")
                mixer.music.play(1)
        else:
            self.progress_face.set(self.progress_face.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_face)
            self.label_face_completed["foreground"] = "yellow"
            
    def update_nothing(self):
        self.canvas.itemconfig(self.show_image, image=self.img_nothing)
        
    # 分析結果
            
    def create_dialog_graph(self):
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        
        dlg = tk.Toplevel(self)
        dlg.title("analysis report")
        dlg.geometry("700x650")
        
        notebook = ttk.Notebook(dlg)
        notebook.pack(expand=True, fill='both')
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
        
        notebook.add(tab1, text="today")
        notebook.add(tab2, text="3 days")
        
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
    
    # 計測用画面
    
    def create_dialog_measure(self):
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        
        dlg = tk.Toplevel(self)
        dlg.title("measure & merge")
        dlg.geometry("500x500")
        
        self.count = 0
        
        frame = ttk.Frame(
            dlg,
            padding=10,
        )
        frame.pack(expand=True, fill='both')
        
        # 1. create your directry
        
        frame_yourdir = ttk.Frame(
            frame,
            padding=10,
            width=500,
            height=100
        )
        frame_yourdir.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_yourdir_explanation = ttk.Label(
            frame_yourdir,
            padding=5,
            text='1. Create Your Directory'
        )
        label_yourdir_explanation.grid(column=0, row=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_yourdir_yourname = ttk.Label(
            frame_yourdir,
            padding=5,
            text='Your Name'
        )
        label_yourdir_yourname.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.entry_yourdir = ttk.Entry(
            frame_yourdir,
        )
        self.entry_yourdir.grid(column=1, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_yourdir = ttk.Button(
            frame_yourdir,
            text='create',
            command=self.create_your_directory
        )
        button_yourdir.grid(column=2, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # 2. create each action's directory
        
        frame_action_label = ttk.Frame(
            frame,
            padding=10,
            width=500,
            height=100
        )
        frame_action_label.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_action_explanation = ttk.Label(
            frame_action_label,
            padding=5,
            text='2. Create Directory of Each Action'
        )
        label_action_explanation.grid(column=0, row=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_action_label = ttk.Label(
            frame_action_label,
            padding=5,
            text='Action Name'
        )
        label_action_label.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.entry_action = ttk.Entry(
            frame_action_label,
        )
        self.entry_action.grid(column=1, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_action = ttk.Button(
            frame_action_label,
            text='create',
            command=self.create_action_directory
        )
        button_action.grid(column=2, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # 3. measure your data
        
        frame_measure = ttk.Frame(
            frame,
            padding=10,
            width=500,
            height=200
        )
        frame_measure.grid(column=0, row=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_measure_explanation = ttk.Label(
            frame_measure,
            padding=5,
            text='3. Measure Your Data'
        )
        label_measure_explanation.grid(column=0, row=0, columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_start = ttk.Button(
            frame_measure,
            padding=5,
            text='start',
            command=self.start_study
        )
        button_start.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_stop = ttk.Button(
            frame_measure,
            padding=5,
            text='stop',
            command=self.stop_measure
        )
        button_stop.grid(column=1, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.labelvar = tk.StringVar("")
        self.labelvar.set("ready to start")
        label_doing = ttk.Label(
            frame_measure,
            padding=5,
            textvariable=self.labelvar
        )
        label_doing.grid(column=0, row=2, columnspan=2)
        
        button_save = ttk.Button(
            frame_measure,
            padding=5,
            text='save',
            command=self.save
        )
        button_save.grid(column=0, row=3, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.count_label = tk.StringVar("")
        self.count_label.set(str(self.count) + "/10 completed")
        label_donecount = ttk.Label(
            frame_measure,
            padding=5,
            textvariable=self.count_label
        )
        label_donecount.grid(column=1, row=3)
        
        # 4. merge data
        
        frame_merge = ttk.Frame(
            frame,
            padding=10,
            width=500,
            height=200
        )
        frame_merge.grid(column=0, row=3, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        label_merge_explanation = ttk.Label(
            frame_merge,
            padding=5,
            text='3. Merge Data'
        )
        label_merge_explanation.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        button_merge = ttk.Button(
            frame_merge,
            padding=5,
            text='merge',
            command=self.merge
        )
        button_merge.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
    def create_your_directory(self):
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        name = self.entry_yourdir.get()
        os.mkdir("data/"+name)
        
    def create_action_directory(self):
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        self.count = 0
        self.count_label.set(str(self.count) + "/10 completed")
        name = self.entry_yourdir.get()
        action = self.entry_action.get()
        os.mkdir("data/"+name+"/"+action)
        
    def start_study(self):
        t1 = threading.Thread(target=arduino.ArduinoRun, args=("t1",))
        t1.start()
        self.after(10, self.measure)
        mixer.init()
        mixer.music.load("sounds/start.mp3")
        mixer.music.play(1)
        self.labelvar.set("doing......")
        
    def measure(self):
        self.data = []
        self.after(1000, self.loop_measure)
        
    def loop_measure(self):
        acc = [float(arduino.ble.accx),
                    float(arduino.ble.accy),
                    float(arduino.ble.accz),
                    float(arduino.ble.gyrx),
                    float(arduino.ble.gyry),
                    float(arduino.ble.gyrz),
                    float(arduino.ble.magx),
                    float(arduino.ble.magy),
                    float(arduino.ble.magz)]
        if acc != [0,0,0,0,0,0,0,0,0]:
            self.data.append(acc)
        self.jobID = self.after(10, self.loop_measure)
        
    def stop_measure(self):
        self.after_cancel(self.jobID)
        arduino.ble.stop()
        mixer.init()
        mixer.music.load("sounds/finish.mp3")
        mixer.music.play(1)
        self.labelvar.set("wait for saving")
    
    def save(self):
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        name = self.entry_yourdir.get()
        action = self.entry_action.get()
        cindex = classifier.category2index[action]
        self.count = self.count + 1
        save_path = 'data/'+name+'/'+action+'/'+action+str(self.count)+'.csv'
        data_readmake.make_csvfile_onecategory(self.data, cindex, save_path)
        self.count_label.set(str(self.count) + "/10 completed")
        self.labelvar.set("ready to start")
        
    def merge(self):
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        name = self.entry_yourdir.get()
        yourpath = 'data/'+name
        data_readmake.make_merged_csvfiles(yourpath)

if __name__ == "__main__":
    root = ThemedTk()
    model = classifier.load_model(
        model_path='models/model_9freedom.pth',
        input_dim=9,
        hidden_dim=128,
        target_dim=5
    )
    app = App(master = root)
    app.mainloop()