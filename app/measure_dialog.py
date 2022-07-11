import tkinter as tk
from tkinter import ttk
from pygame import mixer
import main
import os
import arduino.arduino as arduino
import machine_learning.classifier as classifier
import machine_learning.data_readmake as data_readmake


class MeasureDialog(main.App):
    def __init__(self, master=None):
        super().__init__(master)
        
    def main(self):
        self.count = 0
        
        self.frame_main = ttk.Frame(
            self.master,
            padding=10,
        )
        self.frame_main.pack(expand=True, fill='both')
        
        self.create_frame_yourdir()
        self.create_frame_action_label()
        self.create_frame_measure()
        self.create_frame_merge()
        
    def create_frame_yourdir(self):
        frame_yourdir = ttk.Frame(
            self.frame_main,
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
        
    def create_frame_action_label(self):
        frame_action_label = ttk.Frame(
            self.frame_main,
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
        
    def create_frame_measure(self):
        frame_measure = ttk.Frame(
            self.frame_main,
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
        
    def create_frame_merge(self):
        
        frame_merge = ttk.Frame(
            self.frame_main,
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