import threading
import time
import tkinter as tk
from tkinter import ttk
from ttkthemes import *
from PIL import ImageTk, Image
from pygame import mixer
import torch
import arduino.arduino as arduino
import machine_learning.classifier as classifier

class App(ttk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        
        self.progress_brush = tk.IntVar(value=0)
        self.progress_drink = tk.IntVar(value=0)
        self.progress_senobi = tk.IntVar(value=0)
        self.progress_walk = tk.IntVar(value=0)
        self.progress_face = tk.IntVar(value=0)
        self.progress_all = tk.IntVar(value=0)
        
        self.img_nothing = ImageTk.PhotoImage(Image.open('images/sleep.png').resize((200,200)))
        self.img_brush = ImageTk.PhotoImage(Image.open('images/brush.png').resize((200,200)))
        self.img_drink = ImageTk.PhotoImage(Image.open('images/drink.png').resize((200,200)))
        self.img_senobi = ImageTk.PhotoImage(Image.open('images/senobi.png').resize((200,200)))
        self.img_walk = ImageTk.PhotoImage(Image.open('images/walk.png').resize((200,200)))
        self.img_face = ImageTk.PhotoImage(Image.open('images/washface.png').resize((200,200)))
        self.img_finish = ImageTk.PhotoImage(Image.open('images/finish.png').resize((200,200)))
        
    # メイン画面
        
    def main(self):
        t1 = threading.Thread(target=arduino.ArduinoRun, args=("t1",))
        t1.start()
        
        self.master.title("main window (wearable device)")
        self.master.geometry("600x300")
        
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
        
        self.label_brush_completed = ttk.Label(
            frame_actions,
            text="Washroom: Brush Teeth",
            foreground="white",
        )
        self.label_brush_completed.grid(column=0, row=1)
        
        # drink
    
        self.label_drink_completed = ttk.Label(
            frame_actions,
            text="Kitchen: Drink Water",
            foreground="white",
        )
        self.label_drink_completed.grid(column=0, row=2)
        
        # senobi
        
        self.label_senobi_completed = ttk.Label(
            frame_actions,
            text="Bedroom: Senobi",
            foreground="white",
        )
        self.label_senobi_completed.grid(column=0, row=3)
        
        # walk
        
        self.label_walk_completed = ttk.Label(
            frame_actions,
            text="EveryWhere: Walk",
            foreground="white",
        )
        self.label_walk_completed.grid(column=0, row=4)
        
        # face
        
        self.label_face_completed = ttk.Label(
            frame_actions,
            text="Washroom: Wash Face",
            foreground="white",
        )
        self.label_face_completed.grid(column=0, row=5)
        
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
        self.update()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.show_image = self.canvas.create_image(
            w / 2,
            h / 2,                   
            image=self.img_nothing
        )
    # 分析画面
            
    def create_dialog_graph(self):
        from app.graph_dialog import GraphDialog
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        
        self.dlg = tk.Toplevel(self)
        self.dlg.title("analysis report")
        self.dlg.geometry("700x650")
        
        app = GraphDialog(self.progress_brush, self.progress_drink, self.progress_senobi, self.progress_walk, self.progress_face, self.dlg)
        app.main()
    
    # 計測用画面
    
    def create_dialog_measure(self):
        from app.measure_dialog import MeasureDialog
        mixer.init()
        mixer.music.load("sounds/button.mp3")
        mixer.music.play(1)
        
        self.dlg = tk.Toplevel(self)
        self.dlg.title("measure & merge")
        self.dlg.geometry("500x500")
        
        app = MeasureDialog(self.dlg)
        app.main()
        
    # 更新処理
        
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
            
        self.jobID = self.after(10, self.loop)
        
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
        arduino.ble.stop()
        self.after_cancel(self.jobID)
        mixer.init()
        mixer.music.load("sounds/finish.mp3")
        mixer.music.play(1)

    def update_brush(self):
        if self.progress_brush.get() >= 4:
            self.canvas.itemconfig(self.show_image, image=self.img_brush)
        elif self.progress_brush.get() == 3:
            mixer.init()
            mixer.music.load("sounds/shining.mp3")
            mixer.music.play(1)
            self.progress_brush.set(self.progress_brush.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.label_brush_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_brush.set(self.progress_brush.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_brush)
            self.label_brush_completed["foreground"] = "yellow"
            
    def update_drink(self):
        if self.progress_drink.get() >= 4:
            self.canvas.itemconfig(self.show_image, image=self.img_drink)
        elif self.progress_drink.get() == 3:
            mixer.init()
            mixer.music.load("sounds/shining.mp3")
            mixer.music.play(1)
            self.progress_drink.set(self.progress_drink.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.label_drink_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_drink.set(self.progress_drink.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_drink)
            self.label_drink_completed["foreground"] = "yellow"
            
    def update_senobi(self):
        if self.progress_senobi.get() >= 4:
            self.canvas.itemconfig(self.show_image, image=self.img_senobi)
        elif self.progress_senobi.get() == 3:
            mixer.init()
            mixer.music.load("sounds/shining.mp3")
            mixer.music.play(1)
            self.progress_senobi.set(self.progress_senobi.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.label_senobi_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_senobi.set(self.progress_senobi.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_senobi),
            self.label_senobi_completed["foreground"] = "yellow"
            
    def update_walk(self):
        if self.progress_walk.get() >= 4:
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
        elif self.progress_walk.get() == 3:
            mixer.init()
            mixer.music.load("sounds/shining.mp3")
            mixer.music.play(1)
            self.progress_walk.set(self.progress_walk.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.label_walk_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_walk.set(self.progress_walk.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
            self.label_walk_completed["foreground"] = "yellow"
            
    def update_face(self):
        if self.progress_face.get() >= 4:
            self.canvas.itemconfig(self.show_image, image=self.img_face)
        elif self.progress_face.get() == 3:
            mixer.init()
            mixer.music.load("sounds/shining.mp3")
            mixer.music.play(1)
            self.progress_face.set(self.progress_face.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.label_face_completed["foreground"] = "lime green"
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_face.set(self.progress_face.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.canvas.itemconfig(self.show_image, image=self.img_face)
            self.label_face_completed["foreground"] = "yellow"
            
    def update_nothing(self):
        self.canvas.itemconfig(self.show_image, image=self.img_nothing)

if __name__ == "__main__":
    root = ThemedTk()
    model = classifier.load_model(
        model_path='models/model_hirata.pth',
        input_dim=9,
        hidden_dim=128,
        target_dim=5
    )
    app = App(master = root)
    app.main()
    app.mainloop()
    arduino.ble.stop()