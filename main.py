import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import arduino

class App(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        master.title("ウェアラブルデバイスの画面")
        master.geometry("500x600")
        
        ttk.Style().theme_use('classic')
        
        # 変数
        
        self.progress_brush = tk.IntVar(0)
        self.progress_drink = tk.IntVar(0)
        self.progress_senobi = tk.IntVar(0)
        self.progress_walk = tk.IntVar(0)
        self.progress_face = tk.IntVar(0)
        self.progress_all = tk.IntVar(0)
        
        # frame1
        
        frame1 = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=250
        )
        frame1.grid(column=0, row=0)
        
        # brush
        
        button_brush = ttk.Button(
            frame1,
            text="brush",
            command=self.button_brush
        )
        button_brush.grid(column=0, row=0)
        
        self.brush_completed = tk.StringVar()
        self.brush_completed.set("Not yet")
        label_brush_completed = ttk.Label(
            frame1,
            textvariable=self.brush_completed
        )
        label_brush_completed.grid(column=1, row=0)
        
        # drink
        
        button_drink = ttk.Button(
            frame1,
            text="drink",
            command=self.button_drink
        )
        button_drink.grid(column=0, row=1)
        
        self.drink_completed = tk.StringVar()
        self.drink_completed.set("Not yet")
        label_drink_completed = ttk.Label(
            frame1,
            textvariable=self.drink_completed
        )
        label_drink_completed.grid(column=1, row=1)
        
        # senobi
        
        button_senobi = ttk.Button(
            frame1,
            text="senobi",
            command=self.button_senobi
        )
        button_senobi.grid(column=0, row=2)
        
        self.senobi_completed = tk.StringVar()
        self.senobi_completed.set("Not yet")
        label_senobi_completed = ttk.Label(
            frame1,
            textvariable=self.senobi_completed
        )
        label_senobi_completed.grid(column=1, row=2)
        
        # walk
        
        button_walk = ttk.Button(
            frame1,
            text="walk",
            command=self.button_walk
        )
        button_walk.grid(column=0, row=3)
        
        self.walk_completed = tk.StringVar()
        self.walk_completed.set("Not yet")
        label_walk_completed = ttk.Label(
            frame1,
            textvariable=self.walk_completed
        )
        label_walk_completed.grid(column=1, row=3)
        
        # face
        
        button_face = ttk.Button(
            frame1,
            text="face",
            command=self.button_face
        )
        button_face.grid(column=0, row=4)
        
        self.face_completed = tk.StringVar()
        self.face_completed.set("Not yet")
        label_face_completed = ttk.Label(
            frame1,
            textvariable=self.face_completed
        )
        label_face_completed.grid(column=1, row=4)
        
        # 共通化したやつ
        
        label_all = ttk.Label(
            frame1,
            text="進捗"
        )
        label_all.grid(column=0, row=5)

        pbar_all = ttk.Progressbar(
            frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_all,
            maximum=20,
            length=200,
            mode="determinate"
        )
        pbar_all.grid(column=1, row=5)
        
        # frame2
        
        frame2 = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=400
        )
        frame2.grid(column=0, row=1)
        
        self.canvas = tk.Canvas(frame2)
        self.canvas.grid(column=0, row=0)
        self.img_sleep = ImageTk.PhotoImage(Image.open('images/sleep.png').resize((200,200)))
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
            image=self.img_sleep
        )
        
        # frame 3
        
        frame3 = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=400
        )
        frame3.grid(column=0, row=2)
        
        button_report = ttk.Button(
            frame3,
            text="report",
            command=self.create_dialog_graph
        )
        button_report.grid(column=0, row=0)

    def button_brush(self):
        if self.progress_brush.get() >= 4:
            self.brush_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_brush.set(self.progress_brush.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.brush_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_brush)
            
    def button_drink(self):
        if self.progress_drink.get() >= 4:
            self.drink_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_drink.set(self.progress_drink.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.drink_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_drink)
            
    def button_senobi(self):
        if self.progress_senobi.get() >= 4:
            self.senobi_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_senobi.set(self.progress_senobi.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.senobi_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_senobi)
            
    def button_walk(self):
        if self.progress_walk.get() >= 4:
            self.walk_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_walk.set(self.progress_walk.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.walk_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
            
    def button_face(self):
        if self.progress_face.get() >= 4:
            self.face_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        else:
            self.progress_face.set(self.progress_face.get() + 1)
            self.progress_all.set(self.progress_all.get() + 1)
            self.face_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_face)
            
    def create_dialog_graph(self):
        dlg = tk.Toplevel(self)
        dlg.title("今日の結果")
        dlg.geometry("700x500")
        
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
        result = arduino.ramdom_generate()
        
        if result == 0:
            self.button_brush()
        elif result == 1:
            self.button_drink()
        elif result == 2:
            self.button_face()
        elif result == 3:
            self.button_walk()
        elif result == 4:
            self.button_senobi()
            
        self.after(1000, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(master = root)
    app.after(1000, app.loop)
    app.mainloop()