import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

class App(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        self.master.title("ウェアラブルデバイスの画面")
        self.master.geometry("500x600")
        
        ttk.Style().theme_use('classic')
        
        # frame_brush
        
        self.frame_brush = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=50
        )
        self.frame_brush.pack(side = tk.TOP)
        
        self.label_brush = ttk.Label(
            self.frame_brush,
            text="brush"
        )
        self.label_brush.pack(side = tk.LEFT)
        
        self.progress_brush = tk.IntVar(0)
        self.pbar_brush = ttk.Progressbar(
            self.frame_brush,
            orient=tk.HORIZONTAL,
            variable=self.progress_brush,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_brush.pack(side = tk.LEFT)
        
        self.button_brush = ttk.Button(
            self.frame_brush,
            text="brush",
            command=self.button_brush
        )
        self.button_brush.pack(side=tk.LEFT)
        
        self.brush_completed = tk.StringVar()
        self.brush_completed.set("Not yet")
        self.label_brush_completed = ttk.Label(
            self.frame_brush,
            textvariable=self.brush_completed
        )
        self.label_brush_completed.pack(side = tk.LEFT)
        
        # frame_drink
        
        self.frame_drink = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=50
        )
        self.frame_drink.pack(side = tk.TOP)
        
        self.label_drink = ttk.Label(
            self.frame_drink,
            text="drink"
        )
        self.label_drink.pack(side = tk.LEFT)
        
        self.progress_drink = tk.IntVar(0)
        self.pbar_drink = ttk.Progressbar(
            self.frame_drink,
            orient=tk.HORIZONTAL,
            variable=self.progress_drink,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_drink.pack(side = tk.LEFT)
        
        self.button_drink = ttk.Button(
            self.frame_drink,
            text="drink",
            command=self.button_drink
        )
        self.button_drink.pack(side=tk.LEFT)
        
        self.drink_completed = tk.StringVar()
        self.drink_completed.set("Not yet")
        self.label_drink_completed = ttk.Label(
            self.frame_drink,
            textvariable=self.drink_completed
        )
        self.label_drink_completed.pack(side = tk.LEFT)
        
        # frame_senobi
        
        self.frame_senobi = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=50
        )
        self.frame_senobi.pack(side = tk.TOP)
        
        self.label_senobi = ttk.Label(
            self.frame_senobi,
            text="senobi"
        )
        self.label_senobi.pack(side = tk.LEFT)
        
        self.progress_senobi = tk.IntVar(0)
        self.pbar_senobi = ttk.Progressbar(
            self.frame_senobi,
            orient=tk.HORIZONTAL,
            variable=self.progress_senobi,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_senobi.pack(side = tk.LEFT)
        
        self.button_senobi = ttk.Button(
            self.frame_senobi,
            text="senobi",
            command=self.button_senobi
        )
        self.button_senobi.pack(side=tk.LEFT)
        
        self.senobi_completed = tk.StringVar()
        self.senobi_completed.set("Not yet")
        self.label_senobi_completed = ttk.Label(
            self.frame_senobi,
            textvariable=self.senobi_completed
        )
        self.label_senobi_completed.pack(side = tk.LEFT)
        
        # frame_walk
        
        self.frame_walk = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=50
        )
        self.frame_walk.pack(side = tk.TOP)
        
        self.label_walk = ttk.Label(
            self.frame_walk,
            text="senobi"
        )
        self.label_walk.pack(side = tk.LEFT)
        
        self.progress_walk = tk.IntVar(0)
        self.pbar_walk = ttk.Progressbar(
            self.frame_walk,
            orient=tk.HORIZONTAL,
            variable=self.progress_walk,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_walk.pack(side = tk.LEFT)
        
        self.button_walk = ttk.Button(
            self.frame_walk,
            text="walk",
            command=self.button_walk
        )
        self.button_walk.pack(side=tk.LEFT)
        
        self.walk_completed = tk.StringVar()
        self.walk_completed.set("Not yet")
        self.label_walk_completed = ttk.Label(
            self.frame_walk,
            textvariable=self.walk_completed
        )
        self.label_walk_completed.pack(side = tk.LEFT)
        
        # frame_face
        
        self.frame_face = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=50
        )
        self.frame_face.pack(side = tk.TOP)
        
        self.label_face = ttk.Label(
            self.frame_face,
            text="face"
        )
        self.label_face.pack(side = tk.LEFT)
        
        self.progress_face = tk.IntVar(0)
        self.pbar_face = ttk.Progressbar(
            self.frame_face,
            orient=tk.HORIZONTAL,
            variable=self.progress_face,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_face.pack(side = tk.LEFT)
        
        self.button_face = ttk.Button(
            self.frame_face,
            text="face",
            command=self.button_face
        )
        self.button_face.pack(side=tk.LEFT)
        
        self.face_completed = tk.StringVar()
        self.face_completed.set("Not yet")
        self.label_face_completed = ttk.Label(
            self.frame_face,
            textvariable=self.face_completed
        )
        self.label_face_completed.pack(side = tk.LEFT)
        
        # frame3
        
        self.frame3 = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=400
        )
        self.frame3.pack(side = tk.TOP)
        
        self.text_completed = tk.StringVar("")
        self.label2 = ttk.Label(
            self.frame3,
            textvariable=self.text_completed
        )
        self.label2.pack(side=tk.TOP)
        
        self.canvas = tk.Canvas(self.frame3)
        self.canvas.pack()
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

    def button_brush(self):
        self.progress_brush.set(self.progress_brush.get() + 1)
            
        if self.progress_brush.get() >= 4:
            self.brush_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        elif  1 <= self.progress_brush.get() <= 3:
            self.brush_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_brush)
        else:
            self.brush_completed.set("Not yet")
            self.canvas.itemconfig(self.show_image, image=self.img_sleep)
            
    def button_drink(self):
        self.progress_drink.set(self.progress_drink.get() + 1)
            
        if self.progress_drink.get() >= 4:
            self.drink_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        elif  1 <= self.progress_drink.get() <= 3:
            self.drink_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_drink)
        else:
            self.drink_completed.set("Not yet")
            self.canvas.itemconfig(self.show_image, image=self.img_sleep)
            
    def button_senobi(self):
        self.progress_senobi.set(self.progress_senobi.get() + 1)
            
        if self.progress_senobi.get() >= 4:
            self.senobi_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        elif  1 <= self.progress_senobi.get() <= 3:
            self.senobi_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_senobi)
        else:
            self.senobi_completed.set("Not yet")
            self.canvas.itemconfig(self.show_image, image=self.img_sleep)
            
    def button_walk(self):
        self.progress_walk.set(self.progress_walk.get() + 1)
            
        if self.progress_walk.get() >= 4:
            self.walk_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        elif  1 <= self.progress_senobi.get() <= 3:
            self.walk_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
        else:
            self.walk_completed.set("Not yet")
            self.canvas.itemconfig(self.show_image, image=self.img_walk)
            
    def button_face(self):
        self.progress_face.set(self.progress_face.get() + 1)
            
        if self.progress_face.get() >= 4:
            self.face_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
        elif  1 <= self.progress_face.get() <= 3:
            self.face_completed.set("Doing......")
            self.canvas.itemconfig(self.show_image, image=self.img_face)
        else:
            self.face_completed.set("Not yet")
            self.canvas.itemconfig(self.show_image, image=self.img_sleep)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(master = root)
    app.mainloop()