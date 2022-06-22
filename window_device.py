import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from playsound import playsound

class App(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        self.master.title("ウェアラブルデバイスの画面")
        self.master.geometry("500x600")
        
        ttk.Style().theme_use('classic')
        
        # frame1
        
        self.frame1 = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=250
        )
        self.frame1.grid(column=0, row=0)
        
        # brush
        
        self.label_brush = ttk.Label(
            self.frame1,
            text="brush"
        )
        self.label_brush.grid(column=0, row=0)
        
        self.progress_brush = tk.IntVar(0)
        self.pbar_brush = ttk.Progressbar(
            self.frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_brush,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_brush.grid(column=1, row=0)
        
        self.button_brush = ttk.Button(
            self.frame1,
            text="brush",
            command=self.button_brush
        )
        self.button_brush.grid(column=2, row=0)
        
        self.brush_completed = tk.StringVar()
        self.brush_completed.set("Not yet")
        self.label_brush_completed = ttk.Label(
            self.frame1,
            textvariable=self.brush_completed
        )
        self.label_brush_completed.grid(column=3, row=0)
        
        # drink
        
        self.label_drink = ttk.Label(
            self.frame1,
            text="drink"
        )
        self.label_drink.grid(column=0, row=1)
        
        self.progress_drink = tk.IntVar(0)
        self.pbar_drink = ttk.Progressbar(
            self.frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_drink,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_drink.grid(column=1, row=1)
        
        self.button_drink = ttk.Button(
            self.frame1,
            text="drink",
            command=self.button_drink
        )
        self.button_drink.grid(column=2, row=1)
        
        self.drink_completed = tk.StringVar()
        self.drink_completed.set("Not yet")
        self.label_drink_completed = ttk.Label(
            self.frame1,
            textvariable=self.drink_completed
        )
        self.label_drink_completed.grid(column=3, row=1)
        
        # senobi
        
        self.label_senobi = ttk.Label(
            self.frame1,
            text="senobi"
        )
        self.label_senobi.grid(column=0, row=2)
        
        self.progress_senobi = tk.IntVar(0)
        self.pbar_senobi = ttk.Progressbar(
            self.frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_senobi,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_senobi.grid(column=1, row=2)
        
        self.button_senobi = ttk.Button(
            self.frame1,
            text="senobi",
            command=self.button_senobi
        )
        self.button_senobi.grid(column=2, row=2)
        
        self.senobi_completed = tk.StringVar()
        self.senobi_completed.set("Not yet")
        self.label_senobi_completed = ttk.Label(
            self.frame1,
            textvariable=self.senobi_completed
        )
        self.label_senobi_completed.grid(column=3, row=2)
        
        # walk
        
        self.label_walk = ttk.Label(
            self.frame1,
            text="senobi"
        )
        self.label_walk.grid(column=0, row=3)
        
        self.progress_walk = tk.IntVar(0)
        self.pbar_walk = ttk.Progressbar(
            self.frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_walk,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_walk.grid(column=1, row=3)
        
        self.button_walk = ttk.Button(
            self.frame1,
            text="walk",
            command=self.button_walk
        )
        self.button_walk.grid(column=2, row=3)
        
        self.walk_completed = tk.StringVar()
        self.walk_completed.set("Not yet")
        self.label_walk_completed = ttk.Label(
            self.frame1,
            textvariable=self.walk_completed
        )
        self.label_walk_completed.grid(column=3, row=3)
        
        # face
        
        self.label_face = ttk.Label(
            self.frame1,
            text="face"
        )
        self.label_face.grid(column=0, row=4)
        
        self.progress_face = tk.IntVar(0)
        self.pbar_face = ttk.Progressbar(
            self.frame1,
            orient=tk.HORIZONTAL,
            variable=self.progress_face,
            maximum=4,
            length=200,
            mode="determinate"
        )
        self.pbar_face.grid(column=1, row=4)
        
        self.button_face = ttk.Button(
            self.frame1,
            text="face",
            command=self.button_face
        )
        self.button_face.grid(column=2, row=4)
        
        self.face_completed = tk.StringVar()
        self.face_completed.set("Not yet")
        self.label_face_completed = ttk.Label(
            self.frame1,
            textvariable=self.face_completed
        )
        self.label_face_completed.grid(column=3, row=4)
        
        # frame2
        
        self.frame2 = ttk.Frame(
            master,
            padding=10,
            width=400,
            height=400
        )
        self.frame2.grid(column=0, row=1)
        
        self.canvas = tk.Canvas(self.frame2)
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

    def button_brush(self):
        self.progress_brush.set(self.progress_brush.get() + 1)
            
        if self.progress_brush.get() >= 4:
            self.brush_completed.set("Task Completed!")
            self.canvas.itemconfig(self.show_image, image=self.img_finish)
            playsound("sounds/jenga.mp3")
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
            playsound("sounds/jenga.mp3")
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
            playsound("sounds/jenga.mp3")
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
            playsound("sounds/jenga.mp3")
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
            playsound("sounds/jenga.mp3")
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