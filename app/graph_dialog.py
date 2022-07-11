import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import main
import app.kakugen as kakugen

class GraphDialog(main.App):
    def __init__(self, progress_brush, progress_drink, progress_senobi, progress_walk, progress_face, master=None):
        super().__init__(master)
        self.progress_brush = progress_brush
        self.progress_drink = progress_drink
        self.progress_senobi = progress_senobi
        self.progress_walk = progress_walk
        self.progress_face = progress_face
        
    def main(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both')
        
        self.create_tab1()
        self.create_tab2()
        
    def create_tab1(self):
        tab1 = ttk.Frame(
            self.notebook,
            width=600,
            height=600
        )
        self.notebook.add(tab1, text="today")
        
        fig_today = self.graph_today()
        canvas_today = FigureCanvasTkAgg(fig_today, master=tab1)
        canvas_today.draw()
        canvas_today.get_tk_widget().grid(column=0, row=0)
        
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
        
    def create_tab2(self):
        tab2 = ttk.Frame(
            self.notebook,
            width=600,
            height=600
        )
        
        self.notebook.add(tab2, text="3 days")
        
        fig_3days = self.graph_3days()
        canvas_3days = FigureCanvasTkAgg(fig_3days, master=tab2)
        canvas_3days.draw()
        canvas_3days.get_tk_widget().grid(column=0, row=0)
        
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
        label = np.array(['7/11', '7/12', '7/13', ''])
        ax.bar(x, y_brush, tick_label=label, label='brush', color='lawngreen', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_drink, tick_label=label, label='drink', bottom=y_brush, color='lightskyblue', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_senobi, tick_label=label, label='senobi', bottom=y_drink+y_brush, color='lightsalmon', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_walk, tick_label=label, label='walk', bottom=y_senobi+y_drink+y_brush, color='yellow', edgecolor='black', linewidth=0.3)
        ax.bar(x, y_face, tick_label=label, label='face', bottom=y_walk+y_senobi+y_drink+y_brush, color='aqua', edgecolor='black', linewidth=0.3)
        ax.legend(fontsize=7)
        ax.set_ylabel('scores')
        return fig