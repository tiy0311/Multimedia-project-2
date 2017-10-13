#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pylab import *

import pickle
import sift

from Tkinter import *
import ImageTk 
import tkMessageBox
from ttk import Frame, Button, Label, Style

from random import randint
from PIL import Image


class Example(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        
        self.initUI()
    
        
    def initUI(self):
      
        self.parent.title("Listbox") 

        self.pack(fill=BOTH, expand=1)

        acts = ['mode 1', 'mode 2', 
            'mode 3', 'mode 4', 'mode 5', 'mode 6']

        lb = Listbox(self)
        for i in acts:
            lb.insert(END, i)
            
        lb.bind("<<ListboxSelect>>", self.onSelect)    
            
        lb.place(x=400, y=400)

        self.var = StringVar()
        self.label = Label(self, text=0, textvariable=self.var)        
        self.label.place(x=20, y=410)
        
        abtn = Button(self, image=img[0], command = lambda: helloCallBack(img[0],self.var.get()))
        abtn.grid(row=0, column=0, pady=5)

        bbtn = Button(self, image=img[1], command = lambda: helloCallBack(img[1],self.var.get()))
        bbtn.grid(row=0, column=1, pady=5)

        cbtn = Button(self, image=img[2], command = lambda: helloCallBack(img[2],self.var.get()))
        cbtn.grid(row=0, column=2, pady=4)

        dbtn = Button(self, image=img[3], command = lambda: helloCallBack(img[3],self.var.get()))
        dbtn.grid(row=1, column=0, pady=4)

        ebtn = Button(self, image=img[4], command = lambda: helloCallBack(img[4],self.var.get()))
        ebtn.grid(row=1, column=1, pady=5)

        fbtn = Button(self, image=img[5], command = lambda: helloCallBack(img[5],self.var.get()))
        fbtn.grid(row=1, column=2, pady=4)

        gbtn = Button(self, image=img[6], command = lambda: helloCallBack(img[6],self.var.get()))
        gbtn.grid(row=3, column=0, pady=5)

        hbtn = Button(self, image=img[7], command = lambda: helloCallBack(img[7],self.var.get()))
        hbtn.grid(row=3, column=1, pady=4)

        ibtn = Button(self, image=img[8], command = lambda: helloCallBack(img[8],self.var.get()))
        ibtn.grid(row=3, column=2, pady=4)



    def onSelect(self, val):
      
        sender = val.widget
        idx = sender.curselection()
        value = sender.get(idx)   

        self.var.set(value)
 


def helloCallBack(im,val):
    print val
    tkMessageBox.showinfo(title="Greetings", message=str(val))

    ### Your stuff here ###


if __name__ == '__main__':
    root = Tk()
    size = 128, 128
    imname = []
    im = []
    img = []

    for i in range(10):
        imname.append(0)
        im.append(0)
        img.append(0)

        ### This is an example. ### 
        ### You can change your own ### 
        
        random = randint(0,103)

        imname[i] = 'dataset/ukbench00' + (3-len(str(random))) * '0' + str(random) + '.jpg'

        im[i] = Image.open(imname[i])
        im[i].thumbnail(size, Image.ANTIALIAS)

        img[i] = ImageTk.PhotoImage(im[i])

    app = Example(root)
    root.geometry("600x650+400+300")
    root.mainloop()

  