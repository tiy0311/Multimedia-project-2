#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cherrypy, os, urllib, pickle
from numpy import *

# makes the index URL accessible and the last line starts the CherryPy web server with configurations read from service.conf.

class SearchDemo:
    
    def __init__(self):
        # load list of images
        now_path =  os.path.dirname(os.path.realpath(__file__))
        self.imlist = [now_path+'/dataset/ukbench00001.jpg',now_path+'/dataset/ukbench00002.jpg',now_path+'/dataset/ukbench00003.jpg',now_path+'/dataset/ukbench00004.jpg',now_path+'/dataset/ukbench00005.jpg']

        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)
        
        
        # set max number of results to show
        self.maxres = 5
        
        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search example</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """
        
    def index(self,query=None):
        
        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:
            # query the database and get top images
            for imname in self.imlist:
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='100' />"
                html += "</a>"
        else:
            # show random selection if no query
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='100' />"
                html += "</a>"
                
        html += self.footer
        return html
    
    index.exposed = True

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
