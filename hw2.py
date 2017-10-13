import os, sys, glob
import numpy as np
from scipy.spatial import distance
from scipy.fftpack import dct
from operator import itemgetter
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from Tkinter import *
import ttk
from PIL import Image, ImageTk
import sift
from pylab import *

dst_list_Q1_unsorted = []
dst_list_Q2_unsorted = []
dst_list_Q3_unsorted = []
dst_list_Q4_unsorted = []

targetArray = [x for x in range(1000)]
questionArray = ['Q1. Color Histogram', 'Q2. Color Layout', 'Q3. SIFT Visual Words', 'Q4. Q3 with Stop Words', 'Q5. Merge Q1, Q2, Q4']

def ChangeTargetToTargetAddress(num):
	target = 'dataset/ukbench00'
	if 0 <= num and num < 10:
		target += '00'
	elif 10 <= num and num < 100:
		target += '0'
	target += str(num) + '.jpg'
	return target

def showDistanceAndID(dst_list):
	dis1.config(text=str(dst_list[0][1]))
	dis2.config(text=str(dst_list[1][1]))
	dis3.config(text=str(dst_list[2][1]))
	dis4.config(text=str(dst_list[3][1]))
	dis5.config(text=str(dst_list[4][1]))
	dis6.config(text=str(dst_list[5][1]))
	dis7.config(text=str(dst_list[6][1]))
	dis8.config(text=str(dst_list[7][1]))
	dis9.config(text=str(dst_list[8][1]))
	dis10.config(text=str(dst_list[9][1]))
	id1.config(text=str(dst_list[0][0]))
	id2.config(text=str(dst_list[1][0]))
	id3.config(text=str(dst_list[2][0]))
	id4.config(text=str(dst_list[3][0]))
	id5.config(text=str(dst_list[4][0]))
	id6.config(text=str(dst_list[5][0]))
	id7.config(text=str(dst_list[6][0]))
	id8.config(text=str(dst_list[7][0]))
	id9.config(text=str(dst_list[8][0]))
	id10.config(text=str(dst_list[9][0]))

def print_similarity(dst_list):
	for x in range(9):
		print "No.0%d --- %s (%s)" % (x+1, dst_list[x][0], dst_list[x][1])
	print "No.10 --- %s (%s)" % (dst_list[9][0], dst_list[9][1])
	print "----------------------------------------------"

def Q1(qname):
	im_target = Image.open(qname)
	im_target_histo = im_target.histogram()
	global dst_list_Q1_unsorted
	dst_list_Q1_unsorted = []
	dst_list_Q1 = []

	for fname in glob.glob('dataset/*.jpg'):
		#print fname
		im = Image.open(fname)
		im_histo = im.histogram()

		# Euclidean distance
		#dst = distance.euclidean(im_target_histo, im_histo)

		# Manhattan distance
		dst = distance.cityblock(im_target_histo, im_histo)

		dst_list_Q1.append((fname,dst))

	
	dst_list_Q1_unsorted = dst_list_Q1

	# sort by distance
	dst_list_Q1.sort(key=itemgetter(1))
	showDistanceAndID(dst_list_Q1)
	print "Q1. Using Color Histogram---------------------"
	print_similarity(dst_list_Q1)

def Q2(qname):
	im_target = Image.open(qname)
	def average(pixel, x, y, width, height):
		R_list, G_list, B_list = [], [], []
		for X in range(x, x+(width/8)):
			for Y in range(y, y+(height/8)):
				R, G, B = pixel[X,Y]
				R_list.append(R)
				G_list.append(G)
				B_list.append(B)
		return  np.mean(R_list), np.mean(G_list), np.mean(B_list)
                

	def rgb_ycbcr(avg_list):
		Y_list, CB_list, CR_list = [], [], []
		for x in range(64):
			y = .299*avg_list[x][0] + .587*avg_list[x][1] + .114*avg_list[x][2]
			cb = 128 -.168736*avg_list[x][0] -.331364*avg_list[x][1] + .5*avg_list[x][2]
			cr = 128 +.5*avg_list[x][0] - .418688*avg_list[x][1] - .081312*avg_list[x][2]
			Y_list.append(y)
			CB_list.append(cb)
			CR_list.append(cr)

		return Y_list, CB_list, CR_list


	def zigzag(dct_y, dct_cb, dct_cr):
		DY, DCb, DCr = [], [], []
    
		temp = 0
		DY.append(dct_y[temp])
		DCb.append(dct_cb[temp])
		DCr.append(dct_cr[temp])
		for x in range(1,8):
			if (x % 2 != 0):
				temp += 1
				DY.append(dct_y[temp])
				DCb.append(dct_cb[temp])
				DCr.append(dct_cr[temp])
			else:
				temp += 8
				DY.append(dct_y[temp])
				DCb.append(dct_cb[temp])
				DCr.append(dct_cr[temp])
			for y in range(x):
				if (x % 2 == 0):
					temp -= 7
					DY.append(dct_y[temp])
					DCb.append(dct_cb[temp])
					DCr.append(dct_cr[temp])
				else:
					temp += 7
					DY.append(dct_y[temp])
					DCb.append(dct_cb[temp])
					DCr.append(dct_cr[temp])
   
		for x in range(8,1,-1):
			if (x % 2 == 0):
				temp += 1
				DY.append(dct_y[temp])
				DCb.append(dct_cb[temp])
				DCr.append(dct_cr[temp])
			else:
				temp += 8
				DY.append(dct_y[temp])
				DCb.append(dct_cb[temp])
				DCr.append(dct_cr[temp])
			for y in range(x-2):
				if (x % 2 == 0):
					temp -= 7
					DY.append(dct_y[temp])
					DCb.append(dct_cb[temp])
					DCr.append(dct_cr[temp])
				else:
					temp += 7
					DY.append(dct_y[temp])
					DCb.append(dct_cb[temp])
					DCr.append(dct_cr[temp])

		return DY, DCb, DCr


	def CLD(pixel):
		width, height = im_target.size
		avg_list = []
		for x in range(width):
			for y in range(height):
				if ( (y % (height/8) == 0) and (x % (width/8) == 0)):
					avg_list.append(average(pixel, x, y, width, height))

		Y_list, Cb_list, Cr_list = [], [], []
		Y_list, Cb_list, Cr_list = rgb_ycbcr(avg_list)

		DCT_Y, DCT_Cb, DCT_Cr = [], [], []
		DCT_Y = dct(Y_list)
		DCT_Cb = dct(Cb_list)
		DCT_Cr = dct(Cr_list)

		DY, DCb, DCr = zigzag(DCT_Y, DCT_Cb, DCT_Cr)

		return DY, DCb, DCr


    # in Q2 main
	pixel = im_target.load()
	input_DY, input_DCb, input_DCr = CLD(pixel)


	dst_list_Q2 = []
	for fname in glob.glob('dataset/*.jpg'):
		#print fname
		im = Image.open(fname)
		pixel = im.load()
		output_DY, output_DCb, output_DCr = CLD(pixel)

		dst_y, dst_cb, dst_cr = .0, .0, .0
		for x in range(64):
			dst_y += .2 * ( (input_DY[x] - output_DY[x]) ** 2 )
			dst_cb += .4 * ( (input_DCb[x] - output_DCb[x]) ** 2 )
			dst_cr += .4 * ( (input_DCr[x] - output_DCr[x]) ** 2 )

		dst = sqrt(dst_y) + sqrt(dst_cb) + sqrt(dst_cr)
		dst_list_Q2.append((fname,dst))

	global dst_list_Q2_unsorted
	dst_list_Q2_unsorted = dst_list_Q2

	#print dst_list_Q2

	# sort by distance
	dst_list_Q2.sort(key=itemgetter(1))
	showDistanceAndID(dst_list_Q2)
	print "Q2. Using Color Layout------------------------"
	print_similarity(dst_list_Q2)

def Q3(qname):

	QFnumber = qname[18:21]
	QFnumber = int(QFnumber)

	d1_list = []
	d1_length_list = []
	statistics_list = []
	dst_list_Q3 = []
	fnum = 0
	for fname in glob.glob('dataset/*.jpg'):
		#print fname
		fnum += 1

		if os.path.isfile(fname + '.sift') == False :
			im = array(Image.open(fname).convert('L'))
			sift.process_image(fname, fname+'.sift')

		l1, d1 = sift.read_features_from_file(fname+'.sift')

		#print "d1's len: %s" % len(d1)
		d1_length_list.append(len(d1))

		for x in d1:
			d1_list.append(x)


	#print "d1_length_list"
	#print d1_length_list

	#print "---clusters---"
	a = KMeans(n_clusters=3).fit(d1_list)
	#print "a"
	#print len(a.labels_)
	#print a.labels_
	#print "---"


	# Statistic the number of 0, 1, 2 in each file
	counter_0, counter_1, counter_2, start, end = 0, 0, 0, 0, 0
	for f in range(fnum):
		#print "f = %d" % f
		end += d1_length_list[f]
		#print ">>>> %d ~ %d" % (start, end)
		counter_0, counter_1, counter_2 = 0, 0, 0
		for i in range(start, end):
			if a.labels_[i] == 0:
				counter_0 += 1
			elif a.labels_[i] == 1:
				counter_1 += 1
			elif a.labels_[i] == 2:
				counter_2 += 1
		start += d1_length_list[f]
		statistics_list.append((counter_0, counter_1, counter_2))
		#print "0: %d, 1:%d, 2:%d ---- total:%d" % (counter_0, counter_1, counter_2, counter_0+counter_1+counter_2)

	#print statistics_list


	# Querying
	for f in range(fnum):
		#print "f = %d" % f
		dst = np.sqrt((statistics_list[QFnumber][0]-statistics_list[f][0])**2 + (statistics_list[QFnumber][1]-statistics_list[f][1])**2 + (statistics_list[QFnumber][2]-statistics_list[f][2])**2)

		if (0 <= f and f < 10):
			fname = "dataset/ukbench0000"+ str(f)  +".jpg"
		elif (10 <= f and f < 100):
			fname = "dataset/ukbench000"+ str(f)  +".jpg"
		else:
			fname = "dataset/ukbench00"+ str(f)  +".jpg"


		#print fname
		dst_list_Q3.append((fname,dst))

	global dst_list_Q3_unsorted
	dst_list_Q3_unsorted = dst_list_Q3

	# sort by distance
	dst_list_Q3.sort(key=itemgetter(1))
	showDistanceAndID(dst_list_Q3)
	print "Q3. Using SIFT Visual Words-------------------"
	print_similarity(dst_list_Q3)

def Q4(qname):

	QFnumber = qname[17:20]
	QFnumber = int(QFnumber)

	d1_list = []
	d1_length_list = []
	statistics_list = []
	dst_list_Q4 = []
	fnum = 0
	for fname in glob.glob('dataset/*.jpg'):
		#print fname
		fnum += 1
        
		if os.path.isfile(fname + '.sift') == False :
			im = array(Image.open(fname).convert('L'))
			sift.process_image(fname, fname+'.sift')

		l1, d1 = sift.read_features_from_file(fname+'.sift')

		d1_length_list.append(len(d1))

		for x in d1:
			d1_list.append(x)
    

	#print "d1_length_list"
	#print d1_length_list

	#print "---clusters---"
	a = KMeans(n_clusters=3).fit(d1_list)
	#print "a"
	#print len(a.labels_)
	#print a.labels_
	#print "---"


	# find max
	total_0, total_1, total_2 = 0, 0, 0
	for i in range(len(a.labels_)):
		if a.labels_[i] == 0:
			total_0 += 1
		elif a.labels_[i] == 1:
			total_1 += 1
		elif a.labels_[i] == 2:
			total_2 += 1
	#print "total_0: %d, total_1: %d, total_2: %d" % (total_0, total_1, total_2)

	#print "max *= 0.9"
	ignore = 0
	if( total_0 > total_1 and total_0 > total_2):
		total_0 *= 0.9
		ignore = 0
	elif( total_1 > total_0 and total_1 > total_2):
		total_1 *= 0.9
		ignore = 1
	elif( total_2 > total_0 and total_2 > total_1):
		total_2 *= 0.9
		ignore = 2
	#print "total_0: %d, total_1: %d, total_2: %d" % (total_0, total_1, total_2)



	# Statistic the number of 0, 1, 2 in each file
	counter_0, counter_1, counter_2, start, end = 0, 0, 0, 0, 0
	for f in range(fnum):
		#print "f = %d" % f
		end += d1_length_list[f]
		#print ">>>> %d ~ %d" % (start, end)
		counter_0, counter_1, counter_2 = 0, 0, 0
		for i in range(start, end):
			if a.labels_[i] == 0:
				counter_0 += 1
			elif a.labels_[i] == 1:
				counter_1 += 1
			elif a.labels_[i] == 2:
				counter_2 += 1
		start += d1_length_list[f]
		if ignore == 0:
			counter_0 *= 0.9
		elif ignore == 1:
			counter_1 *= 0.9
		elif ignore == 2:
			counter_2 *= 0.9
		statistics_list.append((counter_0, counter_1, counter_2))
		#print "0: %d, 1:%d, 2:%d ---- total:%d" % (counter_0, counter_1, counter_2, counter_0+counter_1+counter_2)

	#print statistics_list


	# Querying
	for f in range(fnum):
		#print "f = %d" % f
		dst = np.sqrt((statistics_list[QFnumber][0]-statistics_list[f][0])**2 + (statistics_list[QFnumber][1]-statistics_list[f][1])**2 + (statistics_list[QFnumber][2]-statistics_list[f][2])**2)

		if (0 <= f and f < 10):
			fname = "dataset/ukbench0000"+ str(f)  +".jpg"
		elif (10 <= f and f < 100):
			fname = "dataset/ukbench000"+ str(f)  +".jpg"
		else:
			fname = "dataset/ukbench00"+ str(f)  +".jpg"


		#print fname
		dst_list_Q4.append((fname,dst))


	global dst_list_Q4_unsorted
	dst_list_Q4_unsorted = dst_list_Q4

	# sort by distance
	dst_list_Q4.sort(key=itemgetter(1))
	showDistanceAndID(dst_list_Q4)
	print "Q4. Using SIFT Visual Words with Stop Words---"
	print_similarity(dst_list_Q4)

def Q5(qname):
	print "qname =  %s" % qname

	Q1(qname)
	Q2(qname)
	Q4(qname)

	dst_list_Q1, dst_list_Q2, dst_list_Q4 = [], [], []

	dst_list_Q1_unsorted.sort(key=itemgetter(0))
	dst_list_Q2_unsorted.sort(key=itemgetter(0))
	dst_list_Q4_unsorted.sort(key=itemgetter(0))

	for x in range(2, len(dst_list_Q1_unsorted)):
		dst_list_Q1.append((dst_list_Q1_unsorted[x][0],float(dst_list_Q1_unsorted[x][1] / dst_list_Q1_unsorted[1][1])))
		dst_list_Q2.append((dst_list_Q2_unsorted[x][0],float(dst_list_Q2_unsorted[x][1] / dst_list_Q2_unsorted[1][1])))
		dst_list_Q4.append((dst_list_Q4_unsorted[x][0],float(dst_list_Q4_unsorted[x][1] / dst_list_Q4_unsorted[1][1])))

	print "dst_list_Q1"
	print dst_list_Q1
	print "dst_list_Q2"
	print dst_list_Q2
	print "dst_list_Q4"
	print dst_list_Q4

	dst = []
	dst_list_Q5 = []
	for x in range(len(dst_list_Q1)):
		if qname == dst_list_Q1[x][0]:
			dst_list_Q5.append((qname,0.0))
		else:
			dst = (dst_list_Q1[x][1]) + (dst_list_Q2[x][1]) + (dst_list_Q4[x][1])
			dst_list_Q5.append((dst_list_Q1[x][0],dst))
    
	print "dst_list_Q5"
	print dst_list_Q5

	# sort by distance
	dst_list_Q5.sort(key=itemgetter(1))
	showDistanceAndID(dst_list_Q5)
	print "Q5. Merge All Results-------------------------"
	print_similarity(dst_list_Q5)

def ShowImg(imgAddress):
	im = Image.open(imgAddress)
	im.show()
	
def okButtonClick():
	if question.current() == -1 and target.current() == -1:
		info.config(text='Please Select Question And Target !!!!!!!!')
	elif question.current() == -1:
		info.config(text='Please Select Question Thank You :D')
	elif target.current() == -1:
		info.config(text='Please Select Target Thank You :D')
	else:
		info.config(text='Now is Processing !!!! Please Wait:D')
		okButton.state(['disabled'])
		qtarget = target.current()
		qname = ChangeTargetToTargetAddress(qtarget)
		print "----------------------------------------------"
		print "Query: %s" % qname
		print "----------------------------------------------"
		if question.current() == 0:
			Q1(qname)
		elif question.current() == 1:
			Q2(qname)
		elif question.current() == 2:
			Q3(qname)
		elif question.current() == 3:
			Q4(qname)
		elif question.current() == 4:
			Q5(qname)
		okButton.state(['!disabled'])
		info.config(text='Results are below !')

root = Tk()
root.title("MM_HW2")

mainframe = ttk.Frame(root, padding=(3, 3, 12, 12))
mainframe.grid(column=0, row=0)
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

questionStringVar = StringVar()
targetStringVar = StringVar()

ttk.Label(mainframe, text="Question:").grid(row=0,column=0)
question = ttk.Combobox(mainframe, textvariable=questionStringVar)
question.grid(row=0,column=1)
question['values'] = questionArray

ttk.Label(mainframe, text="Target").grid(row=1,column=0)
target = ttk.Combobox(mainframe, textvariable=targetStringVar)
target.grid(row=1,column=1)
target['values'] = targetArray

okButton = ttk.Button(mainframe, text='Okay' , command=okButtonClick)
okButton.grid(row=0,column=2, rowspan=2)

info = ttk.Label(mainframe, text="Please Select Question And Target")
info.grid(row=0,column=3,columnspan=2,rowspan=2)


ttk.Label(mainframe, text="1.  Distance: ").grid(row=2,column=0)
dis1 = ttk.Label(mainframe, text="")
dis1.grid(row=2,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=2,column=2)
id1 = ttk.Label(mainframe, text="")
id1.grid(row=2,column=3)
showimg1 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id1.cget("text")))
showimg1.grid(row=2,column=4)

ttk.Label(mainframe, text="2.  Distance: ").grid(row=3,column=0)
dis2 = ttk.Label(mainframe, text="")
dis2.grid(row=3,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=3,column=2)
id2 = ttk.Label(mainframe, text="")
id2.grid(row=3,column=3)
showimg2 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id2.cget("text")))
showimg2.grid(row=3,column=4)

ttk.Label(mainframe, text="3.  Distance: ").grid(row=4,column=0)
dis3 = ttk.Label(mainframe, text="")
dis3.grid(row=4,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=4,column=2)
id3 = ttk.Label(mainframe, text="")
id3.grid(row=4,column=3)
showimg3 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id3.cget("text")))
showimg3.grid(row=4,column=4)

ttk.Label(mainframe, text="4.  Distance: ").grid(row=5,column=0)
dis4 = ttk.Label(mainframe, text="")
dis4.grid(row=5,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=5,column=2)
id4 = ttk.Label(mainframe, text="")
id4.grid(row=5,column=3)
showimg4 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id4.cget("text")))
showimg4.grid(row=5,column=4)

ttk.Label(mainframe, text="5.  Distance: ").grid(row=6,column=0)
dis5 = ttk.Label(mainframe, text="")
dis5.grid(row=6,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=6,column=2)
id5 = ttk.Label(mainframe, text="")
id5.grid(row=6,column=3)
showimg5 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id5.cget("text")))
showimg5.grid(row=6,column=4)

ttk.Label(mainframe, text="6.  Distance: ").grid(row=7,column=0)
dis6 = ttk.Label(mainframe, text="")
dis6.grid(row=7,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=7,column=2)
id6 = ttk.Label(mainframe, text="")
id6.grid(row=7,column=3)
showimg6 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id6.cget("text")))
showimg6.grid(row=7,column=4)

ttk.Label(mainframe, text="7.  Distance: ").grid(row=8,column=0)
dis7 = ttk.Label(mainframe, text="")
dis7.grid(row=8,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=8,column=2)
id7 = ttk.Label(mainframe, text="")
id7.grid(row=8,column=3)
showimg7 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id7.cget("text")))
showimg7.grid(row=8,column=4)

ttk.Label(mainframe, text="8.  Distance: ").grid(row=9,column=0)
dis8 = ttk.Label(mainframe, text="")
dis8.grid(row=9,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=9,column=2)
id8 = ttk.Label(mainframe, text="")
id8.grid(row=9,column=3)
showimg8 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id8.cget("text")))
showimg8.grid(row=9,column=4)

ttk.Label(mainframe, text="9.  Distance: ").grid(row=10,column=0)
dis9 = ttk.Label(mainframe, text="")
dis9.grid(row=10,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=10,column=2)
id9 = ttk.Label(mainframe, text="")
id9.grid(row=10,column=3)
showimg9 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id9.cget("text")))
showimg9.grid(row=10,column=4)

ttk.Label(mainframe, text="10.  Distance: ").grid(row=11,column=0)
dis10 = ttk.Label(mainframe, text="")
dis10.grid(row=11,column=1)
ttk.Label(mainframe, text="ID: ").grid(row=11,column=2)
id10 = ttk.Label(mainframe, text="")
id10.grid(row=11,column=3)
showimg10 = ttk.Button(mainframe, text='Show Image' , command= lambda: ShowImg(id10.cget("text")))
showimg10.grid(row=11,column=4)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

root.mainloop()