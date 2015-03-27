import numpy as np
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tessierstyle as tstyle
from fiddle import Fiddle
from matplotlib.widgets import Button

def parseUnitAndNameFromColumnName(input):
	reg = re.compile(r'\{(.*?)\}')
	z = reg.findall(input)
	return z


def loadCustomColormap(file='./cube1.xls'):
	xl = pd.ExcelFile(file)

	dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
	for i in dfs.keys():
		r = dfs[i]
		ls = [r.iloc[:,0],r.iloc[:,1],r.iloc[:,2]]
		do = list(zip(*ls))

	ccmap=mpl.colors.LinearSegmentedColormap.from_list('name',do)
	return ccmap


def buildLogicals(xs):
#combine the logical uniques of each column into boolean index over those columns
#infers that each column has
#like
# 1, 3
# 1, 4
# 1, 5
# 2, 3
# 2, 4
# 2, 5
#uniques of first column [1,2], second column [3,4,5]
#go through list and recursively combine all unique values
	if len(xs) > 1:
		for i in xs[0].unique():
			if np.isnan(i):
				continue
			for j in buildLogicals(xs[1:]):
				yield (xs[0] == i) & j ## boolean and
	elif len(xs) == 1:
		for i in xs[0].unique():
			if (np.isnan(i)):
				#print('NaN found, skipping')
				continue
			yield xs[0] == i
	else:
		#empty list
		yield slice(None) #return a 'semicolon' to select all the values when there's no value to filter on


class Plot3DSlices:
	fig = None
	data = []
	uniques_col_str=None
	exportData = []
	exportDataMeta =[]
	def show(self):
		plt.show()

	def exportToMtx(self):

		for j, i in enumerate(self.exportData):

			data = i
			print(j)
			m = self.exportDataMeta[j]

			sz = np.shape(data)
			#write
			try:
				fid = open('{:s}{:d}{:s}'.format(self.data.name, j, '.mtx'),'w+')
			except Exception as e:
				print('Couldnt create file: {:s}'.format(str(e)))
				return

			#example of first two lines
			#Units, Data Value at Z = 0.5 ,X, 0.000000e+000, 1.200000e+003,Y, 0.000000e+000, 7.000000e+002,Nothing, 0, 1
			#850 400 1 8
			str1 = 'Units, Name: {:s}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}\n'.format(
				m['datasetname'],
				m['xname'],
				m['xlims'][0],
				m['xlims'][1],
				m['yname'],
				m['ylims'][0],
				m['ylims'][1],
				m['zname'],
				m['zlims'][0],
				m['zlims'][1]
				)
			floatsize = 8
			str2 = '{:d} {:d} {:d} {:d}\n'.format(m['xu'],m['yu'],1,floatsize)
			fid.write(str1)
			fid.write(str2)
			#reshaped = np.reshape(data,sz[0]*sz[1],1)
			data.tofile(fid)
			fid.close()



	def __init__(self, data, n_index=None, meshgrid=False, hilbert=False, didv=False, fiddle=True, uniques_col_str=[], style=[], clim=(0,0), aspect='auto', interpolation='none'):
		#uniques_col_str, array of names of the columns that are e.g. the slices of the
		#style, 'normal,didv,didv2,log'
		#clim, limits of the colorplot c axis

		self.exportData =[]
		self.data = data
		self.uniques_col_str=uniques_col_str

		#n_index determines which plot to plot,
		# 0 value for plotting all
		
		

		print('sorting...')
		cols = data.columns.tolist()
		filterdata = data.sort(cols[:-1])
		filterdata = filterdata.dropna(how='any')

		uniques_col = []
		self.uniques_per_col=[]
		
		
		sweepdirection = data[cols[-1]][0] > data[cols[-1]][1] #True is sweep neg to pos


		uniques_col_str = list(uniques_col_str)
		for i in uniques_col_str:
			col = getattr(filterdata,i)
			uniques_col.append(col)
			self.uniques_per_col.append(list(col.unique()))

		self.ccmap = loadCustomColormap()


		#fig,axs = plt.subplots(1,1,sharex=True)
		self.fig = plt.figure()
		self.fig.subplots_adjust(top=0.96, bottom=0.03, left=0.1, right=0.9,hspace=0.0)


		nplots = 1
		for i in self.uniques_per_col:
			nplots *= len(i)

		if n_index != None:
			n_index = np.array(n_index)
			nplots = len(n_index)
			

		cnt=0
		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(buildLogicals(uniques_col)):
			if n_index != None:
				if j not in n_index:
					continue

			slicy = filterdata.loc[ind]
			#get all the last columns, that we assume contains the to be plotted data
			x=slicy.iloc[:,-3]
			y=slicy.iloc[:,-2]
			z=slicy.iloc[:,-1]

			xu = np.size(x.unique())
			yu = np.size(y.unique())


			## if the measurement is not complete this will probably fail so trim of the final sweep?
			print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

			if xu*yu != len(z):
				xu = (len(z) / yu) #dividing integers so should automatically floor the value

			#trim the first part of the sweep, for different min max, better to trim last part?
			#or the first since there has been sorting
			#this doesnt work for e.g. a hilbert measurement

			print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))
			if hilbert:

				Z = np.zeros((xu,yu))

				#make a meshgrid for indexing
				xs = np.linspace(x.min(),x.max(),xu)
				ys = np.linspace(y.min(),y.max(),yu)
				xv,yv = np.meshgrid(xs,ys,sparse=True)
				#evaluate all datapoints
				for i,k in enumerate(xs):
					print(i,k)
					for j,l in enumerate(ys):
						ind = (k == x) & (l == y)
						#print(z[ind.index[0]])
						Z[i,j] = z[ind.index[0]]
				#keep a z array, index with datapoints from meshgrid+eval
				XX = Z
			else:
				#sorting sorts negative to positive, so beware:
				#sweep direction determines which part of array should be cut off
				if sweepdirection:
					z = z[-xu*yu:]
					x = x[-xu*yu:]
					y = y[-xu*yu:]
				else:
					z = z[:xu*yu]
					x = x[:xu*yu]
					y = y[:xu*yu]

				XX = np.reshape(z,(xu,yu))

			self.x = x
			self.y = y
			self.z = z
			#now set the lims
			xlims = (x.min(),x.max())
			ylims = (y.min(),y.max())

			#determine stepsize for di/dv, inprincipe only y step is used (ie. the diff is also taken in this direction and the measurement swept..)
			xstep = (xlims[0] - xlims[1])/xu
			ystep = (ylims[0] - ylims[1])/yu
			ext = xlims+ylims


#             if meshgrid:
#                 X, Y = np.meshgrid(xi, yi)
#                 scipy.interpolate.griddata((xs, ys), Z, X, Y)
#                 Z = griddata(x,y,Z,xi,yi)

			self.XX = XX

			self.exportData.append(XX)
			try:
				m={
					'xu':xu,
					'yu':yu,
					'xlims':xlims,
					'ylims':ylims,
					'zlims':(0,0),
					'xname':cols[-3],
					'yname':cols[-2],
					'zname':'unused',
					'datasetname':data.name}
				self.exportDataMeta = np.append(self.exportDataMeta,m)
			except:
				pass
			print('plotting...')

			ax = plt.subplot(nplots, 1, cnt+1)
			cbar_title = ''

			if type(style) != list:
				style = list([style])

			#smooth the datayesplz
			#import scipy.ndimage as ndimage
			#XX = ndimage.gaussian_filter(XX,sigma=1.0,order=0)


			measAxisDesignation = parseUnitAndNameFromColumnName(self.data.keys()[-1])
			#wrap all needed arguments in a datastructure
			cbar_quantity = measAxisDesignation[0]
			cbar_unit = measAxisDesignation[1]
			cbar_trans = [] #trascendental tracer :P For keeping track of logs and stuff

			w = tstyle.TessierWrap(
					ext=ext, ystep=ystep, XX=XX, cbar_quantity=cbar_quantity,
					cbar_unit=cbar_unit, cbar_trans=cbar_trans,
					flipaxes=False, has_title=True)
			for st in style:
				st.execute(w)

			#unwrap
			ext = w.ext
			XX = w.XX
			cbar_trans_formatted = ''.join([''.join(s+'(') for s in w.cbar_trans])
			cbar_title = cbar_trans_formatted + w.cbar_quantity + ' (' + w.cbar_unit + ')'
			if len(w.cbar_trans) is not 0:
				cbar_title = cbar_title + ')'

			#postrotate np.rot90
			XX = np.rot90(XX)

			if 'deinterXXodd' in w: # If deinterlace style is used
				self.fig = plt.figure()
				ax_deinter_odd  = plt.subplot(2, 1, 1)
				w.deinterXXodd = np.rot90(w.deinterXXodd)
				ax_deinter_odd.imshow(w.deinterXXodd,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)

				ax_deinter_even = plt.subplot(2, 1, 2)
				w.deinterXXeven = np.rot90(w.deinterXXeven)
				ax_deinter_even.imshow(w.deinterXXeven,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)

			self.im = ax.imshow(XX,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation, norm=w.imshow_norm)
			if clim != (0,0):
			   self.im.set_clim(clim)

			if w.flipaxes:
				ax.set_xlabel(cols[-2])
				ax.set_ylabel(cols[-3])
			else:
				ax.set_xlabel(cols[-3])
				ax.set_ylabel(cols[-2])


			title = ''
			for i in uniques_col_str:
				title = '\n'.join([title, '{:s}: {:g} (mV)'.format(i,getattr(slicy,i).iloc[0])])
			print(title)
			if w.has_title:
				ax.set_title(title)
			# create an axes on the right side of ax. The width of cax will be 5%
			# of ax and the padding between cax and ax will be fixed at 0.05 inch.
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)

			pos = list(ax.get_position().bounds)

			self.cbar = plt.colorbar(self.im, cax=cax)
			cbar = self.cbar

			cbar.set_label(cbar_title)

			cnt+=1 #counter for subplots


		if mpl.get_backend() == 'Qt4Agg':
			from IPython.core import display
			display.display(self.fig)

		if fiddle and (mpl.get_backend() == 'Qt4Agg'):
			self.fiddle = Fiddle(self.fig)
			axFiddle = plt.axes([0.1, 0.85, 0.15, 0.075])


			self.bnext = Button(axFiddle, 'Fiddle')
			self.bnext.on_clicked(self.fiddle.connect)

			#attach to the relevant figure to make sure the object does not go out of scope
			self.fig.fiddle = self.fiddle
			self.fig.bnext = self.bnext
		#plt.show()

