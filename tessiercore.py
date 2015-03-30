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
	'''
	Yield an iterator over all unique combinations of values of stepped
	quantities.
	
	Old docstring:
	combine the logical uniques of each column into boolean index over those columns
	infers that each column has
	like
	 1, 3
	 1, 4
	 1, 5
	 2, 3
	 2, 4
	 2, 5
	uniques of first column [1,2], second column [3,4,5]
	go through list and recursively combine all unique values
	'''
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
		'''Export data to .mtx file'''
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



	def __init__(self, data, n_index=None, meshgrid=False, hilbert=False, fiddle=True, uniques_col_str=[], style=[], clim=None, aspect='auto', interpolation='none'):
		'''
		Constructor. Parameters:
			data: the measured data to plot (already read from file)
			n_index: which plot to plot (for which value of stepval columns I think), None for all
			meshgrid: I have no idea
			hilbert: True if the measurement was done in Hilbert space
			fiddle: True if you want an interactive fiddle control
			uniques_col_str: names of step value columns (I think)
			style: TessierStyles to apply
			clim: color limits (TODO deprecate and replace with a style)
			aspect: I have no idea
			interpolation: I have no idea
		'''
		self.exportData =[]
		self.data = data

		print('sorting...')
		self.cols = data.columns.tolist()
		self.filterdata = data.sort(self.cols[:-1]).dropna(how='any')

		self.uniques_col = []
		self.uniques_per_col=[]

		#True is sweep neg to pos
		self.sweepdirection = data[self.cols[-1]][0] > data[self.cols[-1]][1]

		self.uniques_col_str = list(uniques_col_str)
		for i in self.uniques_col_str:
			col = getattr(filterdata, i)
			self.uniques_col.append(col)
			self.uniques_per_col.append(list(col.unique()))

		self.ccmap = loadCustomColormap()


		#fig,axs = plt.subplots(1,1,sharex=True)
		self.fig = plt.figure()
		self.fig.subplots_adjust(top=0.96, bottom=0.03, left=0.1, right=0.9,hspace=0.0)

		if n_index is None:
			self.nplots = np.product(self.uniques_per_col)
		else:
			n_index = np.array(n_index)
			self.nplots = len(n_index)

		cnt=0
		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(buildLogicals(uniques_col)):
			if n_index is None or j in n_index:
				self.create_subplot(cnt, ind, hilbert, style, clim)
				cnt += 1

		if mpl.get_backend() == 'Qt4Agg':
			from IPython.core import display
			display.display(self.fig)
			if fiddle:
				self.add_fiddle()


	def create_subplot(
			self, cnt, ind, hilbert, style, clim, aspect, interpolation):
		'''
		Create one of the subplots of this plot.
		Parameters:
			cnt: subplot index
			ind: stepval uniques index
			hilbert: True if measurement was done in Hilbert space
			style: list of TessierStyles to apply
			clim: colorbar limits (TODO turn this into a style)
			aspect: no idea
			interpolation: no idea
		'''
		slicy = self.filterdata.loc[ind]

		#get all the last columns, that we assume contains the to be plotted data
		x=slicy.iloc[:,-3]
		y=slicy.iloc[:,-2]
		z=slicy.iloc[:,-1]

		xu = np.size(x.unique())
		yu = np.size(y.unique())

		## if the measurement is not complete this will probably fail so trim of the final sweep?
		print('xu: {:d}, yu: {:d}, len(z): {:d}'.format(xu, yu, len(z)))

		if xu * yu != len(z):
			xu = (len(z) / yu) #dividing integers so should automatically floor the value
			print('xu: {:d}, yu: {:d}, len(z): {:d}'.format(xu, yu, len(z)))

		#trim the first part of the sweep, for different min max, better to trim last part?
		#or the first since there has been sorting
		#this doesnt work for e.g. a hilbert measurement

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
			self.XX = Z
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

			self.XX = np.reshape(z,(xu,yu))

		self.x = x
		self.y = y
		self.z = z
		#now set the lims
		xlims = (x.min(),x.max())
		ylims = (y.min(),y.max())

		#determine stepsize for di/dv, inprincipe only y step is used (ie. the diff is also taken in this direction and the measurement swept..)
		xstep = (xlims[0] - xlims[1])/xu
		ystep = (ylims[0] - ylims[1])/yu

#             if meshgrid:
#                 X, Y = np.meshgrid(xi, yi)
#                 scipy.interpolate.griddata((xs, ys), Z, X, Y)
#                 Z = griddata(x,y,Z,xi,yi)

		self.exportData.append(self.XX)
		try:
			m={
				'xu': xu,
				'yu': yu,
				'xlims': xlims,
				'ylims': ylims,
				'zlims': (0,0),
				'xname': self.cols[-3],
				'yname': self.cols[-2],
				'zname': 'unused',
				'datasetname': data.name}
			self.exportDataMeta = np.append(self.exportDataMeta, m)
		except:
			pass
		print('plotting...')

		ax = plt.subplot(nplots, 1, cnt + 1)
		cbar_title = ''

		if type(style) != list:
			style = list([style])

		# smooth the datayesplz
		#import scipy.ndimage as ndimage
		#XX = ndimage.gaussian_filter(XX,sigma=1.0,order=0)
		# Maybe make a TessierStyle for this

		measAxisDesignation = parseUnitAndNameFromColumnName(self.data.keys()[-1])
		# wrap all needed arguments in a datastructure
		cbar_quantity = measAxisDesignation[0]
		cbar_unit = measAxisDesignation[1]
		cbar_trans = [] # trascendental tracer :P For keeping track of logs and stuff

		w = tstyle.TessierWrap(
				ext=xlims+ylims, ystep=ystep, XX=self.XX,
				cbar_quantity=cbar_quantity, cbar_unit=cbar_unit,
				cbar_trans=cbar_trans, flipaxes=False, has_title=True,
				aspect=aspect, interpolation=interpolation,
				cmap=plt.get_cmap(self.ccmap), cols = self.cols)
		for st in style:
			st.apply_to_wrap(w)

		#unwrap
		cbar_trans_formatted = ''.join([''.join(s+'(') for s in w.cbar_trans])
		cbar_title = cbar_trans_formatted + w.cbar_quantity + ' (' + w.cbar_unit + ')'
		if len(w.cbar_trans) is not 0:
			cbar_title += ')'

		#postrotate np.rot90
		XX = np.rot90(w.XX)

		self.im = ax.imshow(
				XX, extent=w.ext, cmap=plt.get_cmap(self.ccmap), aspect=aspect,
				interpolation=interpolation, norm=w.imshow_norm, clim=clim)

		ax.set_xlabel(self.cols[-3])
		ax.set_ylabel(self.cols[-2])
		
		for st in style:
			st.apply_to_axis(w, ax, self)

		title = ''
		for i in self.uniques_col_str:
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

	def add_fiddle(self):
		'''Add a fiddle handler and fiddle-enabling button'''
		self.fiddle = Fiddle(self.fig)
		axFiddle = plt.axes([0.1, 0.85, 0.15, 0.075])

		self.bnext = Button(axFiddle, 'Fiddle')
		self.bnext.on_clicked(self.fiddle.connect)

		#attach to the relevant figure to make sure the object does not go out of scope
		self.fig.fiddle = self.fiddle
		self.fig.bnext = self.bnext

