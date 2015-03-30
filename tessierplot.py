#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

# data = loadFile(...)
# a = Plot3DSlices(data,)

import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('Qt4Agg')


import pandas as pd
import numpy as np

import re

import tessierstyles as tstyles
import tessierstyle as tstyle
from tessiercore import Plot3DSlices, \
		buildLogicals, loadCustomColormap, parseUnitAndNameFromColumnName

# Reload style modules, useful for debugging
from imp import reload
reload(tstyle)
reload(tstyles)
reload(tstyles.tessierstyle)


def parseheader(file):
	names = []
	skipindex = 0
	with open(file) as f:
		colname = re.compile(r'^#.*?name\:{1}(.*?)\r?$')

		for i, line in enumerate(f):
			if i < 3:
				continue
			#print(line)
			a = colname.findall(line)
			#print(a)
			if len(a) >= 1:
				names.append(a[0])
			if i > 5:
				if line[0] != '#': #find the skiprows accounting for the first linebreak in the header
					skipindex = i
					break
			if i > 300:
				break
	#print(names)
	return names, skipindex

def quickplot(file,**kwargs):
	names,skipindex = parseheader(file)
	data = loadFile(file,names=names,skiprows=skipindex)
	p = Plot3DSlices(data,**kwargs)
	return p

def scanplot(file, fig=None, n_index=None, style=[], data=None, **kwargs):
	# kwargs go into matplotlib/pyplot plot command
	if not fig:
		fig = plt.figure()
	names,skip = parseheader(file)

	if data is None:
		print('loading')
		data = loadFile(file,names=names,skiprows=skip)

	uniques_col = []

	# scanplot assumes 2d plots with data in the two last columns
	uniques_col_str = names[:-2]

	reg = re.compile(r'\{(.*?)\}')
	parsedcols = []
	# do some filtering of the colstr to get seperate name and unit of said name
	for a in uniques_col_str:
		z = reg.findall(a)
		if len(z) > 0:
			parsedcols.append(z[0])
		else:
			parsedcols.append('')
		# name is first


	for i in uniques_col_str:
		col = getattr(data,i)
		uniques_col.append(col)

	if n_index != None:
		n_index = np.array(n_index)
		nplots = len(n_index)

	for i,j in enumerate(buildLogicals(uniques_col)):
		if n_index != None:
				if i not in n_index:
					continue
		filtereddata = data.loc[j]
		title =''
		for i,z in enumerate(uniques_col_str):
			title = '\n'.join([title, '{:s}: {:g} (mV)'.format(parsedcols[i],getattr(filtereddata,z).iloc[0])])

		measAxisDesignation = parseUnitAndNameFromColumnName(filtereddata.keys()[-1])

		wrap = tstyle.TessierWrap()
		# put in the last column, the 'measured' value so to say
		wrap.XX = filtereddata.iloc[:,-1]
		for st in style:
			st.apply_to_wrap(wrap)

		p = plt.plot(filtereddata.iloc[:,-2],wrap.XX,label=title,**kwargs)

	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=2, mode="expand", borderaxespad=0.)
	#from IPython.core import display
	#display.display(fig)
	return fig,data

def loadFile(file, names=['L','B1','B2','vsd','zz'], skiprows=25):
	#print('loading...')
	data = pd.read_csv(file, sep='\t', comment='#',skiprows=skiprows,names=names)
	data.name = file
	return data

def demoCustomColormap():
	ccmap = loadCustomColormap()
	a = np.linspace(0, 1, 256).reshape(1,-1)
	a = np.vstack((a,a))
	fig = plt.figure()
	plt.imshow(a, aspect='auto', cmap=plt.get_cmap(ccmap), origin='lower')
	plt.show()
	 
