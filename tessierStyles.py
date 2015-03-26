import numpy as np
from scipy import signal
import matplotlib.colors as mplc

USIEMENS_PER_CONDUCTANCE_QUANTUM = 0.0129064037

class TessierStyle:
	'''All TessierPlot styles (plugins) should extend this class'''
	def __init__(self):
		pass

	def execute(self, w):
		raise NotImplementedError()

def nonzeromin(x):
	'''
	Get the smallest non-zero value from an array
	Also works recursively for multi-dimensional arrays
	Returns None if no non-zero values are found
	'''
	x = np.array(x)
	nzm = None
	if len(x.shape) > 1:
		for i in range(x.shape[0]):
			xnow = nonzeromin(x[i])
			if xnow != 0 and xnow is not None and (nzm is None or nzm > xnow):
				nzm = xnow
	else:
		for i in range(x.shape[0]):
			if x[i] != 0 and (nzm is None or nzm > x[i]):
				nzm = x[i]
	return nzm

def moving_average_2d(data, window):
	"""Moving average on two-dimensional data."""
	# Makes sure that the window function is normalized.
	window /= window.sum()
	# Makes sure data array is a numpy array or masked array.
	if type(data).__name__ not in ['ndarray', 'MaskedArray']:
		data = np.asarray(data)

	# The output array has the same dimensions as the input data
	# (mode='same') and symmetrical boundary conditions are assumed
	# (boundary='symm').
	return signal.convolve2d(data, window, mode='same', boundary='symm')

class Deinterlace(TessierStyle):
	'''Tessier style for deinterlacing a measurement'''
	def execute(self, w):
		w['deinterXXodd'] = w['XX'][1::2,1:]
		w['deinterXXeven'] = w['XX'][::2,:]

class SmoothingFilter(TessierStyle):
	'''Tessier style for smoothing with a custom linear filter'''
	def __init__(self, win):
		self.win = win

	def execute(self, w):
		w['XX'] = moving_average_2d(w['XX'], self.win)

class MovAvg(SmoothingFilter):
	'''Tessier style for moving average smoothing'''
	def __init__(self, m=1, n=5):
		SmoothingFilter.__init__(self, np.ones((m, n)))

class SavGol(TessierStyle):
	'''Tessier style for Savitzky-Golay smoothing'''
	def __init__(self, samples=3, order=1):
		(self.samples, self.order) = (int(samples), int(order))

	def execute(self, w):
		w['XX'] = signal.savgol_filter(w['XX'], self.samples, self.order)

class DIDV(TessierStyle):
	'''Tessier style for taking a derivative'''
	def __init__(self, axis=1, scale=1.,
			quantity='dI/dV', unit='$\mu$Siemens'):
		(self.axis, self.scale, self.quantity, self.unit) = (
				axis, float(scale), quantity, unit)

	def execute(self, w):
		w['XX'] = np.diff(w['XX'], axis=self.axis) * self.scale / w['ystep']
		(w['cbar_quantity'], w['cbar_unit']) = (self.quantity, self.unit)

class DIDV_Conductancequantum(DIDV):
	'''DIDV style converting to conductance quantum'''
	def __init__(self, axis=1, quantity='dI/dV'):
		DIDV.__init__(self, axis, USIEMENS_PER_CONDUCTANCE_QUANTUM,
				quantity, '$\mathrm{e}^2/\mathrm{h}$')

class Logscale(TessierStyle):
	'''
	Tessier style for taking the base-10 logarithm of the absolute value
	of each value
	'''
	def execute(self, w):
		w['XX'] = np.log10(np.abs(w['XX']))
		w['cbar_trans'] = ['log$_{10}$','abs'] + w['cbar_trans']

class FancyLogscale(TessierStyle):
	'''
	Tessier style for taking the absolute value of each value and
	using a logarithmic normalising function, resulting in a fancy
	logarithmic colorbar.
	This style might be incompatible with Fiddle.
	'''
	def __init__(self, cmin=None, cmax=None):
		if type(cmin) is str:
			cmin = float(cmin)
		if type(cmax) is str:
			cmax = float(cmax)
		(self.cmin, self.cmax) = (cmin, cmax)

	def execute(self, w):
		w['XX'] = abs(w['XX'])
		if self.cmin is None:
			self.cmin = w['XX'].min()
		if self.cmin == 0:
			self.cmin = 0.1 * nonzeromin(w['XX'])
		if self.cmax is None:
			self.cmax = w['XX'].max()
		w['imshow_norm'] = mplc.LogNorm(vmin=self.cmin, vmax=self.cmax)

class Abs(TessierStyle):
	'''Tessier style for taking the absolute value of each value'''
	def execute(self, w):
		w['XX'] = np.abs(w['XX'])
		w['cbar_trans'] = ['abs'] + w['cbar_trans']

class Flipaxes(TessierStyle):
	'''Tessier style for flipping the X and Y axes'''
	def execute(self, w):
		w['XX'] = np.transpose( w['XX'])
		w['ext'] = (w['ext'][2],w['ext'][3],w['ext'][0],w['ext'][1])
		w['flipaxes'] = True

class NoTitle(TessierStyle):
	'''Style to not display a title'''
	def execute(self, w):
		w['has_title'] = False


def getEmptyWrap():
	'''Get empty wrap with default parameter values'''
	w = {'ext':(0,0,0,0), 'ystep':1, 'XX': [], 'cbar_quantity': '', 'cbar_unit': 'a.u.', 'cbar_trans': [], 'imshow_norm': None, 'flipaxes': False, 'has_title': True}
	return w

