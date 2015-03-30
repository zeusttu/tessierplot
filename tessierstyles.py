import numpy as np
from scipy import signal
import matplotlib.colors as mplc

import tessierstyle


class Deinterlace(tessierstyle.TessierStyle):
	'''Tessier style for deinterlacing a measurement'''
	def wrapaction(self, w):
		self.XXodd = np.rot90(w.XX[1::2,1:])
		self.XXeven = np.rot90(w.XX[::2,:])

	def axisaction(self, w, ax, tessierobj):
		(self.fig, (self.ax_odd, self.ax_even)) = plt.subplots(2, 1)
		self.im_odd = self.ax_odd.imshow(
				self.XXodd, extent=w.ext, cmap=w.cmap,
				aspect=w.aspect, interpolation=w.interpolation)
		self.im_even = self.ax_even.imshow(
				self.XXeven, extent=w.ext, cmap=w.cmap,
				aspect=w.aspect, interpolation=w.interpolation)
		if tessierobj is not None:
			(tessierobj.fig, tessierobj.deinter) = (self.fig, self)


class SmoothingFilter(tessierstyle.TessierStyle):
	'''Tessier style for smoothing with a custom linear filter'''
	def __init__(self, win, normalise=True, mode='same', boundary='symm'):
		if normalise:
			win /= np.sum(win)
		self.win = win
		self.mode = mode
		self.boundary = boundary

	def wrapaction(self, w):
		XX = w.XX
		if type(XX) not in (np.ndarray, np.ma.core.MaskedArray):
			XX = np.asarray(XX)
		w.XX = signal.convolve2d(
				XX, self.win, mode=self.mode, boundary=self.boundary)


class MovAvg(SmoothingFilter):
	'''Tessier style for moving average smoothing'''
	def __init__(self, m=1, n=5):
		SmoothingFilter.__init__(self, np.ones((m, n)), normalise=True)


class SavGol(tessierstyle.TessierStyle):
	'''Tessier style for Savitzky-Golay smoothing'''
	def __init__(self, samples=3, order=1):
		(self.samples, self.order) = (int(samples), int(order))

	def wrapaction(self, w):
		w.XX = signal.savgol_filter(w.XX, self.samples, self.order)


class DIDV(tessierstyle.TessierStyle):
	'''Tessier style for taking a derivative'''
	def __init__(self, axis=1, scale=1.,
			quantity='dI/dV', unit='$\mu$Siemens'):
		(self.axis, self.scale, self.quantity, self.unit) = (
				axis, float(scale), quantity, unit)

	def wrapaction(self, w):
		w.XX = np.diff(w.XX, axis=self.axis) * self.scale / w.ystep
		(w.cbar_quantity, w.cbar_unit) = (self.quantity, self.unit)


class DIDV_Conductancequantum(DIDV):
	'''DIDV style converting to conductance quantum'''
	USIEMENS_PER_CONDUCTANCE_QUANTUM = 0.0129064037

	def __init__(self, axis=1, quantity='dI/dV'):
		DIDV.__init__(self, axis, self._USIEMENS_PER_CONDUCTANCE_QUANTUM,
				quantity, '$\mathrm{e}^2/\mathrm{h}$')


class Logscale(tessierstyle.TessierStyle):
	'''
	Tessier style for taking the base-10 logarithm of the absolute value
	of each value
	'''
	def wrapaction(self, w):
		w.XX = np.log10(np.abs(w.XX))
		w.cbar_trans = ['log$_{10}$','abs'] + w.cbar_trans


class FancyLogscale(tessierstyle.TessierStyle):
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
				xnow = FancyLogscale.nonzeromin(x[i])
				if (xnow != 0 and xnow is not None and
						(nzm is None or nzm > xnow)):
					nzm = xnow
		else:
			for i in range(x.shape[0]):
				if x[i] != 0 and (nzm is None or nzm > x[i]):
					nzm = x[i]
		return nzm

	def wrapaction(self, w):
		w.XX = abs(w.XX)
		if self.cmin is None:
			self.cmin = w.XX.min()
		if self.cmin == 0:
			self.cmin = 0.1 * FancyLogscale.nonzeromin(w.XX)
		if self.cmax is None:
			self.cmax = w.XX.max()
		w.imshow_norm = mplc.LogNorm(vmin=self.cmin, vmax=self.cmax)


class Abs(tessierstyle.TessierStyle):
	'''Tessier style for taking the absolute value of each value'''
	def wrapaction(self, w):
		w.XX = np.abs(w.XX)
		w.cbar_trans = ['abs'] + w.cbar_trans


class FlipAxes(tessierstyle.TessierStyle):
	'''Tessier style for flipping the X and Y axes'''
	def wrapaction(self, w):
		w.XX = np.transpose( w.XX)
		w.ext = (w.ext[2],w.ext[3],w.ext[0],w.ext[1])
		w.flipaxes = True

	def axisaction(self, w, ax, tessierobj):
		(xlbl, ylbl) = (ax.get_ylabel(), ax.get_xlabel())
		ax.set_xlabel(xlbl)
		ax.set_ylabel(ylbl)


class NoTitle(tessierstyle.TessierStyle):
	'''Style to not display a title'''
	def wrapaction(self, w):
		w.has_title = False

