from matplotlib.axes import Axes

class TessierStyle:
	'''All TessierPlot styles (plugins) should extend this class'''
	def __init__(self):
		'''Constructor. Empty by default. You may want to override this.'''
		pass

	def wrapaction(self, w):
		'''
		Action to apply in wrap manipulation phase (before plot exists).
		Empty by default. You may want to override this.
		'''
		pass

	def axisaction(self, w, ax, tessierobj):
		'''
		Action to apply in axis manipulation phase (when plot exists).
		Empty by default. You may want to override this.
		'''
		pass

	def apply_to_wrap(self, w):
		'''
		Apply the action defined in wrapaction() after some type checks.
		Useful for preprocessing the data before actually plotting it.
		Overriding this function is not recommended,
		override wrapaction() instead.
		'''
		if isinstance(w, TessierWrap):
			self.wrapaction(w)
		else:
			raise TypeError(
					'Wrap {:} is no TessierWrap but a {:}'.format(w, type(w)))

	def apply_to_axis(self, w, ax, tessierobj=None):
		'''
		Apply the action defined in axisaction() after some type checks.
		Useful for post-processing the plot.
		Overriding this function is not recommended,
		override axisaction() instead.
		'''
		if isinstance(w, TessierWrap) and isinstance(ax, Axes):
			self.axisaction(w, ax)
		else:
			raise TypeError(
					'{:} is no TessierWrap but a {:} '
					'or {:} is no matplotlib.axes.Axes but a {:}'
					.format(w, type(w), ax, type(ax)))


class TessierWrap:
	'''Get wrap with (default) parameter values'''
	def __init__(self, **kwargs):
		# Default values
		self.__dict__.update({
				'ext':(0,0,0,0), 'ystep':1, 'XX': [], 'cbar_quantity': '',
				'cbar_unit': 'a.u.', 'cbar_trans': [], 'imshow_norm': None,
				'flipaxes': False, 'has_title': True, 'deinterlace': False})
		# Values passed to constructor (optional)
		self.__dict__.update(kwargs)

