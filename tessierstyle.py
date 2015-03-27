class TessierStyle:
	'''All TessierPlot styles (plugins) should extend this class'''
	def __init__(self):
		pass

	def exe(self, w):
		raise NotImplementedError()

	def execute(self, w):
		if isinstance(w, TessierWrap):
			self.exe(w)
		else:
			raise TypeError(
					'Wrap {:} is no TessierWrap but a {:}'.format(w, type(w)))


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

