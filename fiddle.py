import matplotlib.pyplot as plt

class Fiddle:

	def __init__(self,fig):
		self.fig = fig
		self.press = None
		self.Fiddle = False
		self.fiddleOn = False

	def turnon(self, event, current_ax):
		self.Fiddle = True
		toggle_selector.RS = RectangleSelector(current_ax, self.line_select_callback,
									   drawtype='box', useblit=True,
									   button=[3], # don't use middle button
									   minspanx=5, minspany=5,
									   spancoords='pixels')
		plt.connect('key_press_event', toggle_selector)

	def line_select_callback(eclick, erelease):
		'eclick and erelease are the press and release events'
		x1, y1 = eclick.xdata, eclick.ydata
		x2, y2 = erelease.xdata, erelease.ydata
		print("({:3.2f}, {:3.2f}) --> ({:3.2f}, {:3.2f})".format(x1, y1, x2, y2))
		print(" The button you used were: {:s} {:s}".format(eclick.button, erelease.button))

	def connect(self,event):
		'(dis-)connect to all the events we need'
		if(not self.fiddleOn):
			self.cidpress = self.fig.canvas.mpl_connect(
					'button_press_event', self.on_press)
			self.cidrelease = self.fig.canvas.mpl_connect(
					'button_release_event', self.on_release)
			self.cidmotion = self.fig.canvas.mpl_connect(
					'motion_notify_event', self.on_motion)
			self.fiddleOn = True
		else:
			self.fig.canvas.mpl_disconnect(self.cidpress)
			self.fig.canvas.mpl_disconnect(self.cidrelease)
			self.fig.canvas.mpl_disconnect(self.cidmotion)
			self.fiddleOn = False

	def disconnect(self,event):
		'disconnect all the events we needed'
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)
		self.fiddleOn = False

	def on_press(self, event):
		'on button press we will see if the mouse is over us and store some data'

		contains, attrd = self.fig.contains(event)
		if not contains: return
		if(event.xdata==None): return
		self.press = event.xdata, event.ydata
		a = event.inaxes.images
		for i in a:
			self.clim = i.get_clim()

	def on_motion(self, event):
		'on motion we will move the rect if the mouse is over us'
		if self.press is None: return
		if(event.xdata==None): return
		#if math.isnan(float(event.xdata)): return
		xpress, ypress = self.press
		dx = event.xdata - xpress
		dy = event.ydata - ypress
		#print('xp={:f}, ypress={:f}, event.xdata={:f},event.ydata={:f}, dx={:f}, dy={:f}'.format(xpress,ypress,event.xdata, event.ydata,dx, dy))
		#self.rect.set_x(x0+dx)
		#self.rect.set_y(y0+dy)


		#dx controls limits width
		#dy controls limits center
		a = event.inaxes.images

		for i in a:
			xmin,xmax,ymin,ymax = i.get_extent()

			vmin,vmax=self.clim
			xrel=abs(xmin - xmax)
			yrel=abs(ymin - ymax)

			newvmin = dy/yrel*(vmin-vmax) + vmin + dx/xrel*(vmin-vmax)
			newvmax = dy/yrel*(vmin-vmax) + vmax - dx/xrel*(vmin-vmax)

			if (newvmax > newvmin):
				i.set_clim(vmax=newvmax, vmin=newvmin)




		self.fig.canvas.draw()


	def on_release(self, event):
		'on release we reset the press data'
		self.press = None

