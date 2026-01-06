
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
from time import sleep

class BlittedCursor(object):
    def __init__(self, axes, sharex=True):
        self.axes = axes
        self.backgrounds = []
        self.cursors = []

        ax0 = self.axes[0]
        self.playing = False
        self.x = 0

        for i, ax in enumerate(self.axes):
            if i > 0 and sharex:
                ax.sharex(ax0)

            lines = ax.get_lines()
            if lines:
                min_x = min(min(line.get_xdata()) for line in lines)
                self.cursors.append(ax.axvline(x=min_x,
                                               color='black',
                                               linestyle='--',
                                               lw=0.8,
                                               visible=False))
            else:
                self.cursors.append(None)

        # get unique canvasses by iterating over axes
        # this is necessary to avoid multiple connections to the same canvas
        self.canvasses = [self.axes[0].figure.canvas]
        for ax in self.axes[1:]:
            if ax.figure.canvas is not self.canvasses:
                self.canvasses.append(ax.figure.canvas)

        for canvas in self.canvasses:
            # connect the canvas to the draw and motion events
            canvas.mpl_connect('draw_event', self._on_draw)
            canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            canvas.mpl_connect('key_press_event', self._on_key_press)

        self.timer = self.axes[0].figure.canvas.new_timer(interval=50)
        self.timer.add_callback(self._on_timer)

    def _on_draw(self, event):
        self.backgrounds.clear()
        for ax in self.axes:
            canvas = ax.figure.canvas
            self.backgrounds.append(canvas.copy_from_bbox(ax.bbox))

    def _on_mouse_move(self, event):
        if event.xdata is None or not self.backgrounds:
            return

        self.x = event.xdata

        for ax, line, bg in zip(self.axes, self.cursors, self.backgrounds):
            canvas = ax.figure.canvas
            canvas.restore_region(bg)
            if line is not None:
                line.set_xdata([event.xdata])
                line.set_visible(True)
                ax.draw_artist(line)
            canvas.blit(ax.bbox)

    def _on_timer(self):

        # generate mouse event to canvas, so that all children get updated
        from matplotlib.backend_bases import MouseEvent

        self.x += 0.02
        canvas = self.axes[0].figure.canvas
        event = MouseEvent(
            name="motion_notify_event",
            canvas=canvas,
            x=self.x,      # pixel coordinates
            y=0,
            button=None,
            key=None,
            step=0,
            dblclick=False,
            guiEvent=None
        )
        event.xdata = self.x  # data coordinates
        canvas.callbacks.process(
            "motion_notify_event", event
        )


    def _on_key_press(self, event):
        """Callback function to update the viewport on key press

        Args:
            event (matplotlib.backend_bases.KeyEvent): key event

        Returns:
            None
        """
        if event.key == ' ':
            if self.playing:
                self.playing = False
                self.timer.stop()
            else:
                self.playing = True
                self.timer.start()
