"""
A collection of utilities to assist in designing terminal applications
with the assistance of ANSI escape sequences.
"""
import io
import sys


class Style:
    RESET          = 0
    # intensity
    BOLD           = 1
    FAINT          = 2
    NO_INTENSITY   = 22
    # italics
    ITALIC         = 3
    NO_ITALIC      = 23
    # underline
    UNDERLINE      = 4
    NO_UNDERLINE   = 24
    # blikning
    SLOW_BLINK     = 5
    RAPID_BLINK    = 6
    NO_BLINK       = 25
    # inversion
    INVERT         = 7
    NO_INVERT      = 27


class Color:
    BLACK   = (30, 40)
    RED     = (31, 41)
    GREEN   = (32, 42)
    YELLOW  = (33, 43)
    BLUE    = (34, 44)
    MAGENTA = (35, 45)
    CYAN    = (36, 46)
    WHITE   = (37, 47)


class InputDevice:

    def __init__(self, fin, name=None):
        # STUB
        pass


class OutputDevice:

    def __init__(self, stream, esc_char=None, name=None):
        self.stream = stream
        self.name = name

    # ----- Non-Control Sequence interface -----
    def print(self, data, persistent_style=False):
        control_start = '&'
        control_end = ';'

        strdata = str(data)
        strlen = len(strdata)
        i = 0
        while i < strlen:
            run = io.StringIO()
            control = io.StringIO()

            escape_next = False
            control_reached = False
            control_index = -1
            while i < strlen:
                char = strdata[i]
                i += 1
                if escape_next:
                    run.write(char)
                    escape_next = False
                    continue
                if char == '\\': # the "escape" control character
                    escape_next = True
                    continue
                if char == control_start:
                    control_reached = True
                    control_index = i - 1
                    break
                run.write(char)

            if control_reached:
                # note: we already passed the starting control character
                control_ended = False
                while i < strlen:
                    char = strdata[i]
                    i += 1
                    if char == control_end:
                        control_ended = True
                        break # note: we are just past the ending control char 
                    control.write(char)
                    
                if not control_ended:
                    raise ValueError(f"Formatted print string does not complete format directive at index {control_index}")

                # parse the control string and write out the fragment
                strrun = run.getvalue()
                strcontrol = control.getvalue().upper()
                self.stream.write(strrun)
                if strcontrol in Style.__dict__:
                    style = Style.__dict__[strcontrol]
                    self.style(style)
                elif strcontrol in Color.__dict__:
                    foreground = Color.__dict__[strcontrol]
                    self.color(foreground)
                else:
                    raise ValueError(f"Unknown format directive '{strcontrol}'")
            else:
                # make sure we don't forget to print the ending run if it isn't
                # followed by a format directive.
                strrun = run.getvalue()
                if len(strrun) > 0:
                    self.stream.write(strrun)
            # i += 1 is not required here

        if not persistent_style: # make sure to automatically reset if requested
            self.style(Style.RESET)
        # end print
    
    def println(self, data, persistent_style=False):
        self.print(data, persistent_style)
        self.stream.write('\n')

    def flush(self):
        self.stream.flush()
    
    # ----- Direct CSI interface ------
    def write_csi(self, *params):
        buffer = io.StringIO()
        buffer.write("\x1b[")
        for param in params:
            buffer.write(str(param))
        self.stream.write(buffer.getvalue())
    
    # ----- Cursor Positioning -----
    def cursor_up(self, n=1):
        self.write_csi(n, 'A')
    
    def cursor_down(self, n=1):
        self.write_csi(n, 'B')
    
    def cursor_forward(self, n=1):
        self.write_csi(n, 'C')
    
    def cursor_back(self, n=1):
        self.write_csi(n, 'D')
    
    def cursor_next_line(self, n=1):
        self.write_csi(n, 'E')
    
    def cursor_prev_line(self, n=1):
        self.write_csi(n, 'F')
    
    def cursor_absolute(self, col=1):
        self.write_csi(col, 'G')
    
    def cursor_begin(self):
        self.cursor_absolute(1)

    def cursor_position(self, x=1, y=1):
        self.write_csi(y, ';', x, 'H')
    
    def cursor_move(self, dx=1, dy=0):
        if dy > 0:
            self.cursor_down(dy)
        elif dy < 0:
            self.cursor_up(-dy)
        
        if dx > 0:
            self.cursor_forward(dx)
        elif dx < 0:
            self.cursor_back(-dx)
    
    # ----- Clearing Display ------
    def erase_display(self):
        self.write_csi(2, 'J')
    
    def erase_display_goto_origin(self):
        self.erase_display()
        self.cursor_

    def erase_forward(self):
        self.write_csi(0, 'K')
    
    def erase_back(self):
        self.write_csi(1, 'K')
    
    def erase_line(self, carriage_return=False):
        self.write_csi(2, 'K')
        if carriage_return:
            self.cursor_absolute(1)
    
    def erase_line_cr(self):
        self.erase_line(carriage_return=True)
    
    # ----- Select Graphic Rendition (Styling) -----
    def write_sgr(self, n):
        self.write_csi(n, 'm')

    def style(self, style):
        self.write_sgr(style)

    def color(self, foreground=None, background=None):
        if foreground is not None and background is not None:
            self.write_sgr(foreground[0], ';', background[1])
        elif foreground is not None:
            self.write_sgr(foreground[0])
        elif background is not None:
            self.write_sgr(background[1])


class Console:

    def __init__(
        self,
        fin=sys.stdin, 
        fout=sys.stdout, 
        ferr=sys.stderr):

        self.inp = InputDevice(fin, "fin")
        self.out = OutputDevice(fout, "fout")
        self.err = OutputDevice(ferr, "ferr")