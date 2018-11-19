import os, time, datetime, cv2
import numpy as np

class ColorfulPrint(object):# {{{

    """Docstring for ColorfulPrint. """

    def __init__(self):
        """nothing needs to be define """
        self.colors = {
            'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
            'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
        }
        self.done = '(#g)done(#)'

    def trans(self, *args, auto_end=True):
        s = ' '.join(map('{}'.format, args))
        s = s.replace('(##)', '\033[0m')
        s = s.replace('(#)', '\033[0m')
        for color, value in self.colors.items():
            color_tag = '(#%s)'%color
            s_color_tag = '(#%s)'%color[0]
            s = s.replace(color_tag, '\033[%dm'%value).\
                    replace(s_color_tag, '\033[%dm'%value)
        if auto_end:
            s = s + '\033[0m'
        return s

    def err(self, *args, **kwargs):
        return self('(#r)[ERR](#)', *args, **kwargs)

    def log(self, *args, **kwargs):
        return self('(#blue)[LOG](#)', *args, **kwargs)

    def wrn(self, *args, **kwargs):
        return self('(#y)[WRN](#)', *args, **kwargs)

    def suc(self, *args, **kwargs):
        return self('(#g)[SUC](#)', *args, **kwargs)

    def __call__(self, *args, **kwargs):
        print(self.trans(*args), **kwargs)


    def square(self, color=(0, 0, 0)):

        def rgb(red, green, blue):
            """
            Calculate the palette index of a color in the 6x6x6 color cube.
            The red, green and blue arguments may range from 0 to 5.
            """
            return 16 + (red * 36) + (green * 6) + blue
        color = rgb(*[int(each / 256 * 6) for each in color])
        return '\x1b[48;5;{}m  \x1b[0m'.format(color)


    def print_img(self, img, width=100, height=100, clear=False):
        if clear:
            os.system('clear')
        h, w = img.shape[:2]
        rate = min(width/w, height/h)
        img = cv2.resize(img, (int(rate * w), int(rate * h)))
        h, w = img.shape[:2]
        s = ''
        for i in range(h):
            for j in range(w):
                s += cp.square(img[i, j])
            s += '\n'
        print(s)


cp = ColorfulPrint()
class procedure:
    def __init__(self, msg, same_line=True):
        self.msg = msg
        self.time = time.time()
        cp.log(msg, end='\r'if same_line else '\n')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        cp.suc(self.msg, cp.done, 'time:(#b)', datetime.timedelta(seconds=time.time() - self.time))

# with procedure('running epoch #1', True):
#     time.sleep(5)
#     pass

# }}}


def file_stat(path):
    if os.path.isdir(path):
        return 'dir'
    if os.path.isfile(path):
        return 'file'
    if os.path.islink(path):
        return 'link'
    return None

def touchdir(path):
    dirname = os.path.dirname(path)
    file_type = file_stat(dirname)
    assert file_type in {'dir', None},\
            cp.trans('(#r)[ERR](#) father directory \'(#y){}(#)\' is neither a directory nor empty'.format(dirname))
    if file_type == None:
        os.makedirs(dirname)

    if file_stat(path):
        for i in range(1000):
            p = '{}({})'.format(path, i)
            if os.path.isdir(p) or os.path.isfile(p):
                continue
            return p
    return path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    cp('(#r)hi')
