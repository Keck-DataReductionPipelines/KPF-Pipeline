import matplotlib.pyplot as plt

from kpfpipe.models.level1 import KPF1

fpath = ['template_before.fits']
order = 23

def plot(self, data: KPF1, order: int,
         color = 'b') -> None:
    ''' '''
    # fig = plt.figure()
    plt.plot(self._wave[order], self._spec[order], 
                label=comment,
                color=color,
                linewidth=0.5)


if __name__ == '__main__':
    data = []
    fig = plt.figure()
    for f in fpath:
        spec = KPF1()
        spec.from_fits(f)
        data.append(spec)
        plot(spec, 23)