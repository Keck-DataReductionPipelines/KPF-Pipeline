"""L0 quicklook plots for raw KPF detector images."""


class PlotL0:
    """
    Quicklook plots for KPF L0 (raw CCD) data.

    Takes a KPF0 object and generates plots of the raw detector images.
    Pure visualization — no science computation.

    Args:
        l0_obj: KPF0 data object.
        output_dir: Directory to save PNG files. None = return Figure only.
    """

    def __init__(self, l0_obj, output_dir=None):
        self.l0 = l0_obj
        self.output_dir = output_dir
        self.obs_id = getattr(l0_obj, 'obs_id', None) or ''
        self.name = ''
        if 'PRIMARY' in l0_obj.headers:
            self.name = l0_obj.headers['PRIMARY'].get('OBJECT', '')
