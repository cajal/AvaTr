import os
import glob
import json


def to_html_comparison(roots, methods, n_mix=2, n_examples=10):
    list_dirs = []

    for root in roots:
        dirs = glob.glob(f'{root}/ex_*')
        list_dirs.append(dirs)

    visualizer = HTMLVisualizer('index.html')

    header = ['Filename', 'Input Mixed Audio', 'GT Audio 1']

    for method in methods:
        header += [f'{method}']

    visualizer.add_header(header)

    vis_rows = []

    for j, dd in enumerate(list_dirs[0]):
        if j > n_examples:
            break

        with open(os.path.join(dd, 'metrics.json'), 'r') as fp:
            metrics = json.load(fp)
            mix_id = metrics['mix_id'].replace('_', ' v.s. ')
            row_elements = [{'text': mix_id + ': sisnr=%.2f (%.2f)' % (metrics['si_sdr'], metrics['input_si_sdr'])}]
        row_elements.append( {'audio': os.path.join(dd, 'mixture.wav')} )
        row_elements.append( {'audio': os.path.join(dd, 's0.wav')} )

        for i in range(len(methods)):
            dd2 = list_dirs[i][j]
            row_elements.append( {'audio': os.path.join(dd2, f's0_estimate.wav')} )

        vis_rows.append( row_elements )

    # sort by si_sdr
    vis_rows = sorted(vis_rows, key=lambda row:float(row[0]['text'].split('=')[1][:5]))

    visualizer.add_rows(vis_rows)
    visualizer.write_html()



def to_html(root, n_mix=2):
    dirs = glob.glob(f'{root}/ex_*')

    visualizer = HTMLVisualizer('index.html')

    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, n_mix+1):
        header += ['GroundTruth Audio {}'.format(n),
                   'Predicted Audio {:d}'.format(n)]
    visualizer.add_header(header)

    vis_rows = []

    for dd in dirs:
        with open(os.path.join(dd, 'metrics.json'), 'r') as fp:
            metrics = json.load(fp)
            mix_id = metrics['mix_id'].replace('_', ' v.s. ')
            row_elements = [{'text': mix_id + ': sisnr=%.2f (%.2f)' % (metrics['si_sdr'], metrics['input_si_sdr'])}]
        row_elements.append( {'audio': os.path.join(dd, 'mixture.wav')} )

        for i in range(n_mix):
            row_elements.append( {'audio': os.path.join(dd, f's{i}.wav')} )
            row_elements.append( {'audio': os.path.join(dd, f's{i}_estimate.wav')} )

        vis_rows.append( row_elements )

    # sort by si_sdr
    vis_rows = sorted(vis_rows, key=lambda row:float(row[0]['text'].split('=')[1][:5]))

    visualizer.add_rows(vis_rows)
    visualizer.write_html()


def to_html_single(root, pattern='*'):
    dirs = glob.glob(f'{root}/ex_{pattern}')

    visualizer = HTMLVisualizer(f'{root}/index.html')

    header = ['Filename', 'Input Mixed Audio']
    header += ['GroundTruth Audio {}'.format(1),
               'Predicted Audio {:d}'.format(1)]
    visualizer.add_header(header)

    vis_rows = []

    for dd in dirs:
        with open(os.path.join(dd, 'metrics.json'), 'r') as fp:
            metrics = json.load(fp)
            mix_id = metrics['mix_id'].replace('_', ' v.s. ')
            row_elements = [{'text': mix_id + ': sisnr=%.2f (%.2f)' % (metrics['si_sdr'], metrics['input_si_sdr'])}]
        row_elements.append( {'audio': os.path.join(dd, 'mixture.wav')} )

        row_elements.append( {'audio': os.path.join(dd, f's{1}.wav')} )
        row_elements.append( {'audio': os.path.join(dd, f's{1}_estimate.wav')} )

        vis_rows.append( row_elements )

    # sort by si_sdr
    #vis_rows = sorted(vis_rows, key=lambda row:float(row[0]['text'].split('=')[1][:5]))
    vis_rows = sorted(vis_rows, key=lambda row:int(row[0]['text'].split(' v.s. ')[0].split('-')[2]))

    visualizer.add_rows(vis_rows)
    visualizer.write_html()



class HTMLVisualizer():
    def __init__(self, fn_html):
        self.fn_html = fn_html
        self.content = '<table>'
        self.content += '<style> table, th, td {border: 1px solid black;} </style>'

    def add_header(self, elements):
        self.content += '<tr>'
        for element in elements:
            self.content += '<th>{}</th>'.format(element)
        self.content += '</tr>'

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, elements):
        self.content += '<tr>'

        # a list of cells
        for element in elements:
            self.content += '<td>'

            # fill a cell
            for key, val in element.items():
                if key == 'text':
                    self.content += val
                elif key == 'image':
                    self.content += '<img src="{}" style="max-height:256px;max-width:256px;">'.format(val)
                elif key == 'audio':
                    self.content += '<audio controls><source src="{}"></audio>'.format(val)
                elif key == 'video':
                    self.content += '<video src="{}" controls="controls" style="max-height:256px;max-width:256px;">'.format(val)
            self.content += '</td>'

        self.content += '</tr>'

    def write_html(self):
        self.content += '</table>'
        with open(self.fn_html, 'w') as f:
            f.write(self.content)
