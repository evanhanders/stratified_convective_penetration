"""
This script distributes output files across all MPI processes and then loops over 
every write, letting the user simply generate their own output tasks.

To see how it works, run the rayleigh_benard.py example, then type:
    mpirun -n 4 python3 uniform_output_task.py ./

Usage:
    uniform_output_task.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of dedalus handler [default: slices]
    --out_dir=<out_dir>               Name of output directory [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of first Dedalus file to read [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from docopt import docopt
args = docopt(__doc__)

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularColorbarPlotGrid as RCPG

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir'] #TODO: change default to apply to your own simulation.
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
out_dir    = args['--out_dir'] #TODO: Note output file directory, maybe change.
n_files     = args['--n_files']
dpi         = int(args['--dpi'])
if n_files is not None: 
    n_files = int(n_files)

reader = SingleTypeReader(root_dir, data_dir, out_dir, n_files=n_files, distribution='even-write')
prof_reader = SingleTypeReader(root_dir, 'profiles', out_dir, n_files=n_files, distribution='even-write')
output_tasks = ['s1', 'enstrophy'] #TODO: Update this with the names of your output tasks.

grid = RCPG(num_rows=1, num_cols=2, col_inch=4, row_inch=2)
ax1 = grid.axes['ax_0-0']
ax2 = grid.axes['ax_0-1']
cax1 = grid.cbar_axes['ax_0-0']
cax2 = grid.cbar_axes['ax_0-1']
if 'upwards' in root_dir.lower():
    upwards = True
else:
    upwards = False
while reader.writes_remain():
    prof_reader.writes_remain()
    dsets, ni = reader.get_dsets(output_tasks)
    prof_dsets, prof_ni = prof_reader.get_dsets(output_tasks)

    time_data = dsets[output_tasks[0]].dims[0]
    sim_time = time_data['sim_time'][ni]
    write_num = time_data['write_number'][ni]  

    #TODO: do your own output task stuff here, this is just an example!
    s1_data = dsets['s1'][ni]
    enstrophy_data = dsets['enstrophy'][ni]
    s1_mean_data = prof_dsets['s1'][prof_ni]
    x = match_basis(dsets['s1'], 'x')
    z = match_basis(dsets['s1'], 'z')
    zz, xx = np.meshgrid(z, x)

    s1_fluc = s1_data - s1_mean_data
    if upwards:
        vals = np.abs(s1_fluc[(z < z.max()*0.4)*(z > z.max()*0.05)])
    else:
        vals = np.abs(s1_fluc[(z > z.max()*0.6)*(z < z.max()*0.95)])
    vals = np.sort(vals)
    vals = vals[:int(0.98*len(vals))]
    cz_fluc = 0.7*np.max(vals)
    divnorm=colors.TwoSlopeNorm(vmin=-cz_fluc, vcenter=0., vmax=cz_fluc)
    pm1 = ax1.pcolormesh(xx,zz,s1_fluc,cmap='RdBu_r', norm=divnorm)
    cb1 = plt.colorbar(pm1, cax=cax1, orientation='horizontal')

    pm2 = ax2.pcolormesh(xx, zz, enstrophy_data, cmap='Purples_r')
    cb2 = plt.colorbar(pm2, cax=cax2, orientation='horizontal')

    plt.suptitle('t = {:.4e}'.format(sim_time))
    grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(reader.out_dir, out_dir, int(write_num+start_fig+1)), dpi=dpi, bbox_inches='tight')
    for k, ax in grid.axes.items():
        ax.clear()
    for k, cax in grid.cbar_axes.items():
        cax.clear()
