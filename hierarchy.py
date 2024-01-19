'''
Tools for managing Michael's physiology data
'''

# pylint: disable=C0103, R0912, R0914

import subprocess
from itertools import product
from inspect import isclass
from pathlib import Path
from math import ceil, isnan
import pickle
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from scipy import io as spio
from scipy import stats
from sklearn.decomposition import PCA
import h5py
from benlib.utils import loadmat, Progress, sahani_quick
from benlib.strf import split_tx, tensorize_segments
from benlib.plot import scatter_cc, scatter_cc_2, scatter_cc_multi
from benlib.plot import add_panel_labels, label_bottom_right

from scipy.stats import kruskal, wilcoxon

AREA_ORDER = ['ic', 'mgb', 'mgbm', 'ac']
AREA_LABELS = {
    'ic': 'IC', 'mgb':'MGB', 'mgbm': 'MGBm', 'ac': 'A1'
}

cmap = get_cmap('tab10')
all_brain_areas = ['ic', 'mgb', 'mgbm', 'ac']
all_colors = [cmap(2), cmap(1), cmap(3), cmap(0)]
BRAIN_AREA_COLORS = {a:c for a,c in zip(all_brain_areas, all_colors)}

PANEL_LABELS_LOWER_CASE=True

def get_root_dirs():
    '''
    Find michael-data and michael-data/hierarchy
    '''
    pth = Path('.').resolve()
    return pth.parent.parent, pth.parent

MICHAEL_ROOT, HIERARCHY_ROOT = get_root_dirs()

STIMULI = [
    {'stim_type': 12,
     'gridcode': 'type_12_drc%d_dur25_contrast%d',
     'gridcode_params': [[1, 2], [20, 40]],
     'reverse_params': False
    },
    {'stim_type': 19,
     'gridcode': 'type_19_drc%d_dur25_contrast%d',
     'gridcode_params': [range(1,16), [20, 40]],
     'reverse_params': True
    },
    {'stim_type': 25,
     'gridcode': 'type_25_drc%d_dur25_contrast%d',
     'gridcode_params': [[1, 2], [20, 40]],
     'reverse_params': False
    }
]

def stimulus_order(stim_types):
    if isinstance(stim_types, int):
        stim_types = [stim_types]
    order = []
    for stimulus in STIMULI:
        if stimulus['stim_type'] in stim_types:
            if stimulus['reverse_params']:
                all_params = stimulus['gridcode_params'][::-1]
                for params in product(*all_params):
                    order.append(stimulus['gridcode'] % (params[::-1]))
            else:
                for params in product(*stimulus['gridcode_params']):
                    order.append(stimulus['gridcode'] % (params))
    return order

class Stimulus():
    '''
    Load/process stimuli
    '''

    def __init__(self, stim_types):
        if isinstance(stim_types, str):
            stim_types = int(stim_types)

        if isinstance(stim_types, int):
            stim_types = [stim_types]

        self.stim_types = stim_types
        self.grid_file_codes = stimulus_order(stim_types)

        filepath = Path(HIERARCHY_ROOT, 'collated-data-2022', 'A2_stimGrids.mat')
        all_stimgrids = spio.loadmat(filepath, struct_as_record=False, squeeze_me=True)

        self.chord_dur_ms = 25
        self.freqs = np.logspace(np.log10(1), np.log10(64), 25) * 1000

        self.X_tf = []
        self.segment_lengths = []
        self.c_t = []
        self.t_bins_from_onset = []
        self.t_ms_from_onset = []

        for code in self.grid_file_codes:
            stim_info = all_stimgrids['stimGrids'].__dict__[code].__dict__
            self.X_tf.append(stim_info['grid'].transpose())
            self.segment_lengths.append(stim_info['grid'].shape[1])
            self.c_t.append(stim_info['contrast'])
            self.t_bins_from_onset.append(np.arange(stim_info['grid'].shape[1]))
            self.t_ms_from_onset.append(np.arange(stim_info['grid'].shape[1]) * self.chord_dur_ms)

        self.t_ms_continuous = split_tx(
            np.arange(0,sum(self.segment_lengths)) * self.chord_dur_ms,
            self.segment_lengths)

    def tensor(self, n_h=15, n_fut=0):
        return tensorize_segments(self.X_tf, n_h=n_h, n_fut=n_fut)

    def indices(self):
        if self.stim_types == [12]:
            idxes = {}

            idxes['all'] = {}
            idxes['all']['all'] = np.arange(0, 6400)
            idxes['all']['lo'] = np.concatenate((np.arange(1600), np.arange(3200, 4800)))
            idxes['all']['hi'] = np.concatenate((np.arange(1600, 3200), np.arange(4800, 6400)))

            # new, improved, drop 20 initial chords not 21
            idxes['valid20'] = {}
            idxes['valid20']['all'] = np.concatenate((np.arange(20, 1600), np.arange(1600+20, 3200),
                                                    np.arange(3200+20, 4800), np.arange(4800+20, 6400)))
            idxes['valid20']['lo'] = np.intersect1d(idxes['valid20']['all'], idxes['all']['lo'])
            idxes['valid20']['hi'] = np.intersect1d(idxes['valid20']['all'], idxes['all']['hi'])

            # idxes_check = {}
            # n_t = sum(self.segment_lengths)
            # c_t = np.concatenate(self.c_t)
            # t_bins_from_onset = np.concatenate(self.t_bins_from_onset)

            # idxes_check['all'] = {}
            # idxes_check['all']['all'] = np.arange(0, n_t)

            # idxes_check['all']['lo'] = np.where(c_t==0)[0]
            # idxes_check['all']['hi'] = np.where(c_t==1)[0]

            # idxes_check['valid20'] = {}
            # idxes_check['valid20']['all'] = np.where(t_bins_from_onset>=20)[0]
            # idxes_check['valid20']['lo'] = np.intersect1d(idxes_check['valid20']['all'], idxes_check['all']['lo'])
            # idxes_check['valid20']['hi'] = np.intersect1d(idxes_check['valid20']['all'], idxes_check['all']['hi'])

            # return idxes, idxes_check

        elif self.stim_types == [19]:
            idxes = {}

            n_t = sum(self.segment_lengths)
            c_t = np.concatenate(self.c_t)
            t_bins_from_onset = np.concatenate(self.t_bins_from_onset)

            idxes['all'] = {}
            idxes['all']['all'] = np.arange(0, n_t)

            idxes['all']['lo'] = np.where(c_t==0)[0]
            idxes['all']['hi'] = np.where(c_t==1)[0]

            idxes['valid20'] = {}
            idxes['valid20']['all'] = np.where(t_bins_from_onset>=20)[0]
            idxes['valid20']['lo'] = np.intersect1d(idxes['valid20']['all'], idxes['all']['lo'])
            idxes['valid20']['hi'] = np.intersect1d(idxes['valid20']['all'], idxes['all']['hi'])

        return idxes

    def get_indices(self, spec, k_folds, fold_idxes_to_include=None):
        all_idxes = self.indices()
        idxes = all_idxes[spec[0]][spec[1]]

        if k_folds=='all_train':
            return [(idxes, [])]

        if k_folds==None or k_folds==1:
            splits = np.array_split(idxes, 10)
            return [(np.concatenate(splits[:-1]), splits[-1])]

        splits = np.array_split(idxes, k_folds)

        k_folds = []
        for i in range(len(splits)):
            test = splits[i]
            train = np.concatenate(splits[:i]+splits[i+1:])
            k_folds.append((train, test))

        # checks

        # no overlap between training and test sets
        no_overlap = [True if len(set(train).intersection(set(test)))==0 else False for train,test in k_folds]
        assert(all(no_overlap))

        # all indices should be part of exactly one test set
        all_test = np.sort(np.concatenate([k[1] for k in k_folds]))
        assert(all(all_test==idxes))

        # select only certain folds
        if fold_idxes_to_include:
            k_folds = [f for i,f in enumerate(k_folds) if i in fold_idxes_to_include]

        return k_folds

    def show(self):
        plt.figure(figsize=(12,8))
        ax = plt.gca()

        whole_spec = np.concatenate(self.X_tf, axis=0)
        n_t, n_f = whole_spec.shape
        print(f'{n_t} time bins, {n_f} frequencies')
        print(f'{len(self.X_tf)} segments, with lengths:')
        print(f'{self.segment_lengths}')

        plt.imshow(np.transpose(whole_spec),
                   interpolation='nearest',
                   aspect='auto',
                   origin='lower')

        total = 0
        for length in self.segment_lengths[1:]:
            total = total+length
            plt.plot([total, total], ax.get_ylim(), 'k-')

        c_t = np.concatenate(self.c_t)
        plt.plot(c_t)

        ticks = range(0,n_t+1,int(10000/self.chord_dur_ms))
        labels = [int(t * self.chord_dur_ms/1000) for t in ticks]
        ax.set_xticks(ticks, labels)
        ax.set_xlabel('Time (sec)')

        tick_idx = [0,4,9,14,19,24]
        tick_freqs = [int(round(f/1000)) for f in np.array(self.freqs)[tick_idx]]
        ax.set_yticks(tick_idx, tick_freqs)
        ax.set_ylabel('Frequency (kHz)')

        rng = [int(round(np.min(whole_spec))), int(round(np.max(whole_spec)))]
        cbar = plt.colorbar(fraction=0.01, pad=0.02)
        cbar.ax.set_yticks(range(rng[0], rng[1]+1, 10))
        cbar.ax.set_ylabel('dB SPL')


class Clusters():
    '''
    Load/process cluster data from all Michael expts
    '''
    def __init__(self, stim_types=12, spike_times=False):

        # load all clusters where ALL the requested stim types were presented
        self.clusters = []

        if isinstance(stim_types, str):
            stim_types = int(stim_types)

        if isinstance(stim_types, int):
            stim_types = [stim_types]

        self.stim_types = stim_types
        self.stim_types_str = '-'.join([str(t) for t in self.stim_types])
        self.grid_file_codes = stimulus_order(stim_types)
        self.stimulus = Stimulus(stim_types)
        self.results_subdir = 'stim-'+ '-'.join([str(s) for s in self.stim_types])

        # 27,28,30 are timecourse data
        if any([s>26 for s in stim_types]):
            matfile = 'A1_allclusters_timecourse.mat'
        else:
            matfile = 'A1_allclusters_basic_and_opto.mat'

        raw_data = spio.loadmat(Path(HIERARCHY_ROOT, 'collated-data-2022', matfile),
                                    struct_as_record=False, squeeze_me=True)

        if spike_times:
            spike_data = h5py.File(Path(HIERARCHY_ROOT, 'collated-data-2023', matfile), 'r')

            def deref(reference):
                return spike_data[reference]

        self.all_pen_info = []
        for cluster_idx, cluster in enumerate(raw_data['clusters']):
            d = cluster.__dict__
            cluster_stim_types = [int(c.split('_')[1]) for c in d['gridFileCodes']]

            if (d['expt'], d['pen']) not in self.all_pen_info:
                self.all_pen_info.append((d['expt'], d['pen']))

            # check that all requested stim types were presented
            if not set(stim_types).issubset(set(cluster_stim_types)):
                continue

            if spike_times:

                cluster_data = deref(spike_data['clusters']['spikeTimes'][cluster_idx][0])
                n_stim = cluster_data.shape[0]


                all_spike_times = []
                for stim_idx in range(n_stim):
                    stim_data = deref(cluster_data[stim_idx][0])
                    n_trials = stim_data[:].shape[0]

                    stim_spikes = []
                    for trial_idx in range(n_trials):
                        trial_spikes = deref(stim_data[:][trial_idx][0])[:]
                        if not isinstance(trial_spikes, np.ndarray):
                            # print('scalar', trial_spikes)
                            trial_spikes = np.array([trial_spikes])
                        elif len(trial_spikes)==2:
                            assert trial_spikes[0] == 0 and trial_spikes[1] == 1
                            trial_spikes = np.array(())
                        else:
                            trial_spikes = np.squeeze(trial_spikes)

                        stim_spikes.append(trial_spikes*1000)
                    all_spike_times.append(stim_spikes)

            if d['optogenetic']:
                d['optogenetic'] = True
            else:
                d['optogenetic'] = False

            # get light state on each trial
            if d['optogenetic']:
                light_on = np.array([p[3]>0 for p in d['stimParams']])

                stim_idx = [[], []]
                for code in self.grid_file_codes:
                    fnd = np.where((d['gridFileCodes']==code) & (light_on==False))[0]
                    assert len(fnd)==1
                    stim_idx[0].append(fnd[0])

                    fnd = np.where((d['gridFileCodes']==code) & (light_on==True))[0]
                    assert len(fnd)==1
                    stim_idx[1].append(fnd[0])

            else:
                stim_idx = []
                for code in self.grid_file_codes:
                    fnd = np.where(d['gridFileCodes']==code)[0]
                    assert len(fnd)==1
                    stim_idx.append(fnd[0])

            if d['optogenetic']:
                for cond_idx, cond_name in enumerate(['light_off', 'light_on']):
                    cond = {}

                    # stimuli 0::2 are light off; 1::2 are light on
                    cond['y_td_segments'] = \
                        [p.transpose() for p in d['allPsthes'][stim_idx[cond_idx]]]
                    cond['y_t_segments'] = \
                        [np.mean(p, axis=1) for p in cond['y_td_segments']]
                    cond['y_td'] = \
                        np.concatenate(cond['y_td_segments'], axis=0)
                    cond['y_t'] = \
                        np.concatenate(cond['y_t_segments'], axis=0)
                    cond['signal_power'], cond['noise_power'], cond['total_power'] = \
                        sahani_quick(cond['y_td'])
                    if cond['signal_power'] == 0:
                        cond['noiseratio'] = np.nan
                    else:
                        cond['noiseratio'] = cond['noise_power'] / cond['signal_power']
                    if spike_times:
                        cond['spike_times'] = [all_spike_times[idx] for idx in stim_idx[cond_idx]]

                    d[cond_name] = cond

                # make 'light off' condition the primary data set
                d.update(d['light_off'])

            else:
                d['y_td_segments'] = [p.transpose() for p in d['allPsthes'][stim_idx]]
                # d['y_td_segments'] = [d['allPsthes'][idx].transpose() for idx in stim_idx]
                d['y_t_segments'] = [np.mean(p, axis=1) for p in d['y_td_segments']]
                d['y_td'] = np.concatenate(d['y_td_segments'], axis=0)
                d['y_t'] = np.concatenate(d['y_t_segments'], axis=0)
                d['signal_power'], d['noise_power'], d['total_power'] = sahani_quick(d['y_td'])
                d['noiseratio'] = d['noise_power'] / d['signal_power']

                if spike_times:
                    d['spike_times'] = [all_spike_times[int(idx)] for idx in stim_idx]

            d['brain_area'] = d['brainArea'].lower()
            d['cluster_idx'] = d['clusterIdx']
            d['unique_id'] = '%s-e%03d-p%03d-c%03d' % \
                (d['brain_area'], d['expt'], d['pen'], d['cluster_idx'])

            # unique penetration idx, +2 to maintain backward compatibility
            if (d['expt'], d['pen']) not in self.all_pen_info:
                self.all_pen_info.append((d['expt'], d['pen']))
            d['unique_penetration_idx'] = self.all_pen_info.index((d['expt'], d['pen'])) + 1

            d['anaesthetised'] = d['anaesthetised']==1
            d['awake'] = not d['anaesthetised']
            d['state'] = 'anaesthetised' if d['anaesthetised'] else 'awake'

            # d['excluded'] = d['unique_id'] in EXCLUDED_UNITS

            self.clusters.append(d)

        self.n_clusters = len(self.clusters)
        self.cluster_idx = 0
        brain_areas = list(set([c['brain_area'] for c in self.clusters]))
        self.brain_areas = sorted(brain_areas, key=lambda a: AREA_ORDER.index(a))

    def get_data(self, idx):
        '''
        Return all data for a given cluster
        '''
        return self.clusters[idx].__dict__.copy()

    def get_sorted(self, **kwargs):
        '''
        Return data, sorted by kwarg in the specified order, e.g.
        unique_id=['id1', 'id2']
        '''
        key = list(kwargs)
        if len(key) > 1:
            raise ValueError('Can only sort clusters by a single key')
        key = key[0]

        vals = [c[key] for c in self.clusters]

        clusters = []

        for val in kwargs[key]:
            idx = vals.index(val)
            clusters.append(self.clusters[idx])

        return clusters

    def mark_excluded_units(self):
        # this only applies to stimulus 12 -- main analyses for paper

        self.load_analyses(['coch_kernel_main', 'a2a_kernels_main',
                                'coch_kernel_sigmoid_main', 'a2a_kernels_sigmoid_main'])

        def coch_kernel_main_ln_cc_norm_test(cluster, regressor_area=None):
            return cluster['coch_kernel_sigmoid_main']['sigmoid_fits'][-1]['sigmoid']['cc_norm_test'][0]

        def a2a_kernels_main_ln_cc_norm_test(cluster, regressor_area):
            return cluster['a2a_kernels_sigmoid_main']['sigmoid_fits'][regressor_area][-1]['sigmoid']['cc_norm_test'][0]

        for c in self.clusters:
            c['excluded'] = False

            if np.isnan(coch_kernel_main_ln_cc_norm_test(c, 'ic')):
                c['excluded']= True

            # A2A kernels -- LN
            for regressors in ['ic', 'mgb', 'ac', 'ic_mgb', 'ic_mgb_ac']:
                try:
                    if np.isnan(a2a_kernels_main_ln_cc_norm_test(c, regressors)):
                        c['excluded'] = True
                except:
                    c['excluded'] = True

        sel_nr = self.select_data(noiseratio=200)
        sel = self.select_data(noiseratio=200, excluded=False)
        print('%d units with noiseratio<200, excluding %d:' % (len(sel_nr), len(sel)))
        n_ic = sum([c['brain_area']=='ic' for c in sel])
        n_mgb = sum([c['brain_area']=='mgb' for c in sel])
        n_ac = sum([c['brain_area']=='ac' for c in sel])
        print('%d IC, %d MGB and %d AC remaining' % (n_ic, n_mgb, n_ac))

    def build_param_str(self, func, params):
        if isinstance(func, str):
            param_str = [func]
        else:
            param_str = [func.__name__]
        for name, value in params.items():
            if isclass(value):
                value = value.__name__.split('.')[-1]
            elif isinstance(value, list):
                value = '-'.join([str(i) for i in value])
            param_str.append('%s=%s' % (name, str(value)))
        return '--'.join(param_str)

    def build_fieldname(self, func, params):
        if isinstance(func, str):
            param_str = [func]
        else:
            param_str = [func.__name__]
        for name, value in params.items():
            if isclass(value):
                value = value.__name__.split('.')[-1]
            elif isinstance(value, list):
                value = '_'.join([str(i) for i in value])
            param_str.append('%s_%s' % (name, str(value)))
        return '__'.join(param_str)

    def create_slurm(self, func, cluster_id, params, limit_time="30",
                     partition="short", execute=False, load_analyses_first=None):
        preamble = '''\
#!/bin/bash\

#SBATCH --time=%s
#SBATCH --job-name=%s
#SBATCH --cpus-per-task=8
#SBATCH --clusters=htc
#SBATCH --partition=%s
#SBATCH --qos=priority

echo == START == `date`

module purge

module load Anaconda3

export CONPREFIX=$DATA/conda_sklearn
source activate $CONPREFIX
export PYTHONPATH=/home/dpag0036/data/pylib

cd /home/dpag0036/data/michael-data/hierarchy/
python big_analysis.py "%s" %s %s "%s" %s "%s"

echo == FINISH == `date`

'''

        raw_param_str = self.build_param_str(func, params)

        slurm_dir = Path('./slurm/slurm--%s' % (raw_param_str))
        slurm_dir.mkdir(parents=True, exist_ok=True)

        results_dir = Path('./results/%s/%s' % (self.results_subdir, raw_param_str))

        param_str = cluster_id + '--' + raw_param_str
        filename =  param_str + '.slurm.sh'
        file_path = slurm_dir / filename
        with open(file_path, 'w') as f:
            f.write(preamble % (limit_time, cluster_id, partition,
                    str(self.stim_types),
                    func.__name__,
                    cluster_id,
                    str(params),
                    results_dir / (param_str + '.pkl'),
                    load_analyses_first))
        if execute:
            subprocess.call(['/usr/bin/sbatch', file_path])

    def run_analysis_htc(self, func, params, limit_time="00:10:00", partition="short",
                         ignore_existing=False, execute=False, load_analyses_first=None):
        for cluster in self.clusters:
            print('%s: ' % (cluster['unique_id']), end='')
            if not ignore_existing:
                raw_param_str = self.build_param_str(func, params)
                results_dir = Path('./results/%s/%s' % (self.results_subdir, raw_param_str))
                param_str = cluster['unique_id'] + '--' + raw_param_str
                results_path = results_dir / (param_str + '.pkl')
                if results_path.exists():
                    print('results already exist, skipping')
                    continue
            if 'state' in params: # messy
                if cluster['state'] != params['state'] and cluster['state'] not in params['state']:
                    print('wrong state (anaesthetised/awake), skipping')
                    continue
            self.create_slurm(func, cluster['unique_id'], params,
                              limit_time=limit_time, partition=partition, execute=execute,
                              load_analyses_first=load_analyses_first)
            print('created slurm file')

    @staticmethod
    def apply_func_to_cluster(func, cluster, clusters, params, fail_on_error=True):

        if fail_on_error:
            result = func(cluster, clusters, **params)
        else:
            try:
                result = func(cluster, clusters, **params)
            except Exception as err:
                print(err)
                return None

        if not result:
            return None

        r = {'unique_id': cluster['unique_id']}
        r[func.__name__] = result
        return r

    def run_analysis(self, func, params, save_results=True, multiple_results_files=False,
                     ignore_existing=False, this_data_segment=1, total_data_segments=1,
                     fail_on_error=False, load_analyses_first=None):
        '''
        Run analysis function func on clusters and save in results_file
        '''
        segment_length = ceil(len(self.clusters)/total_data_segments)
        first_cluster_idx = (this_data_segment-1) * segment_length

        if isinstance(func, str):
            func = eval(func)

        if load_analyses_first:
            if isinstance(load_analyses_first, str):
                load_analyses_first = [load_analyses_first]
            self.load_analyses(load_analyses_first)

        res = []
        if not save_results:
            progress = Progress(segment_length)
            for cluster in self.clusters[first_cluster_idx:first_cluster_idx+segment_length]:
                r = self.apply_func_to_cluster(func, cluster, self, params,
                                               fail_on_error=fail_on_error)
                if r:
                    res.append(r)
                progress.print()
            return res

        # else:

        raw_param_str = self.build_param_str(func, params)

        if multiple_results_files:
            results_dir = Path(HIERARCHY_ROOT, 'results', self.results_subdir, raw_param_str)
            progress_name = raw_param_str
            progress_file = results_dir.with_suffix('.log')
        else:
            results_dir = Path(HIERARCHY_ROOT, 'results', self.results_subdir)
            filename = self.build_param_str(func, params)
            results_path = Path(results_dir, filename).with_suffix('.pkl')
            progress_name = results_path.with_suffix('').name
            progress_file = results_path.with_suffix('.log')

        results_dir.mkdir(parents=True, exist_ok=True)

        progress = Progress(segment_length, progress_name, progress_file)

        if not multiple_results_files:
            if ignore_existing and results_path.exists():
                res = []
                print('Ignoring existing results file')
            else:
                try:
                    with open(results_path, 'rb') as file:
                        res = pickle.load(file)
                        print('Continuing previous run...')
                except IOError:
                    res = []
                    print('Starting from scratch...')

            processed_ids = [r['unique_id'] for r in res]

            for cluster in self.clusters[first_cluster_idx:first_cluster_idx+segment_length]:

                if cluster['unique_id'] in processed_ids and not ignore_existing:
                    print('Already have data for %s' % (cluster['unique_id']))
                    continue

                r = self.apply_func_to_cluster(func, cluster, self, params,
                                               fail_on_error=fail_on_error)
                if r:
                    res.append(r)

                progress.print()

            with open(results_path, 'wb') as file:
                pickle.dump(res, file)

            return res

        else: # multiple results files

            for cluster in self.clusters[first_cluster_idx:first_cluster_idx+segment_length]:
                print('%s: ' % (cluster['unique_id']), end='')

                param_str = cluster['unique_id'] + '--' + raw_param_str
                results_path = results_dir / (param_str + '.pkl')

                if results_path.exists():
                    if ignore_existing:
                        print('Ignoring existing results file')
                    else:
                        print('Results file already exists, skipping')
                        continue

                r = self.apply_func_to_cluster(func, cluster, self, params)
                if r:
                    with open(results_path, 'wb') as file:
                        pickle.dump(r, file)

                progress.print()

    @classmethod
    def save_res(cls, res, results_file):
        '''
        Dump res to file
        '''
        with open(results_file, 'wb') as file:
            pickle.dump(res, file)


    def save_figures(self, func, prefix, suffix='.eps'):
        '''
        Plot figure for each unit
        '''
        if suffix[0] != '.':
            suffix = '.' + suffix

        fig_dir = Path('./figures/%s' % (prefix))
        fig_dir.mkdir(parents=True, exist_ok=True)

        plt.ioff()

        for cluster in self.clusters:
            fig_file = fig_dir / ('%s-%s%s' % (prefix, cluster['unique_id'], suffix))
            if fig_file.exists():
                print('Figure %s exists, skipping...' % (fig_file.name))
                continue

            print('Plotting %s...' % cluster['unique_id'])

            func(cluster, self)

            plt.savefig(fig_file)
            plt.close()

        plt.ion()

    @classmethod
    def dump_results(cls, res, filename):
        '''
        Dump res to filename
        '''
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(res, file)

    def merge_results(self, filenames_or_res, new_fieldname=None):
        '''
        Merge results from a pickled list (or lists), based on unique_id
        '''
        unique_ids = [d['unique_id'] for d in self.clusters]

        if isinstance(filenames_or_res, (list, tuple)) and 'unique_id' in filenames_or_res[0]:
            for cluster in filenames_or_res:
                idx = unique_ids.index(cluster['unique_id'])
                if new_fieldname:
                    old_fieldname = [f for f in cluster if f not in ['unique_id']][0]
                    self.clusters[idx][new_fieldname] = cluster[old_fieldname]
                else:
                    self.clusters[idx].update(cluster)
            return

        if not isinstance(filenames_or_res, (list, tuple)):
            filenames_or_res = [filenames_or_res]

        for filename in filenames_or_res:
            if not filename.endswith('.pkl'):
                filename = 'results/%s/' % (self.results_subdir) + filename + '.pkl'
            with open(filename, 'rb') as file:
                data = pickle.load(file)

            for cluster in data:
                idx = unique_ids.index(cluster['unique_id'])
                if new_fieldname:
                    old_fieldname = [f for f in cluster if f not in ['unique_id']][0]
                    self.clusters[idx][new_fieldname] = cluster[old_fieldname]
                else:
                    self.clusters[idx].update(cluster)

    def load_results(self, func, params, fieldname=None):
        raw_param_str = self.build_param_str(func, params)

        if fieldname:
            new_fieldname = fieldname
        else:
            new_fieldname = self.build_fieldname(func, params)
        # print(f'Filename: {raw_param_str}')
        print(f'Fieldname: {new_fieldname}')

        try:
            # load single file results
            results_dir = Path(HIERARCHY_ROOT, 'results', self.results_subdir)
            filename = self.build_param_str(func, params)
            results_path = Path(results_dir, filename).with_suffix('.pkl')
            with open(results_path, 'rb') as fil:
                res = pickle.load(fil)
            print(f"Found single data file with {len(res)} clusters")

            for r in res:
                keys= [k for k in r.keys() if k != 'unique_id']
                r[keys[0]].update(
                    {'analysis_func': func,
                     'analysis_params': params})

            self.merge_results(res, new_fieldname)
            return
        except:
            pass

        try:
            # load multiple file results
            res = []
            results_dir = Path(HIERARCHY_ROOT, 'results', self.results_subdir, raw_param_str)
            print(results_dir)
            for filename in results_dir.iterdir():

                with open(filename, 'rb') as fil:
                    r = pickle.load(fil)
                    keys = [k for k in r.keys() if k != 'unique_id']

                    r[keys[0]].update(
                        {'analysis_func': func,
                         'analysis_params': params})
                res.append(r)

            print(f"Found data files with {len(res)} clusters")

            self.merge_results(res, new_fieldname)
        except:
            print('No data found -- param string was:')
            print(f'{raw_param_str}')
            return

    def select_data(self, **kwargs):
        '''
        Select data matching criteria, e.g. brain_area=['ic', 'mgb'], or
        exclude_brain_area=['ic', 'mgb']
        '''
        clusters = self.clusters
        for kw, vals in kwargs.items():
            if kw == 'noiseratio':
                if vals not in ['inf', np.inf]:
                    clusters = [c for c in clusters if not isnan(c['noiseratio'])]
                    clusters = [c for c in clusters if c['noiseratio'] <= vals]
            else:
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                if kw.startswith('exclude_'):
                    kw = kw[len('exclude_'):]
                    clusters = [c for c in clusters if c[kw] not in vals]
                else:
                    if vals[0]=='*':
                        clusters = [c for c in clusters if kw in c]
                    else:
                        clusters = [c for c in clusters if c[kw] in vals]

        return clusters

    def get_all_y_t(self, exclude_pens=None, **kwargs):
        '''
        Return a list of y_tu segments for all clusters, similar to X_tf
        BUGGY
        '''
        if not exclude_pens:
            exclude_pens = []

        clusters = self.select_data(exclude_unique_penetration_idx=exclude_pens, **kwargs)

        all_y_ts = np.stack([c['y_t'] for c in clusters], axis=1)

        return all_y_ts, [c['unique_id'] for c in clusters]

    def get_all_y_t_segments(self, exclude_pens=None, **kwargs):
        '''
        Return a list of y_tu segments for all clusters, similar to X_tf
        '''
        if not exclude_pens:
            exclude_pens = []

        clusters = self.select_data(exclude_unique_penetration_idx=exclude_pens, **kwargs)
        if len(clusters) == 0:
            return None, []

        y_tu_segments = []
        for idx in range(len(clusters[0]['y_t_segments'])):
            segment = [c['y_t_segments'][idx] for c in clusters]
            y_tu_segments.append(np.stack(segment, axis=1))

        return y_tu_segments, [c['unique_id'] for c in clusters]

    def get_all_y_tuh(self, exclude_pens=None, n_h=10, n_fut=0, **kwargs):
        all_y_ts, cluster_ids = self.get_all_y_t_segments(exclude_pens=exclude_pens, **kwargs)
        if all_y_ts == None:
            return None, []
        all_y_tuh = tensorize_segments(all_y_ts, n_h=n_h, n_fut=n_fut)

        return all_y_tuh, cluster_ids

    def get_all_y_tuh_pca(self, exclude_pens=None, n_h=10, n_fut=0, n_components=10, **kwargs):
        all_y_ts, cluster_ids = self.get_all_y_t_segments(exclude_pens=exclude_pens, **kwargs)
        all_y_ts_concat = np.concatenate(all_y_ts, axis=0)
        pca = PCA(n_components=n_components)
        all_pcs_concat = pca.fit_transform(all_y_ts_concat)
        print(all_pcs_concat.shape)
        all_pcs = split_tx(all_pcs_concat, self.stimulus.segment_lengths)
        all_y_tuh = tensorize_segments(all_pcs, n_h=n_h, n_fut=n_fut)
        return all_y_tuh, cluster_ids

    def find_missing_results(self, fieldname, noiseratio='inf', state=None, optogenetic=False):
        if not state:
            state = ['anaesthetised', 'awake']

        n_found = 0
        missing = []

        for cluster in self.clusters:
            # check whether this unit has the correct awake/anaesthetised state
            if cluster['state'] != state and cluster['state'] not in state:
                continue

            if optogenetic and not cluster['optogenetic']:
                continue

            # if a noise ratio threshold is specified, skip units that are above it
            if noiseratio not in ['inf', np.inf]:
                if np.isnan(cluster['noiseratio']) or cluster['noiseratio'] > noiseratio:
                    continue

            # otherwise, expect a result
            if fieldname in cluster:
                n_found = n_found + 1
            else:
                missing.append(cluster['unique_id'])

        return missing, n_found

    def summarize(self):
        n_units = np.zeros([2,len(AREA_ORDER),2])

        states = ['anaesthetised', 'awake']
        areas = AREA_ORDER
        noise_ratio_threshold = 200

        for c in self.clusters:
            state_idx = states.index(c['state'])
            area_idx = areas.index(c['brain_area'])
            if np.isnan(c['noiseratio']) or c['noiseratio'] > noise_ratio_threshold:
                noise_ratio_idx = 1
            else:
                noise_ratio_idx = 0

            n_units[state_idx, area_idx, noise_ratio_idx] = \
                n_units[state_idx, area_idx, noise_ratio_idx] + 1

        print('NR <= 200:')
        bits = []
        for area, n in zip(areas, n_units[0,:,0]):
            bits.append(f'{area}: {n:3.0f}')
        bits.append(f'total: {np.sum(n_units[0,:,0]):4.0f}')
        print('Anaesthetised: ' + ', '.join(bits))

        bits = []
        for area, n in zip(areas, n_units[1,:,0]):
            bits.append(f'{area}: {n:3.0f}')
        bits.append(f'total: {np.sum(n_units[1,:,0]):4.0f}')
        print('Awake:         ' + ', '.join(bits), end='\n')

        bits = []
        for area, n_anaesth, n_awake in zip(areas, n_units[0,:,0], n_units[1,:,0]):
            bits.append(f'{area}: {n_anaesth+n_awake:3.0f}')
        bits.append(f'total: {np.sum(n_units[:,:,0]):4.0f}')
        print('Total:         ' + ', '.join(bits), end='\n\n')

        print('NR > 200:')
        bits = []
        for area, n in zip(areas, n_units[0,:,1]):
            bits.append(f'{area}: {n:3.0f}')
        bits.append(f'total: {np.sum(n_units[0,:,1]):4.0f}')
        print('Anaesthetised: ' + ', '.join(bits))

        bits = []
        for area, n in zip(areas, n_units[1,:,1]):
            bits.append(f'{area}: {n:3.0f}')
        bits.append(f'total: {np.sum(n_units[1,:,1]):4.0f}')
        print('Awake:         ' + ', '.join(bits), end='\n')

        bits = []
        for area, n_anaesth, n_awake in zip(areas, n_units[0,:,1], n_units[1,:,1]):
            bits.append(f'{area}: {n_anaesth+n_awake:3.0f}')
        bits.append(f'total: {np.sum(n_units[:,:,1]):4.0f}')
        print('Total:         ' + ', '.join(bits), end='\n\n')

    def check_results_present(self, fieldname, noiseratio='inf', state=None, optogenetic=False):
        if not state:
            state = ['anaesthetised', 'awake']

        missing, n_found = self.find_missing_results(
            fieldname, noiseratio, state, optogenetic)

        if missing:
            if len(missing) > 15:
                print('** Missing more than 15 results **\n')
            else:
                print(f'** Missing {len(missing)} results: {missing} **\n')
        else:
            print(f'All results present ({n_found} clusters)\n')

    def load_analyses(self, fieldnames):
        if isinstance(fieldnames, str):
            fieldnames = [fieldnames]

        for fieldname in fieldnames:
            analysis = ANALYSES[self.stim_types_str][fieldname]

            print(f'Loading "{analysis["description"]}"')
            self.load_results(analysis['func'], analysis['params'], fieldname)
            if 'noiseratio' in analysis['params']:
                noiseratio = analysis['params']['noiseratio']
            else:
                noiseratio = 'inf'

            if 'state' in analysis['params']:
                state = analysis['params']['state']
            else:
                state = ['anaesthetised', 'awake']

            if 'optogenetic' in analysis['params']:
                optogenetic = analysis['params']['optogenetic']
            elif 'opto' in analysis['func']:
                optogenetic = True
            else:
                optogenetic = False

            self.check_results_present(fieldname, noiseratio, state, optogenetic)

    def load_coch_analyses(self, fieldnames=None):
        if not fieldnames:
            fieldnames = COCH_ANALYSES[self.stim_types_str].keys()
        self.load_analyses(fieldnames)

    def load_a2a_analyses(self, fieldnames=None):
        if not fieldnames:
            fieldnames = A2A_ANALYSES[self.stim_types_str].keys()
        self.load_analyses(fieldnames)

# named analyses

COCH_ANALYSES = {}
A2A_ANALYSES = {}
ANALYSES = {}

COCH_ANALYSES['12'] = {
    'coch_kernel_main':
        {'description': 'coch kernels; 16 folds; ElNet',
         'func': 'coch_kernel',
         'params':
            {'subset': ['valid20', 'all'],
             'n_h': 13,
             'k_folds': 16,
             'regress': 'ElNet'}},
    'coch_kernel_sigmoid_main':
        {'description': 'coch kernels; 16 folds; ElNet; LN (sigmoid)',
         'func': 'fit_sigmoids_coch',
         'params':
            {'parent_analysis_name': ['coch_kernel_main', 'coch_kernel']
            }
        },
    'coch_kernel_responses_main':
        {'description': 'coch kernels; 16 folds; ElNet; responses of linear and LN (sigmoid)',
         'func': 'get_coch_model_responses',
         'params': {
            }
        },
    'coch_kernel_responses_main_final_fold_only':
        {'description': 'coch kernels; 16 folds - 15th only; ElNet; responses of linear and LN (sigmoid)',
         'func': 'get_coch_model_responses',
         'params': {
             'final_fold_only': True
            }
        },
    'coch_kernel_hi_contrast':
        {'description': 'coch kernels; hi contrast only; 8 folds; ElNet',
         'func': 'coch_kernel',
         'params':
            {'subset': ['valid20', 'hi'],
             'n_h': 13,
             'k_folds': 8,
             'regress': 'ElNet'}},
    'coch_kernel_hi_contrast_sigmoid':
        {'description': 'coch kernels; hi contrast only LN; 8 folds; ElNet',
         'func': 'estimate_sigmoids',
         'params':
             {'result_fieldname': 'coch_kernel_hi_contrast',
              'kernel_fieldnames': 'coch_kernel',
              'X_tfh_func': 'stim_tfh_func'}
        },
    'coch_kernel_all_train':
        {'description': 'coch kernels; all data used for training (no folds); ElNet',
         'func': 'coch_kernel',
         'params':
            {'subset': ['valid20', 'all'],
             'n_h': 13,
             'k_folds': 'all_train',
             'regress': 'ElNet'}},
    'coch_kernel_nrf_main':
        {'description': 'NRF coch kernels; comparable with main coch LN kernels',
         'func': 'coch_kernel_nrf',
         'params':
            {'subset': ['valid20', 'all'],
             'n_h': 13,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'regress': 'TorchNRFCV'}},
    'best_frequency':
        {'description': 'Coch kernel BF',
         'func': 'get_best_frequency',
         'params': {}},
    # 'coch_kernel_main_lasso_subset':
    #     {'description': 'coch kernels; 16 folds; ElNetLassoSubset',
    #      'func': 'coch_kernel',
    #      'params':
    #         {'subset': ['valid20', 'all'],
    #          'n_h': 13,
    #          'k_folds': 16,
    #          'regress': 'ElNetLassoSubset'}},
    # 'coch_kernel_all_train_lasso_subset':
    #     {'description': 'coch kernels; all valid data (no folds); ElNetLassoSubset',
    #      'func': 'coch_kernel',
    #      'params':
    #         {'subset': ['valid20', 'all'],
    #          'n_h': 13,
    #          'k_folds': 'all_train',
    #          'regress': 'ElNetLassoSubset'}},
    # 'coch_kernel_hi_contrast_lasso_subset':
    #     {'description': 'coch kernels; hi contrast only; 8 folds; ElNetLassoSubset',
    #      'func': 'coch_kernel',
    #      'params':
    #         {'subset': ['valid20', 'hi'],
    #          'n_h': 13,
    #          'k_folds': 8,
    #          'regress': 'ElNetLassoSubset'}
    #     }
}

A2A_ANALYSES['12'] = {
    'a2a_kernels_main':
        {'description': 'main elnet a2a kernels (NR=200; all units; all valid data; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1,2,3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
        },
    'a2a_kernels_sigmoid_main':
        {'description': 'a2a kernels; 1 fold; ElNet; LN (sigmoid)',
         'func': 'fit_sigmoids_a2a',
         'params':
            {'parent_analysis_name': 'a2a_kernels_main'
            }
        },
    'a2a_kernels_main_extra':
        {'description': 'extra analyses (e.g. sigmoids) main elnet a2a kernels',
         'func': 'a2a_extra',
         'params':
            {'result_fieldname': 'a2a_kernels_main',
             'state': ['anaesthetised', 'awake'],
             'noiseratio': 200
            }
        },
    'a2a_kernels_anaesthetised':
        {'description': 'a2a kernels for anaesthetised only (NR=200; all valid data; elnet; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': 'anaesthetised',
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
        },
    'a2a_kernels_awake':
        {'description': 'a2a kernels for awake only (NR=200; all valid data; elnet; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': 'awake',
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
        },
    'a2a_kernels_hi_contrast':
        {'description': 'hi contrast a2a kernels (NR=200; all units; elnet; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'hi'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 8,
             'fold_idxes_to_include': [7],
             'noiseratio': 200,
             'regress': 'ElNet'}
        },
    'a2a_kernels_hi_contrast_sigmoid':
        {'description': 'hi contrast a2a kernels LN (NR=200; all units; elnet; 1 fold)',
         'func': 'estimate_sigmoids',
         'params':
             {'result_fieldname': 'a2a_kernels_hi_contrast',
              'kernel_fieldnames': ['ic', 'mgb', 'ac'],
              'X_tfh_func': 'a2a_tfh_func'}
        },
    'a2a_kernels_folds':
        {'description': 'a2a kernels, 4 of 16 folds (NR=200; elnet; all units)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [12,13,14,15],
             'noiseratio': 200,
             'regress': 'ElNet'}
        },
    'a2a_kernels_lasso_subset':
        {'description': 'lasso subset a2a kernels (NR=200; all units; all valid data; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1,2,3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNetLassoSubset'}
        },
    'a2a_residuals':
        {'description': 'a2a fitted to residuals of coch_kernel_main fits; single fold, nr<=200',
         'func': 'a2a_residual_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1,2,3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
            },
        'a2a_pca':
        {'description': 'a2a reduced dimensionality 5...160 PCs; single fold, nr<=200',
         'func': 'a2a_pca_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet',
             'n_pcs': [5,10,15,20,25,30,40,50,60,70,80,90,100,120,140,160]}
            },
        'a2a_pca_small':
        {'description': 'a2a reduced dimensionality 1...20 PCs; single fold, nr<=200',
         'func': 'a2a_pca_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet',
             'n_pcs': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,28,19,20]}
        },
    'a2a_nrf_kernels_main_old':
        {'description': 'nrf a2a kernels (NR=200; all units; all valid data; 1 fold)',
         'func': 'a2a_nrf_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'TorchNRF'}
        },
    'a2a_nrf_kernels_main':
        {'description': 'nrf a2a kernels (NR=200; all units; all valid data; 1 fold)',
         'func': 'a2a_nrf_kernels',
         'params': {
            'state': ['anaesthetised', 'awake'],
            'subset': ['valid20', 'all'],
            'combo_lens': [1,3],
            'n_h': 8,
            'n_fut': 5,
            'k_folds': 16,
            'fold_idxes_to_include': [15],
            'noiseratio': 200,
            'regress': 'TorchNRFCV'}
        },
    'm2a_kernels_ic_main':
        {'description': 'main m2a IC kernels',
         'func': 'm2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
         },
    'm2a_kernels_all_areas_main':
        {'description': 'main m2a ic_mgb_ac kernels',
         'func': 'm2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
         },
    'a2a_plus_coch_kernels_old':
        {'description': 'a2a plus cochleagram kernels - old, buggy',
         'func': 'coch_plus_a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1,2,3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
         },
    'a2a_plus_coch_kernels_new':
        {'description': 'a2a plus cochleagram kernels - new',
         'func': 'coch_plus_a2a_kernels_2',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [15],
             'noiseratio': 200,
             'regress': 'ElNet'}
         },
    # 'a2a_kernels_all_noiseratios_lassosubset':
    #     {'description': 'a2a kernels for NR<inf (all units; all valid data; lasso subset; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [1,2,3],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 16,
    #          'fold_idxes_to_include': [15],
    #          'noiseratio': 'inf',
    #          'regress': 'ElNetLassoSubset'}
    #     },
    # 'a2a_kernels_anaesthetised_lassosubset':
    #     {'description': 'a2a kernels for anaesthetised only (NR=200; all valid data; lasso subset; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': 'anaesthetised',
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [1,2,3],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 16,
    #          'fold_idxes_to_include': [15],
    #          'noiseratio': 200,
    #          'regress': 'ElNetLassoSubset'}
    #     },
    # 'a2a_kernels_awake_lassosubset':
    #     {'description': 'a2a kernels for awake only (NR=200; all valid data; lasso subset; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': 'awake',
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [1,2,3],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 16,
    #          'fold_idxes_to_include': [15],
    #          'noiseratio': 200,
    #          'regress': 'ElNetLassoSubset'}
    #     },
    # 'a2a_kernels_hi_contrast_lassosubset':
    #     {'description': 'hi contrast a2a kernels (NR=200; all units; lasso subset; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'hi'],
    #          'combo_lens': [1,2,3],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 8,
    #          'fold_idxes_to_include': [7],
    #          'noiseratio': 200,
    #          'regress': 'ElNetLassoSubset'}
    #     },
    # 'a2a_kernels_all_train_lasso_subset':
    #     {'description': 'a2a kernels, all data used for training (NR=200; all units; lasso subset; no folds)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [1,2,3],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 'all_train',
    #          'noiseratio': 200,
    #          'regress': 'ElNetLassoSubset'}
    #     },
    'a2a_kernels_folds_lassosubset':
        {'description': 'a2a kernels, 4 of 16 folds (NR=200; lasso subset; all units)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1,2,3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 16,
             'fold_idxes_to_include': [12,13,14,15],
             'noiseratio': 200,
             'regress': 'ElNetLassoSubset'}
        }
}

ANALYSES['12'] = COCH_ANALYSES['12'].copy()
ANALYSES['12'].update(A2A_ANALYSES['12'])

COCH_ANALYSES['19'] = {
    'coch_kernel_main':
        {'description': 'coch kernels; 15 folds; elnet',
         'func': 'coch_kernel',
         'params':
            {'subset': ['valid20', 'all'],
             'n_h': 13,
             'k_folds': 15,
             'regress': 'ElNet'}},
    'coch_kernel_all_train':
        {'description': 'coch kernels; all data used for training (no folds); elnet',
         'func': 'coch_kernel',
         'params':
            {'subset': ['valid20', 'all'],
             'n_h': 13,
             'k_folds': 'all_train',
             'regress': 'ElNet'}},
    'coch_kernel_opto':
        {'description': 'coch kernels light on; elnet',
         'func': 'coch_kernel_opto',
         'params':
            {'subset': ['valid20', 'all'],
             'n_h': 13,
             'k_folds': 15,
             'regress': 'ElNet'}},
    'coch_kernel_main_sigmoid':
        {'description': 'coch kernels light off; elnet, sigmoid',
         'func': 'fit_sigmoids_coch',
         'params':
            {'parent_analysis_name': ['coch_kernel_main', 'coch_kernel']}},
    'coch_kernel_opto_sigmoid':
        {'description': 'coch kernels light on; elnet, sigmoid',
         'func': 'fit_sigmoids_coch_opto',
         'params':
            {'parent_analysis_name': ['coch_kernel_opto', 'coch_kernel']}},
    'glm_fits_main':
        {'description': 'main GLM fits',
         'func': 'fit_glms_sparse_cvglmnet_all_conditions',
         'params':
            {'n_reps_train': 5,
             'n_reps_test': 2}},
    # 'coch_kernel_lassosubset':
    #     {'description': 'coch kernels; 15 folds; elnetlassosubset',
    #      'func': 'coch_kernel',
    #      'params':
    #         {'subset': ['valid20', 'all'],
    #          'n_h': 13,
    #          'k_folds': 15,
    #          'regress': 'ElNetLassoSubset'}},
}

A2A_ANALYSES['19'] = {
    'a2a_kernels_main':
        {'description': 'main a2a kernels (NR=200; all units; all valid data; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 15,
             'fold_idxes_to_include': [14],
             'noiseratio': 200,
             'regress': 'ElNet'}
        },
    'a2a_kernels_opto':
        {'description': 'a2a kernels opto (NR=200; all units; all valid data; 1 fold)',
         'func': 'a2a_kernels',
         'params':
            {'state': ['anaesthetised', 'awake'],
             'subset': ['valid20', 'all'],
             'combo_lens': [1,3],
             'n_h': 8,
             'n_fut': 5,
             'k_folds': 15,
             'fold_idxes_to_include': [14],
             'noiseratio': 200,
             'regress': 'ElNet',
             'optogenetic': True}
        },
    'a2a_kernels_opto_sigmoids':
        {'description': 'a2a kernels opto sigmoids (NR=200; all units; all valid data; 1 fold)',
         'func': 'fit_sigmoids_a2a',
         'params':
            {'parent_analysis_name':'a2a_kernels_opto',
             'optogenetic':True}
        },
    'a2a_kernels_main_sigmoids':
        {'description': 'main a2a kernels (NR=200; all units; all valid data; 1 fold)',
         'func': 'estimate_sigmoids',
         'params':
            {'result_fieldname': 'a2a_kernels_main',
             'kernel_fieldnames': ['ic', 'mgb', 'ac'],
             'X_tfh_func': 'a2a_tfh_func'}
        },
    # 'a2a_kernels_opto_new':
    #     {'description': 'a2a kernels opto (NR=200; all units; all valid data; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': ['ic', 'mgb', 'ac', ['ic', 'mgb']],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 15,
    #          'fold_idxes_to_include': [14],
    #          'noiseratio': 200,
    #          'regress': 'ElNet',
    #          'optogenetic': True}
    #     },
    # 'a2a_kernels_opto_combolen_2':
    #     {'description': 'a2a kernels opto for combo len 2 (NR=200; all units; all valid data; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [2],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 15,
    #          'fold_idxes_to_include': [14],
    #          'noiseratio': 200,
    #          'regress': 'ElNet',
    #          'optogenetic': True}
    #     },
    # 'a2a_kernels_opto_lasso_subset':
    #     {'description': 'a2a kernels opto lassosubset (NR=200; all units; all valid data; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [1],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 15,
    #          'fold_idxes_to_include': [14],
    #          'noiseratio': 200,
    #          'regress': 'ElNetLassoSubset',
    #          'optogenetic': True}
    #     },
    # 'a2a_kernels_lasso_subset':
    #     {'description': 'lasso subset a2a kernels (NR=200; all units; all valid data; 1 fold)',
    #      'func': 'a2a_kernels',
    #      'params':
    #         {'state': ['anaesthetised', 'awake'],
    #          'subset': ['valid20', 'all'],
    #          'combo_lens': [1],
    #          'n_h': 8,
    #          'n_fut': 5,
    #          'k_folds': 15,
    #          'fold_idxes_to_include': [14],
    #          'noiseratio': 200,
    #          'regress': 'ElNetLassoSubset'}
    #     },
}

ANALYSES['19'] = COCH_ANALYSES['19'].copy()
ANALYSES['19'].update(A2A_ANALYSES['19'])


def plot_3x3_opto(clusters, brain_areas, select_params,
                  x_func_off, x_func_on,
                  y_func_off, y_func_on,
                  x_label, y_label):

    fig = plt.gcf()
    axes = fig.subplots(len(brain_areas), len(brain_areas))

    for idx, brain_area in enumerate(brain_areas):
        for reg_idx, regressor_area in enumerate(brain_areas):
            sel = clusters.select_data(brain_area=brain_area, **select_params)

            x_data_off = [x_func_off(cl, regressor_area) for cl in sel]
            x_data_on = [x_func_on(cl, regressor_area) for cl in sel]
            y_data_off = [y_func_off(cl, regressor_area) for cl in sel]
            y_data_on = [y_func_on(cl, regressor_area) for cl in sel]

            plt.sca(axes[reg_idx, idx])
            scatter_cc_2(x_data_off, y_data_off,
                         x_data_on, y_data_on,
                         link_points=True,
                         color1='black', color2='red')
            plt.text(0.05,1-.05, regressor_area.upper()+'$\\rightarrow$'+brain_area.upper(),
                     fontsize=14, color='black', va='top', ha='left')

    for ax, brain_area in zip(axes[:,0], brain_areas):
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in axes[-1,:]:
        plt.sca(ax)
        plt.xlabel(x_label)

    plt.sca(axes.reshape(-1)[0])
    plt.legend(['Light off', 'Light on'])

    add_panel_labels(axes, lowercase=PANEL_LABELS_LOWER_CASE)

def scatter_cc_multi_old(x, y, xlim=[0, 1], ylim=[0, 1], markers='.+x', colors='darkblue',
               plot_central_tendency=True, central_tendency=np.median,
               central_tendency_color='darkorange', central_tendency_lines_full_width=False,
               p_value=False, p_value_font_size=14, accept_nans=True,
               ax=None):

    if ax is None:
        ax = plt.gca()

    def _format_ticks(lim, extra_values=None):
        vals = []
        if lim[0] < 0:
            vals = [lim[0], 0]
        else:
            vals = [lim[0]]
        if lim[1] > 1:
            vals.extend([lim[1], 1])
        else:
            vals.append(lim[1])
        if extra_values:
            try:
                vals.extend(extra_values)
            except:
                vals.append(extra_values)
        vals = np.sort(vals)

        labels = ['%0.2f' % v for v in vals]
        labels = ['0' if l == '0.00' else l for l in labels]
        labels = ['1' if l == '1.00' else l for l in labels]

        return vals, labels

    if matplotlib.rcParams['font.size'] < 10:
        marker_params = {'markersize': 12, 'lw': .75}
        marker_params_scatter = {'s':12, 'lw':.75}
    else:
        marker_params = {}
        marker_params_scatter = {}

    x = [np.array(d) for d in x]
    y = [np.array(d) for d in y]

    all_x = np.hstack(x)
    all_y = np.hstack(y)

    if accept_nans:
        valid_idxes = np.where(np.isfinite(all_x) & np.isfinite(all_y))[0]
        all_x = all_x[valid_idxes]
        all_y = all_y[valid_idxes]
    else:
        if np.any(np.isnan(all_x)) or np.any(np.isnan(all_y)):
            raise ValueError('NaN values present (you may want accept_nans=True)')

    for idx, (this_x, this_y) in enumerate(zip(x, y)):
        if accept_nans:
            valid_idxes = np.where(np.isfinite(this_x) & np.isfinite(this_y))[0]
            if len(valid_idxes)==0:
                continue
            this_x = this_x[valid_idxes]
            this_y = this_y[valid_idxes]

        if idx < len(markers):
            marker = markers[idx]
        else:
            marker = markers[0]
        if type(colors) in (str,tuple):
            color = colors
        elif idx < len(colors):
            color = colors[idx]
        else:
            color = colors[0]

        ax.scatter(this_x, this_y, marker=marker, color=color, **marker_params_scatter)

    ax.plot(xlim, ylim, 'k', linewidth=.75)
    ax.axis('square')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if plot_central_tendency:
        ct_x = central_tendency(all_x)
        ct_y = central_tendency(all_y)
        if central_tendency_lines_full_width:
            lim = ylim
        else:
            lim = [ylim[0], ct_y]

        ax.plot([ct_x, ct_x], lim, color=central_tendency_color, label='_nolegend_',
                **marker_params)

        if central_tendency_lines_full_width:
            lim = xlim
        else:
            lim = [xlim[0], ct_x]
        ax.plot(lim, [ct_y, ct_y], color=central_tendency_color, label='_nolegend_',
                **marker_params)
        ax.scatter(ct_x, ct_y, color=central_tendency_color, label='_nolegend_',
                **marker_params_scatter)

        vals, labels = _format_ticks(xlim, ct_x)
        ax.set_xticks(vals, labels)
        vals, labels = _format_ticks(ylim, ct_y)
        ax.set_yticks(vals, labels)

    else:
        vals, labels = _format_ticks(xlim)
        ax.set_xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        ax.set_yticks(vals, labels)

    if not central_tendency_lines_full_width:
        ax.spines[['right', 'top']].set_visible(False)

    if p_value:
        # r, p = kruskal(all_x, all_y)
        res = wilcoxon(all_x, all_y)
        p = res.pvalue
        if p>.05:
            p = 'n.s.'
        elif p<0.001:
            p = 'p<0.001'
        else:
            p = 'p='+('%0.3f' % p).lstrip('0')
        label_bottom_right(p, size=p_value_font_size)

def scatter_cc_multi_old(x, y, xlim=[0, 1], ylim=[0, 1], markers='.+x', colors='darkblue',
               plot_central_tendency=True, central_tendency=np.median,
               central_tendency_color='darkorange', central_tendency_lines_full_width=False):

    def _format_ticks(lim, extra_values=None):
        vals = []
        if lim[0] < 0:
            vals = [lim[0], 0]
        else:
            vals = [lim[0]]
        if lim[1] > 1:
            vals.extend([lim[1], 1])
        else:
            vals.append(lim[1])
        if extra_values:
            try:
                vals.extend(extra_values)
            except:
                vals.append(extra_values)
        vals = np.sort(vals)

        labels = ['%0.2f' % v for v in vals]
        labels = ['0' if l == '0.00' else l for l in labels]
        labels = ['1' if l == '1.00' else l for l in labels]

        return vals, labels

    x = [np.array(d) for d in x]
    y = [np.array(d) for d in y]

    all_x = np.hstack(x)
    all_y = np.hstack(y)

    valid_idxes = np.where(np.isfinite(all_x) & np.isfinite(all_y))[0]
    all_x = all_x[valid_idxes]
    all_y = all_y[valid_idxes]

    for idx, (this_x, this_y) in enumerate(zip(x, y)):
        valid_idxes = np.where(np.isfinite(this_x) & np.isfinite(this_y))[0]
#         print(this_x, valid_idxes)
        if len(valid_idxes)==0:
            continue
        this_x = this_x[valid_idxes]
        this_y = this_y[valid_idxes]

        if idx < len(markers):
            marker = markers[idx]
        else:
            marker = markers[0]
        if type(colors) in (str,tuple):
            color = colors
        elif idx < len(colors):
            color = colors[idx]
        else:
            color = colors[0]
        plt.scatter(this_x, this_y, marker=marker, color=color)

    plt.plot(xlim, ylim, 'k', linewidth=.75)
    plt.axis('square')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if plot_central_tendency:
        ct_x = central_tendency(all_x)
        ct_y = central_tendency(all_y)
        if central_tendency_lines_full_width:
            lim = ylim
        else:
            lim = [ylim[0], ct_y]

        plt.plot([ct_x, ct_x], lim, color=central_tendency_color, label='_nolegend_')

        if central_tendency_lines_full_width:
            lim = xlim
        else:
            lim = [xlim[0], ct_x]
        plt.plot(lim, [ct_y, ct_y], color=central_tendency_color, label='_nolegend_')
        plt.scatter(ct_x, ct_y, color=central_tendency_color, label='_nolegend_')

        vals, labels = _format_ticks(xlim, ct_x)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim, ct_y)
        plt.yticks(vals, labels)

    else:
        vals, labels = _format_ticks(xlim)
        plt.xticks(vals, labels)
        vals, labels = _format_ticks(ylim)
        plt.yticks(vals, labels)

def plot_3x3(*args, **kwargs):
    plot_3x3_by_state(*args, **kwargs, ignore_state=True)

def plot_3x3_by_state(clusters, brain_areas, select_params,
             x_func, y_func, x_label, y_label,
             box=False,
             central_tendency_lines_full_width=None,
             p_value=False, accept_nans=True,
             panel_labels=True, panel_letter_offset=0,
             ignore_state=False):

    if central_tendency_lines_full_width is not None:
        box = central_tendency_lines_full_width

    layout = (3,3)

    fig = plt.gcf()
    axes = np.array(fig.subplots(layout[0], layout[1]))

    for idx, brain_area in enumerate(brain_areas):
        for reg_idx, regressor_area in enumerate(brain_areas):
            sel = clusters.select_data(brain_area=brain_area, **select_params)

            x_data = np.array([x_func(cl, regressor_area) for cl in sel])
            y_data = np.array([y_func(cl, regressor_area) for cl in sel])
            if ignore_state:
                x_multi = [x_data]
                y_multi = [y_data]
            else:
                awake = np.where([cl['state']=='awake' for cl in sel])[0]
                anaesth = np.where([cl['state']!='awake' for cl in sel])[0]
                x_multi = [x_data[awake], x_data[anaesth]]
                y_multi = [y_data[awake], y_data[anaesth]]

            plt.sca(axes[reg_idx, idx])
            scatter_cc_multi(x_multi, y_multi,
                             colors=BRAIN_AREA_COLORS[brain_area],
                             box=box,
                             p_value=p_value, accept_nans=accept_nans)
            plt.text(0.05,1-.05,
                     AREA_LABELS[regressor_area].upper()+'$\\rightarrow$' + \
                     AREA_LABELS[brain_area].upper(),
                     fontsize=matplotlib.rcParams['font.size']+2,
                     color='black', va='top', ha='left')

    for ax, brain_area in zip(axes[:,0], brain_areas):
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in axes[-1,:]:
        plt.sca(ax)
        plt.xlabel(x_label)

    if panel_labels:
        add_panel_labels(axes, letter_offset=panel_letter_offset,
            lowercase=PANEL_LABELS_LOWER_CASE)

def plot_3x3_summary(clusters, brain_areas, select_params,
             x_func, y_func, cmap_max=1, x_label=True, y_label=True):

    summary_figure = np.zeros((3,3))
    for idx, brain_area in enumerate(brain_areas):
        for idx2, regressor_area in enumerate(brain_areas):

            sel = clusters.select_data(brain_area=brain_area, **select_params)
            coch = np.array([x_func(c, regressor_area) for c in sel])
            a2a = np.array([y_func(c, regressor_area) for c in sel])
            d = np.nanmedian(a2a) - np.nanmedian(coch) #, nan_policy='omit')
            summary_figure[idx, idx2] = d

    print(np.max(np.abs(summary_figure)))
    plt.imshow(summary_figure,
               vmin=-cmap_max, vmax=cmap_max, cmap='bwr', origin='lower')
    plt.xticks([0,1,2], labels=[AREA_LABELS[a] for a in brain_areas])
    plt.yticks([0,1,2], labels=[AREA_LABELS[a] for a in brain_areas])
    if x_label:
        plt.xlabel('Source population')
    if y_label:
        plt.ylabel('Target population')

def plot_by_brain_area(clusters, brain_areas, select_params,
                       x_func, y_func, x_label, y_label,
                       box=False,
                       central_tendency_lines_full_width=None,
                       p_value=False, accept_nans=True,
                       panel_labels=True,
                       panel_letter_offset=0,
                       xlim=[0, 1], ylim=[0, 1]):

    fig = plt.gcf()

    if central_tendency_lines_full_width is not None:
        box = central_tendency_lines_full_width

    if len(brain_areas)==2:
        layout = (1,2)
    if len(brain_areas)==3:
        layout = (1,3)
    elif len(brain_areas)==4:
        layout = (2,2)

    axes = np.array(fig.subplots(layout[0], layout[1]))

    for idx, brain_area in enumerate(brain_areas):
        sel = clusters.select_data(brain_area=brain_area, **select_params)

        x_data = [x_func(cl) for cl in sel]
        y_data = [y_func(cl) for cl in sel]
        plt.sca(axes.reshape(-1)[idx])
        scatter_cc(x_data, y_data, color=BRAIN_AREA_COLORS[brain_area],
                   box=box,
                   p_value=p_value, accept_nans=accept_nans,
                   xlim=xlim, ylim=ylim)

        plt.text(xlim[0]+(xlim[1]-xlim[0])*0.05, # 0.05 ,
                 ylim[0]+(xlim[1]-xlim[0])*(1-0.05), # 1-.05,
                 AREA_LABELS[brain_area],
                 fontsize=matplotlib.rcParams['font.size']+2,
                 color='black', va='top', ha='left')

    if len(axes.shape) == 1:
        lhs = [axes[0]]
        bottom = axes
    else:
        lhs = axes[:,0]
        bottom = axes[-1, :]

    for ax in lhs:
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in bottom:
        plt.sca(ax)
        plt.xlabel(x_label)

    if panel_labels:
        add_panel_labels(axes, letter_offset=panel_letter_offset,
            lowercase=PANEL_LABELS_LOWER_CASE)

def plot_2_by_brain_area(clusters, brain_areas, select_params,
                       x_func1, y_func1, x_func2, y_func2,
                       x_label, y_label,
                       markers='.+x',
                       box=False,
                       central_tendency=np.median,
                       plot_central_tendency=True,
                       plot_both_central_tendencies=False,
                       central_tendency_lines_full_width=None,
                       panel_labels=True,
                       panel_letter_offset=0,
                       xlim=[0, 1], ylim=[0, 1]):

    # if central_tendency_lines_full_width is not None:
    #     box = central_tendency_lines_full_width

    if len(brain_areas)==2:
        layout = (1,2)
    if len(brain_areas)==3:
        layout = (1,3)
    elif len(brain_areas)==4:
        layout = (2,2)

    fig = plt.gcf()
    axes = np.array(fig.subplots(layout[0], layout[1]))

    for idx, brain_area in enumerate(brain_areas):
        sel = clusters.select_data(brain_area=brain_area, **select_params)

        x_data1 = [x_func1(cl) for cl in sel]
        y_data1 = [y_func1(cl) for cl in sel]
        x_data2 = [x_func2(cl) for cl in sel]
        y_data2 = [y_func2(cl) for cl in sel]
        plt.sca(axes.reshape(-1)[idx])
        scatter_cc_2(x_data1, y_data1, x_data2, y_data2,
                     color1=BRAIN_AREA_COLORS[brain_area],
                     color2=BRAIN_AREA_COLORS[brain_area],
                     markers=markers,
                     plot_central_tendency=plot_central_tendency,
                     box=box,
                     xlim=xlim, ylim=ylim)

        if plot_both_central_tendencies:
            mx_1 = central_tendency(x_data1)
            my_1 = central_tendency(y_data1)
            mx_2 = central_tendency(x_data2)
            my_2 = central_tendency(y_data2)
            print(mx_1)
            plt.plot(mx_1, my_1, '.', c='k', label='_nolegend_')
            plt.plot(mx_2, my_2, '+', c='k', label='_nolegend_')

        plt.text(xlim[0]+(xlim[1]-xlim[0])*0.05, # 0.05 ,
                 ylim[0]+(xlim[1]-xlim[0])*(1-0.05), # 1-.05,
                 AREA_LABELS[brain_area],
                 fontsize=matplotlib.rcParams['font.size']+2,
                 color='black', va='top', ha='left')

    if len(axes.shape) == 1:
        lhs = [axes[0]]
        bottom = axes
    else:
        lhs = axes[:,0]
        bottom = axes[-1, :]

    for ax in lhs:
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in bottom:
        plt.sca(ax)
        plt.xlabel(x_label)

    if panel_labels:
        add_panel_labels(axes, letter_offset=panel_letter_offset,
            lowercase=PANEL_LABELS_LOWER_CASE)

def plot_by_brain_area_by_state(clusters, brain_areas, select_params,
                                x_func, y_func, x_label, y_label,
                                central_tendency_lines_full_width=False,
                                p_value=False, accept_nans=True,
                                panel_labels=True,
                                panel_letter_offset=0):

    fig = plt.gcf()

    if central_tendency_lines_full_width is not None:
        box = central_tendency_lines_full_width

    if len(brain_areas)==2:
        layout = (1,2)
    elif len(brain_areas)==3:
        layout = (1,3)
    elif len(brain_areas)==4:
        layout = (2,2)

    # fig = plt.figure(figsize=(4*layout[1],4*layout[0]))

    axes = np.array(fig.subplots(layout[0], layout[1]))

    for idx, brain_area in enumerate(brain_areas):
        sel = clusters.select_data(brain_area=brain_area, **select_params)

        x_data = np.array([x_func(cl) for cl in sel])
        y_data = np.array([y_func(cl) for cl in sel])
        plt.sca(axes.reshape(-1)[idx])
        awake = np.where([cl['state']=='awake' for cl in sel])[0]
        anaesth = np.where([cl['state']!='awake' for cl in sel])[0]
        scatter_cc_multi([x_data[awake], x_data[anaesth]], [y_data[awake], y_data[anaesth]],
                         colors=BRAIN_AREA_COLORS[brain_area],
                         box=box,
                         p_value=p_value,
                         accept_nans=accept_nans)

        plt.text(0.05,1-.05, AREA_LABELS[brain_area],
                 fontsize=matplotlib.rcParams['font.size']+2,
                 color='black', va='top', ha='left')

    if len(axes.shape) == 1:
        lhs = [axes[0]]
        bottom = axes
    else:
        lhs = axes[:,0]
        bottom = axes[-1, :]

    for ax in lhs:
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in bottom:
        plt.sca(ax)
        plt.xlabel(x_label)

    if panel_labels:
        add_panel_labels(axes, letter_offset=panel_letter_offset,
            lowercase=PANEL_LABELS_LOWER_CASE)

def plot_hist_opto_by_brain_area(clusters, brain_areas, select_params,
                       y_func_off, y_func_on, x_label, y_label,
                       contrast_ratio=True):
    # older, may need updating
    if len(brain_areas)==3:
        layout = (1,3)
    elif len(brain_areas)==4:
        layout = (2,2)

    [fig, axes] = plt.subplots(layout[0], layout[1],
                               figsize=(4*layout[1],4*layout[0]))

    for idx, brain_area in enumerate(brain_areas):
        sel = clusters.select_data(brain_area=brain_area, **select_params)

        y_data_off = [y_func_off(cl) for cl in sel]
        y_data_on = [y_func_on(cl) for cl in sel]

        plt.sca(axes.reshape(-1)[idx])

        if contrast_ratio:
            y_diff = np.array([(y_on-y_off)/(y_on+y_off)/2 for y_on, y_off in zip(y_data_on, y_data_off)])
        else:
            y_diff = np.array([(y_on-y_off) for y_on, y_off in zip(y_data_on, y_data_off)])
        med = np.nanmedian(y_diff)

        plt.hist(y_diff, bins=np.arange(-.95, .95, .1),
                 color=BRAIN_AREA_COLORS[brain_area],
                 linewidth=1, edgecolor=([a/1.2 for a in BRAIN_AREA_COLORS[brain_area]]))
        plt.xlim([-1, 1])
        y_lim = plt.gca().get_ylim()
        plt.plot([0, 0], plt.gca().get_ybound(), color='black', linewidth=0.75)
        plt.plot([med, med], plt.gca().get_ybound(), 'k', linestyle='--')

        y_lim = plt.gca().set_ylim(y_lim)

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.text(xlim[0] + (xlim[1]-xlim[0])*0.05,
                 ylim[0] + (ylim[1]-ylim[0])* (1-.05),
                 AREA_LABELS[brain_area],
                 fontsize=14, color='black', va='top', ha='left')

        stat = stats.wilcoxon(y_diff, nan_policy='omit')
        sig = ''
        if stat.pvalue<0.05:
            sig = '*'
        if stat.pvalue<0.01:
            sig = '**'
        if stat.pvalue<0.001:
            sig = '**'

        plt.text(xlim[0] + (xlim[1]-xlim[0])*0.95,
                 ylim[0] + (ylim[1]-ylim[0])* (1-.05),
                 sig,
                 fontsize=18, color='black', va='top', ha='right')

    if len(axes.shape) == 1:
        lhs = [axes[0]]
        bottom = axes
    else:
        lhs = axes[:,0]
        bottom = axes[-1, :]

    for ax in lhs:
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in bottom:
        plt.sca(ax)
        plt.xlabel(x_label)

    add_panel_labels(axes, lowercase=PANEL_LABELS_LOWER_CASE)

from benlib.strf import show_strf
from benlib.plot import label_top_left, label_top_right

def show_coch_kernel(k_fh, ax=None):
    if ax is None:
        ax = plt.gca()
    show_strf(k_fh, ax=ax)
    ylim = ax.get_ylim()
    ax.set_yticks([ylim[0], ylim[1]], labels=['1k', '64k'])

    xlim = ax.get_xlim()
    ax.set_xticks([xlim[0], xlim[1]], labels=['-300', '0'])
    ax.set_xlabel('History (ms)')
    ax.set_ylabel('Frequency (Hz)')

def show_a2a_kernel_sorted(k_fh_full, n_to_keep=40, ax=None):
    # show single a2a kernel in the style of BNA poster

    if ax is None:
        ax = plt.gca()

    sm = np.sum(k_fh_full, axis=1)
    idx = np.argsort(sm)
    k_fh_sorted = k_fh_full[idx]
    k_fh = np.vstack((k_fh_sorted[:n_to_keep], k_fh_sorted[-n_to_keep:]))
    show_strf(k_fh, ax=ax)
    ax.axhline(n_to_keep, color='k', linewidth=1)
    ax.axvline(7.5, color='k', linewidth=1)
    ylim = ax.get_ylim()
    ax.set_yticks([ylim[0], n_to_keep, ylim[1]], labels=['$-$', '0', '$+$'])

    xlim = ax.get_xlim()
    ax.set_xticks([xlim[0], 7.5, xlim[1]], labels=['-200', '0', '125'])
    ax.set_xlabel('History (ms)')

def show_coch_and_a2a_kernels(unit, n_rows=1, this_row=1, axes=None,
                              show_ylabels=False):
    # show cochleagram and single-area A2A kernels in the style of
    # the BNA poster
    this_row = this_row - 1

    # if axes is None:
    #     fig = plt.figure(figsize=(12,4))

    if axes is None:
        axes = []
        axes.append(plt.subplot(n_rows,4,this_row*4+1))
        axes.append(plt.subplot(n_rows,4,this_row*4+2))
        axes.append(plt.subplot(n_rows,4,this_row*4+3))
        axes.append(plt.subplot(n_rows,4,this_row*4+4))

    show_coch_kernel(unit['coch_kernel_main']['coch_kernel'][-1]['k_fh'], ax=axes[0])
    cc_norm_coch = unit['coch_kernel_sigmoid_main']['sigmoid_fits'][-1]['sigmoid']['cc_norm_test'][0]
    label_top_left('$CC_{norm} = %0.2f$' % cc_norm_coch, margin=.015, outside=True, ax=axes[0],
        size=matplotlib.rcParams['font.size']+2)

    show_a2a_kernel_sorted(unit['a2a_kernels_main']['ic'][-1]['k_fh'], ax=axes[1])
    if show_ylabels:
        axes[1].set_ylabel('IC unit weighting')
    cc_norm_ic = unit['a2a_kernels_sigmoid_main']['sigmoid_fits']['ic'][0]['sigmoid']['cc_norm_test'][0]
    label_top_left('%s %0.2f' % (AREA_LABELS['ic'], cc_norm_ic), margin=.015, outside=True, ax=axes[1],
        size=matplotlib.rcParams['font.size']+2)

    show_a2a_kernel_sorted(unit['a2a_kernels_main']['mgb'][-1]['k_fh'], ax=axes[2])
    if show_ylabels:
        axes[2].set_ylabel('MGB unit weighting')
    cc_norm_mgb = unit['a2a_kernels_sigmoid_main']['sigmoid_fits']['mgb'][0]['sigmoid']['cc_norm_test'][0]
    label_top_left('%s %0.2f' % (AREA_LABELS['mgb'], cc_norm_mgb), margin=.015, outside=True, ax=axes[2],
        size=matplotlib.rcParams['font.size']+2)

    show_a2a_kernel_sorted(unit['a2a_kernels_main']['ac'][-1]['k_fh'], ax=axes[3])
    if show_ylabels:
        axes[3].set_ylabel('AC unit weighting')
    cc_norm_ac = unit['a2a_kernels_sigmoid_main']['sigmoid_fits']['ac'][0]['sigmoid']['cc_norm_test'][0]
    label_top_left('%s %0.2f' % (AREA_LABELS['ac'], cc_norm_ac), margin=.015, outside=True, ax=axes[3],
        size=matplotlib.rcParams['font.size']+2)

    # print unit ID outside axis
    xlim = axes[3].get_xlim()
    ylim = axes[3].get_ylim()
    axes[3].text(xlim[0]+(xlim[1]-xlim[0])*1.02, 0,
             unit['unique_id'] + ' NR=%0.2f' % unit['noiseratio'], rotation=90, ha='left', va='bottom',
             size=matplotlib.rcParams['font.size']-2)

def hist_by_brain_area(clusters, brain_areas, select_params,
                       bins,
                       val_func, x_label=None,
                       plot_central_tendency=True,
                       central_tendency_alignment='right',
                       central_tendency_margin=.02,
                       show_significance=True,
                       label_in_top_right=False,
                       box=False):

    fig = plt.gcf()
    axes = np.array(fig.subplots(len(brain_areas), 1))

    for idx, brain_area in enumerate(brain_areas):
        ax = axes[idx]
        plt.sca(ax)
        # plt.subplot(len(brain_areas), 1, idx+1)
        sel = clusters.select_data(brain_area=brain_area, **select_params)
        vals = [val_func(c) for c in sel]
        vals = [v for v in vals if np.isfinite(v)]
#         print(vals)
        plt.hist(vals, bins, color=BRAIN_AREA_COLORS[brain_area])
        plt.axvline(0, c='k', lw=matplotlib.rcParams['axes.linewidth'])
        if idx<len(brain_areas)-1:
            plt.gca().set_xticklabels([])
        if plot_central_tendency:
            med = np.median(vals)
            plt.axvline(med, c='k', lw=1.5, ls='--')
            ylim = plt.ylim()
            margin = 0.03
            ypos = ylim[0] + (ylim[1]-ylim[0])*(1-margin)
            if central_tendency_alignment=='right':
                sgn = -1
            else:
                sgn = 1

            txt = '%0.2f' % med
            stat_str = ''
            if show_significance:
                res = wilcoxon(vals)
                if res.pvalue <.001:
                    stat_str = '***'
                elif res.pvalue<0.01:
                    stat_str = '**'
                elif res.pvalue<.05:
                    stat_str = '*'
            if stat_str != '':
                txt = txt + ' ' + stat_str
            plt.text(med+sgn*central_tendency_margin, ypos, txt,
                     ha=central_tendency_alignment,
                     va='top', size=matplotlib.rcParams['font.size']+2)

        if label_in_top_right:
            label_top_right(AREA_LABELS[brain_area].upper(), size=matplotlib.rcParams['font.size']+2)
        else:
            label_top_left(AREA_LABELS[brain_area].upper(), size=matplotlib.rcParams['font.size']+2)
        if not box:
            ax.spines[['right', 'top']].set_visible(False)

    plt.xlabel(x_label)

def plot_3x3_old(clusters, brain_areas, select_params,
             x_func, y_func, x_label, y_label,
             panel_letter_offset=None):

    fig = plt.gcf()
    axes = fig.subplots(len(brain_areas), len(brain_areas),
                               figsize=(12,12))

    for idx, brain_area in enumerate(brain_areas):
        for reg_idx, regressor_area in enumerate(brain_areas):
            sel = clusters.select_data(brain_area=brain_area, **select_params)

            x_data = [x_func(cl, regressor_area) for cl in sel]
            y_data = [y_func(cl, regressor_area) for cl in sel]
            plt.sca(axes[reg_idx, idx])
            scatter_cc(x_data, y_data, color=BRAIN_AREA_COLORS[brain_area],
                       central_tendency_color='grey')
            plt.text(0.05,1-.05,
                     regressor_area.upper()+'$\\rightarrow$'+brain_area.upper(),
                     fontsize=matplotlib.rcParams['font.size']+2,
                     color='black', va='top', ha='left')

    for ax, brain_area in zip(axes[:,0], brain_areas):
        plt.sca(ax)
        plt.ylabel(y_label)

    for ax in axes[-1,:]:
        plt.sca(ax)
        plt.xlabel(x_label)

    if panel_letter_offset is not None:
        add_panel_labels(axes, panel_letter_offset=panel_letter_offset,
            lowercase=PANEL_LABELS_LOWER_CASE)

def set_publication_figure_defaults():
    # matplotlib.rcParams.keys()

    matplotlib.rcParams['font.size'] = 7
    matplotlib.rcParams['axes.linewidth'] = 0.5
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['ytick.major.width'] = 0.5
    matplotlib.rcParams['lines.linewidth'] = 0.75

    # make figures larger in notebook
    plt.rcParams['figure.dpi'] = 72*2 # 3 to zoom in