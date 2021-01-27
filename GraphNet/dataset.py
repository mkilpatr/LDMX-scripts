from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import tqdm
import uproot
import awkward
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(12)

MAX_NUM_ECAL_HITS = 50

def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array()
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)


def _pad(a, pad_value=0):
    return a.pad(MAX_NUM_ECAL_HITS, clip=True).fillna(0).regular()

## Higgs to tautau baseline cut ###
#"Pass_NJets30 && SVFitMET_isValid && HiggsSVFit_PassBaseline && HiggsSVFit_PassLepton";
###" && Pass_EventFilter && Pass_JetID 
###"(HiggsSVFit_channel == 5 && HiggsSVFit_DZeta > -35 && HiggsSVFit_elecMuonMT < 60)";
###"(HiggsSVFit_channel == 0 && HiggsSVFit_PassTight && HiggsSVFit_tau1_muMT < 50) || (HiggsSVFit_channel == 1 && HiggsSVFit_PassTight && HiggsSVFit_tau1_elecMT < 50)";
###"(HiggsSVFit_channel == 2 && HiggsSVFit_PassTight && HiggsSVFit_ditauDR > 0.5 && HiggsSVFit_ditauPt > 50)";
###

class ECalHitsDataset(Dataset):

    def __init__(self, siglist, bkglist, load_range=(0, 1), apply_preselection=True, ignore_evt_limits=False, obs_branches=[], coord_ref=None, detector_version='v9'):
        super(ECalHitsDataset, self).__init__()
        # first load baseline cut
        self._passNJets_branch = 'Pass_NJets30'
        self._passIsValid_branch = 'SVFitMET_isValid'
        self._passBaseline_branch = 'HiggsSVFit_PassBaseline'
        self._passLepton_branch = 'HiggsSVFit_PassLepton'
        self._passEventFilter_branch = 'Pass_EventFilter'
        self._passJetID_branch = 'Pass_JetID'
        self._isChannel_branch = 'HiggsSVFit_channel'
        self._DZeta_branch = 'HiggsSVFit_DZeta'
        self._elecMuonMT_branch = 'HiggsSVFit_elecMuonMT'
        self._passTight_branch = 'HiggsSVFit_PassTight'
        self._passMuMT_branch = 'HiggsSVFit_tau1_muMT'
        self._passElecMT_branch = 'HiggsSVFit_tau1_elecMT'
        self._ditauDR_branch = 'HiggsSVFit_ditauDR'
        self._ditauPt_branch = 'HiggsSVFit_ditauPt'

        self._branches = [self._passNJets_branch, self._passIsValid_branch, self._passBaseline_branch, self._passLepton_branch, self._passEventFilter_branch, self._passJetID_branch, self._isChannel_branch, self._DZeta_branch, self._elecMuonMT_branch, self._passTight_branch, self._passMuMT_branch, self._passElecMT_branch, self._ditauDR_branch, self._ditauPt_branch]

        self.extra_labels = []
        self.presel_eff = {}
        self.var_data = {}
        self.obs_data = {k:[] for k in obs_branches}

        def _read_file(table):
            # load data from one file
            start, stop = [int(x * len(table[self._branches[0]])) for x in load_range]
            for k in table:
                table[k] = table[k][start:stop]
            n_inclusive = len(table[self._branches[0]])  # before preselection

            if apply_preselection:
                pass_basesel = table[self._passNJets_branch] * table[self._passIsValid_branch] * table[self._passBaseline_branch] * table[self._passLepton_branch] * table[self._passEventFilter_branch] * table[self._passJetID_branch]
                pass_emusel  = (table[self._isChannel_branch] == 5) * (table[self._DZeta_branch] > -35) * (table[self._elecMuonMT_branch] < 60)
                pass_lephadsel = ((table[HiggsSVFit_channel] == 0) * table[self._passTight_branch] * (table[self._passMuMT_branch] < 50)) + ((table[HiggsSVFit_channel] == 1) * table[self._passTight_branch] * (table[self._passElecMT_branch] < 50))
                pass_hadhadsel = (table[HiggsSVFit_channel] == 2) * table[self._passTight_branch] * (table[self._ditauDR_branch] > 0.5) * (table[self._ditauPt_branch] > 50)
                pos_pass_presel = pass_basesel * (pass_emusel + pass_lephadsel + pass_hadhadsel)
                for k in table:
                    table[k] = table[k][pos_pass_presel]
            n_selected = len(table[self._branches[0]])  # after preselection

            for k in table:
                if isinstance(table[k], awkward.array.objects.ObjectArray):
                    table[k] = awkward.JaggedArray.fromiter(table[k]).flatten()

            eid = table[self._id_branch]
            energy = table[self._energy_branch]
            pos = (energy > 0)
            eid = eid[pos]
            energy = energy[pos]

            var_dict = {}
            obs_dict = {k: table[k] for k in obs_branches}

            return (n_inclusive, n_selected), var_dict, obs_dict

        def _load_dataset(filelist, name):
            # load data from all files in the siglist or bkglist
            n_sum = 0
            for extra_label in filelist:
                filepath, max_event = filelist[extra_label]
                if len(glob.glob(filepath)) == 0:
                    print('No matches for filepath %s: %s, skipping...' % (extra_label, filepath))
                    return
                if ignore_evt_limits:
                    max_event = -1
                n_total_inclusive = 0
                n_total_selected = 0
                var_dict = {}
                obs_dict = {k:[] for k in obs_branches}
                print('Start loading dataset %s (%s)' % (filepath, name))

                with tqdm.tqdm(glob.glob(filepath)) as tq:
                    for fp in tq:
                        t = uproot.open(fp)['Events']
                        if len(t.keys()) == 0:
#                             print('... ignoring empty file %s' % fp)
                            continue
                        load_branches = [k for k in self._branches + obs_branches if '.' in k and k[-1] == '_']
                        table = t.arrays(load_branches, namedecode='utf-8', executor=executor)
                        (n_inc, n_sel), v_d, o_d = _read_file(table)
                        n_total_inclusive += n_inc
                        n_total_selected += n_sel
                        for k in v_d:
                            if k in var_dict:
                                var_dict[k].append(v_d[k])
                            else:
                                var_dict[k] = [v_d[k]]
                        for k in obs_dict:
                            obs_dict[k].append(o_d[k])
                        if max_event > 0 and n_total_selected >= max_event:
                            break

                # calc preselection eff before dropping events more than `max_event`
                self.presel_eff[extra_label] = float(n_total_selected) / n_total_inclusive
                # now we concat the arrays and remove the extra events if needed
                n_total_loaded = None
                upper = None
                if max_event > 0 and max_event < n_total_selected:
                    upper = max_event - n_total_selected
                for k in var_dict:
                    var_dict[k] = _concat(var_dict[k])[:upper]
                    if n_total_loaded is None:
                        n_total_loaded = len(var_dict[k])
                    else:
                        assert(n_total_loaded == len(var_dict[k]))
                for k in obs_dict:
                    obs_dict[k] = _concat(obs_dict[k])[:upper]
                    assert(n_total_loaded == len(obs_dict[k]))
                print('Total %d events, selected %d events, finally loaded %d events.' % (n_total_inclusive, n_total_selected, n_total_loaded))

                self.extra_labels.append(extra_label * np.ones(n_total_loaded, dtype='int32'))
                for k in var_dict:
                    if k in self.var_data:
                        self.var_data[k].append(var_dict[k])
                    else:
                        self.var_data[k] = [var_dict[k]]
                for k in obs_branches:
                    self.obs_data[k].append(obs_dict[k])
                n_sum += n_total_loaded
            return n_sum

        nsig = _load_dataset(siglist, 'sig')
        nbkg = _load_dataset(bkglist, 'bkg')
        # label for training
        self.label = np.zeros(nsig + nbkg, dtype='float32')
        self.label[:nsig] = 1

        self.extra_labels = np.concatenate(self.extra_labels)
        for k in self.var_data:
            self.var_data[k] = _concat(self.var_data[k])
        for k in obs_branches:
            self.obs_data[k] = _concat(self.obs_data[k])

        # training features
        xyz = [_pad(a) for a in (self.var_data['x'], self.var_data['y'], self.var_data['z'])]
        layer_id = _pad(self.var_data['layer_id'])
        log_e = _pad(np.log(self.var_data[self._energy_branch]))
        self.coordinates = np.stack(xyz, axis=1).astype('float32')
        self.features = np.stack(xyz + [layer_id, log_e], axis=1).astype('float32')

        assert(len(self.coordinates) == len(self.label))
        assert(len(self.features) == len(self.label))

    @property
    def num_features(self):
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        pts = self.coordinates[i]
        fts = self.features[i]
        y = self.label[i]
        return pts, fts, y


class _SimpleCustomBatch:

    def __init__(self, data, min_nodes=None):
        pts, fts, labels = list(zip(*data))
        self.coordinates = torch.tensor(pts)
        self.features = torch.tensor(fts)
        self.label = torch.tensor(labels)

    def pin_memory(self):
        self.coordinates = self.coordinates.pin_memory()
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch):
    return _SimpleCustomBatch(batch)
