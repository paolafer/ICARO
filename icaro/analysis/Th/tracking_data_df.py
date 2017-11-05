import os
import sys
import random
import tables as tb
import numpy  as np
import pandas  as pd

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from   invisible_cities.database               import load_db

import invisible_cities.reco.paolina_functions as plf
from   invisible_cities.reco.dst_functions     import load_xy_corrections

from   invisible_cities.io.dst_io              import load_dst
from   invisible_cities.io.hits_io             import load_hits
from   invisible_cities.io.hits_io             import load_hits_skipping_NN

from   invisible_cities.types.ic_types         import xy
from   invisible_cities.types.ic_types         import NN

from   invisible_cities.evm.event_model        import Cluster, Hit

import invisible_cities.core.fit_functions     as fitf

from   invisible_cities.icaro.hst_functions    import hist
from   invisible_cities.icaro.hst_functions    import gausstext
import invisible_cities.reco.dst_functions     as dstf



def merge_NN_hits(hits_all,hits_nonNN):

    # Iterate through the nonNN dictionary and update the energies including the NN hits from the "all" dictionary.
    for (evt,hc) in hits_nonNN.items():
        # Get the corresponding collection of all hits.
        hc_all = hits_all[evt]           
        # Add energy from all NN hits to hits in closest slice.
        for h1 in hc_all.hits:
            if(h1.Q == NN):
                # Find the hits to which the energy will be added.
                zdist_min = -1
                h_add = []
                for h2 in hc.hits:
                    zdist = np.abs(h1.Z - h2.Z)
                    if(zdist_min < 0 or zdist < zdist_min):
                        zdist_min = zdist
                        h_add = []
                        h_add.append(h2)
                    elif(zdist == zdist_min):
                        h_add.append(h2)
                # Add the energy.
                hadd_etot = sum([ha.E for ha in h_add])
                for ha in h_add:
                    ha.energy += h1.E*(ha.E/hadd_etot)


corrections   = "/home/paolafer/data/corrections/corrections_run4734.h5"
LTcorrection = dstf.load_lifetime_xy_corrections(corrections,
                                                 group="XYcorrections",
                                                 node="Lifetime")
XYcorrection  = dstf.load_xy_corrections(corrections,
                                    group = "XYcorrections",
                                    node = f"Geometry_5.0mm",
                                    norm_strategy = "index",
                                    norm_opts = {"index": (40,40)})


vox_size = np.array([15,15,15],dtype=np.int16)    # voxel size
blob_radius = 21.                    # blob radius in mm

start = int(sys.argv[1])
numb = int(sys.argv[2])

lt0 = 1775.891 # lifetime at the time of the run of the correction map being used
lt = 1820.22 # lifetime at the time of the run being analyized

event = []
minZ = []
maxZ = []
maxR = []
evt_energy = []
numb_of_tracks = []
track = []
energy = []
length = []
numb_of_voxels = []
eblob1 = []
eblob2 = []


for n in range(start,start+numb):
    nstring = ''
    if n < 10:
        nstring = '00{0}'.format(n)
    elif n < 100:
        nstring = '0{0}'.format(n)
    else:
        nstring = n
        
    hits_file = '/home/jrenner/analysis/4735/hdf5/tracks_save/tracks_rebin2_qlm35/tracks_{0}_4735_icdev_20171006_th_th2000.h5.h5'.format(nstring)
    if not os.path.isfile(hits_file):
        continue

    print('Analyzing {0}'.format(hits_file))
    rhits = load_hits(hits_file)
    good_hits = load_hits_skipping_NN(hits_file)
    merge_NN_hits(rhits,good_hits)

    hits_evt = {}

    for ee, hc in good_hits.items():
        hc_corr = []                            
        for hh in hc.hits:        
            hecorr = hh.E * LTcorrection(hh.Z, hh.X, hh.Y).value[0]**(lt0/lt)  * XYcorrection(hh.X, hh.Y).value[0]
            hcorr = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0),hh.Z,hecorr)
            hc_corr.append(hcorr)
        
        hits_evt[ee] = hc_corr


    for nevt, hitc in hits_evt.items():

        vmaxR = 0.
        vmaxZ = 0.
        vminZ = 1e+06
        for hh in hitc: 
            rad_pos = np.sqrt(hh.X**2 +  hh.Y**2)
            if rad_pos > vmaxR:
                vmaxR = rad_pos
            if hh.Z > vmaxZ:
                vmaxZ = hh.Z
            if hh.Z < vminZ:
                vminZ = hh.Z

        tot_e = sum([hh.E for hh in hitc])
        
        voxels = plf.voxelize_hits(hitc, vox_size)
        trks = plf.make_track_graphs(voxels, vox_size)
    
        track_id = 0
        for t in trks:
            if (len(t.nodes()) < 1):
                etrk = 0
            else:
                etrk = sum([vox.E for vox in t.nodes()])

            E_blob1, E_blob2 = plf.blob_energies(t,blob_radius)
       
            if (E_blob2 > E_blob1):
                eswap = E_blob1
                E_blob1 = E_blob2
                E_blob2 = eswap

            event = event + [nevt]
            maxR = maxR + [vmaxR]
            minZ = minZ + [vminZ]
            maxZ = maxZ + [vmaxZ]
            evt_energy = evt_energy +[tot_e]
            numb_of_tracks = numb_of_tracks + [len(trks)]
            track = track + [track_id]
            energy = energy + [etrk]
            numb_of_voxels = numb_of_voxels + [len(t.nodes())]
            length = length + [plf.length(t)]
            eblob1 = eblob1 + [E_blob1]
            eblob2 = eblob2 + [E_blob2]

            track_id += 1


df = pd.DataFrame({'event': event, 'minZ': minZ, 'maxZ': maxZ, 'maxR': maxR, 'evt_energy': evt_energy,
                    'numb_of_tracks': numb_of_tracks, 'track': track, 'energy': energy, 'length': length,
                    'numb_of_voxels': numb_of_voxels, 'eblob1': eblob1, 'eblob2': eblob2})

out_name = '/home/data/real_data/4735/rebin2_qlm35/tracking_r4735_{0}_{1}.hdf5'.format(start, numb)
store = pd.HDFStore(out_name, "w", complib=str("zlib"), complevel=4)
store.put('dataframe', df, format='table', data_columns=True)
store.close()





