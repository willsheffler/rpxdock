#!/path/to/your/python (eg. net/software/rpxdock/env/bin/python)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:25:26 2019
@author: quintond, yhsia
Edited 2022-08-29
"""

import sys
sys.path.append('/path/to/your/rpxdock/') #(eg. /net/software/rpxdock/rpxdock/)

import os
import pickle
import rpxdock
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Manage script input')
parser.add_argument("-i", "--input", type=str, help="Output file from RPXDock")
parser.add_argument("-o", "--outpath", type=str, default="./", help="Path to dump pdbs")
parser.add_argument("-n", "--nout", type=int, default=10, help="Number of pdbs to dump")
parser.add_argument("-m", "--model", type=str, help="Dump a specific model instead of topX or bottomX")
parser.add_argument("-f", "--fname", type=str, help="Specific filename to use to dump") 
parser.add_argument("-a", "--asu", default=False, action='store_true', help="Dumps ASU")
parser.add_argument("-c", "--closest", default=False, action='store_true', help="Dumps ASU with two closest chains (Does NOT match Rosetta symdef files)")
parser.add_argument("-b", "--bottom", default=False, action='store_true', help="Dumps bottom docks instead of top")
parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Verbose output text")
parser.add_argument("-l", "--list", default=False, action='store_true', help="List results as tasks file rather than dump pdb")
parser.add_argument("-p", "--listpath", type=str, default="./", help="Path to dump list file")

args = parser.parse_args()

nout_top = args.nout
output_path = args.input
output_name = output_path.split("/")[-1].split(".",1)[0]
output_filetype = output_path.split("/")[-1].split(".",-1)[-1]

if args.verbose:
    print( output_path )
    print( output_name )

#determine how to access the file based on file type
if ( output_filetype == "pickle" ):
    if args.verbose:
        print( "Input is a *.pickle file" )
    rpx_output = pickle.load( open(output_path, "rb"))
elif ( output_filetype == "txz" ):
    if args.verbose:
        print( "Input is a *.txz file" )
    rpx_output = rpxdock.search.result_from_tarball( output_path )
else:
    print( "Unknown input file type. It should be a *.pickle or a *.txz file" )
    exit()

#sort
best = np.argsort(-rpx_output.data.scores)

#check symmetry
sym = rpx_output.sym
print(sym)

if args.verbose:
    print(f"{rpx_output}")

#set top or bottom ranges
if args.bottom:
    direction = "bot"
    dir_range = slice(-nout_top-1,-1)
else:
    direction = "top"
    dir_range = slice(0,nout_top)

if args.verbose:
    if args.model:
        print(f"Dumping model #{args.model}")
    else:
        print(f"Dumping {direction} {nout_top} docks")

#prints tasks file
if args.list:
    f = open(f"{args.outpath}/jobs.list","w")
    count = 0

    for model in best[dir_range]:
        bod = rpx_output.bodies[rpx_output.ijob[model.values].values]
        body_names = [b.label for b in bod]
        output_body = list(range(len(bod)))
        if len(bod) > 1 and rpx_output.labels:
            bodlab = [rpx_output.labels[rpx_output.ijob[model.values].values][z] for z in output_body]
            body_names = [bl + '_' + lbl for bl, lbl in zip(bodlab, body_names)]
        middle = '__'.join(body_names)
        
        max_model_num = best[dir_range].values.max()
        max_model_num_char = len(str(max_model_num))

        if args.verbose:
            print(f"{os.path.abspath(output_path)},{sym}__{direction}{count}_{str(model.values).zfill(max_model_num_char)}__{middle}.pdb")
        f.write(f"{os.path.abspath(output_path)},{sym}__{direction}{count}_{str(model.values).zfill(max_model_num_char)}__{middle}.pdb\n")
        count += 1
    f.close()

    if args.model:
        '''
        if args.verbose:
            print(f"{os.path.abspath(output_path)},{sym}__{args.model}0_{str(model.values).zfill(max_model_num_char)}__{middle}.pdb")
        f.write(f"{os.path.abspath(output_path)},{sym}__{args.model}0_{str(model.values).zfill(max_model_num_char)}__{middle}.pdb\n")
        '''
        print("this script currently does not dump list files for single models.")

#dump pdb
else:
    if args.model:
        max_model_num = rpx_output.model.values.max()
        max_model_num_char = len(str(max_model_num))
        if args.fname:
            rpx_output.dump_pdb(int(args.model), fname=f"{args.outpath}/{args.fname}", use_orig_coords=True, output_asym_only=args.asu, output_closest_subunits=args.closest)
        else:
            rpx_output.dump_pdb(int(args.model), output_prefix=f"{args.outpath}/{sym}__mod0_{str(args.model).zfill(max_model_num_char)}", use_orig_coords=True, output_asym_only=args.asu, output_closest_subunits=args.closest)
    else:
        rpx_output.dump_pdbs(best[0:nout_top], lbl=direction, output_prefix=f"{args.outpath}/{sym}_", use_orig_coords=True, output_asym_only=args.asu, output_closest_subunits=args.closest)
