#example command:
<<<<<<< HEAD
#./rpxdock_to_design.sh RPXDock_Result.pickle,RPXDock_pdb.pdb

#parse inputs
input=${1}
debug=${2}
rpx_pickle_path=`echo ${input} |cut -d',' -f1`

file_input=`echo ${input} |cut -d',' -f2`
=======
#./rpxdock_to_design.sh /path/to/extracted/RPXDock_pdb.pdb

#parse inputs
file_input=${1}
>>>>>>> origin/master

#parse components
sym="I53" #Change to architecture from RPXDock
arche=`echo ${sym} |cut -c1`
axis1=`echo ${sym} |cut -c2`
axis2=`echo ${sym} |cut -c3`
axis3=`echo ${sym} |cut -c4`
<<<<<<< HEAD
if [[ ${debug} ]]; then echo "axis123 " ${axis1} ${axis2} ${axis3}; fi
=======
>>>>>>> origin/master

#check number of components
if [[ -z ${axis1} ]]; then echo "axis 1 cannot be empty!"; exit
elif [[ -z ${axis2} ]]; then num_comp="1"
elif [[ -z ${axis3} ]]; then num_comp="2"
else num_comp="3"; fi

#make folders
mkdir -p output/
outpath="output/"
    
#symmmetry
<<<<<<< HEAD
if [[ ${num_comp} == "1" ]]; then
=======
#cyclic symmetries
if [[ ${arche} == "C" ]]; then
    nsub_bb="1"
    symfile="/path/to/${sym}.sym" #these are included in a standard Rosetta compile
    symdof="JS1"
  
#cage symmetries
elif [[ ${num_comp} == "1" ]]; then
>>>>>>> origin/master
    nsub_bb=${axis1}
    symfile="/path/to/${sym}.sym"
    if   [[ ${sym} == "T2" ]]; then symdof1="JDP1"
    elif [[ ${sym} == "T3" ]]; then symdof1="JAC1"
    elif [[ ${sym} == "O2" ]]; then symdof1="JDP1"
    elif [[ ${sym} == "O3" ]]; then symdof1="JTP1"
    elif [[ ${sym} == "O4" ]]; then symdof1="JQP1"
    elif [[ ${sym} == "I2" ]]; then symdof1="JCD00"
    elif [[ ${sym} == "I3" ]]; then symdof1="JCT00"
    elif [[ ${sym} == "I5" ]]; then symdof1="JCP00"
    else echo "undefined 1-comp sym?"; exit ; fi
    symdof2="${symdof1}"
<<<<<<< HEAD
=======
    
>>>>>>> origin/master
elif [[ ${num_comp} == "2" ]]; then
    nsub_bb="1"
    symfile="/path/to/${sym}.sym"
    if   [[ ${sym} == "T32" ]]; then symdof1="JTP1";  symdof2="JDP1"
    elif [[ ${sym} == "T33" ]]; then symdof1="JAC1";  symdof2="JBC1"
    elif [[ ${sym} == "O32" ]]; then symdof1="JTP1";  symdof2="JDP1"
    elif [[ ${sym} == "O42" ]]; then symdof1="JQP1";  symdof2="JDP1"
    elif [[ ${sym} == "O43" ]]; then symdof1="JQP1";  symdof2="JTP1"
    elif [[ ${sym} == "I32" ]]; then symdof1="JCT00"; symdof2="JCD00"
    elif [[ ${sym} == "I52" ]]; then symdof1="JCP00"; symdof2="JCD00"
    elif [[ ${sym} == "I53" ]]; then symdof1="JCP00"; symdof2="JCT00"
    else echo "undefined 2-comp sym?"; exit ; fi
<<<<<<< HEAD
elif [[ ${num_comp} == "3" ]]; then
    nsub_bb="1"
    symfile="path/to/files/here.sym"
    if [[ 1 == 0 ]]; then echo "???";
    else echo "undefined 3-comp sym?"; exit ; fi
=======
    
elif [[ ${num_comp} == "3" ]]; then
    echo "this script doesn't work for 3-comp stuff yet"
    
>>>>>>> origin/master
else
    echo "undefined sym ?????"; exit
fi

<<<<<<< HEAD
#create run_rosetta function
run_rosetta () {
    exe_path="/path/to/rosetta_scripts.hdf5.linuxgccrelease" \
=======
#run Rosetta
/path/to/rosetta_scripts.hdf5.linuxgccrelease" \
>>>>>>> origin/master
        -out:level 300 \
        -never_rerun_filters false \
        -out::file::pdb_comments \
        -parser:protocol rpxdock_to_design.xml \
        -s      ${file_input} \
        -native ${file_input} \
        -nstruct 1 \
        -parser:script_vars \
            sym="${symfile}" \
            comp="${num_comp}comp" nsub_bb="${nsub_bb}" \
            symdof1="${symdof1}" symdof2="${symdof2}" symdof3="${symdof3}" \
        -overwrite 1 \
        -out:chtimestamp 1 \
        -out:suffix "" \
        -out::path::all ${outpath}/ \
<<<<<<< HEAD
        -output_only_asymmetric_unit true \
        -failed_job_exception false \
        -mute all \
        -unmute  \
            protocols.simple_moves.AddResidueLabelMover \
        -renumber_pdb true \
}

#initialize run_rosetta
if [[ ${debug} ]]; then
    run_rosetta |& tee ${outpath}/rpxdock_to_design.log
    date
    echo "run complete"
else
    run_rosetta > ${outpath}/rpxdock_to_design.log
fi
=======
        -failed_job_exception false \
        -renumber_pdb true \
        > ${outpath}/rpxdock_to_design.log
>>>>>>> origin/master
