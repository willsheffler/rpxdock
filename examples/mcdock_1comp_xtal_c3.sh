cd /home/sheffler/project/rpxtal; 
rm -f *.pdb; 
CC='ccache gcc' \
PYTHONPATH=$folder \
python $folder/rpxdock/app/mcdock.py \
--architecture I213_c3 \
--mc_max_contacts 999 \
--mc_cell_bounds 120 200 \
--inputs1 /home/sheffler/project/diffusion/pointsym/test1/c3_diffusion_test_1.pdb \
--hscore_files ilv_h/1000 \
&> $folder/sublime_build.log \
&& pymol *.pdb
