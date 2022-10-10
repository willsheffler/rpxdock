#TEMPLATE FOR RUNNING

motif_dir="/path/to/hscore/" #(eg. /net/software/rpxdock/hscore/)

PYTHONPATH=/path/to/python/packages \
/path/to/conda/environment/python \
/path/to/rpxdock/app/dock.py \
    --filter_config /path/to/filters.yml \
    --architecture ${architecture_keyword} \
    --inputs1 ${/path/to/input/pdb1} \
    --allowed_residues1 ${/path/to/allowed_residues1.txt} \
    --inputs2 ${/path/to/input/pdb2} \
    --allowed_residues2 ${/path/to/allowed_residues2.txt} \
    --cart_bounds ${d1} ${d2} \
    --hscore_files ilv_h \
    --hscore_data_dir ${motif_dir} \
    --loglevel INFO \
    --use_orig_coords \
    --score_only_ss EHL \
    --output_prefix ${/path/to/output/folder} \
    --function stnd \
    --save_results_as_tarball false \
    --save_results_as_pickle true \
    --overwrite_existing_results \
    --dump_pdbs \
    --dump_result_summary