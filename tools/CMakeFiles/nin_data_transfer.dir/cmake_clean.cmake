FILE(REMOVE_RECURSE
  "../test/nin_data_transfer.pdb"
  "../test/nin_data_transfer"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/nin_data_transfer.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
