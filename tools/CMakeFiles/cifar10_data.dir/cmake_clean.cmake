FILE(REMOVE_RECURSE
  "../test/cifar10_data.pdb"
  "../test/cifar10_data"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/cifar10_data.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
