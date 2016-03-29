make -j8
rm /tmp/nin_cifar10*
#sh scp.sh
mpirun -n 1 -hostfile HOSTFILE ./test/google_cifar10

