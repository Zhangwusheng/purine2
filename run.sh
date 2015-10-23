make
rm /tmp/nin_cifar10*
sh scp.sh
mpirun -n 4 -hostfile HOSTFILE ./test/nin_cifar10
