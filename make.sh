cmake .
make
rm -rf data/mnist/mnist-train-lmdb
rm -rf data/mnist/mnist-test-lmdb
./test/mnist_data
