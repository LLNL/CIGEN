# fetch third-party libraries
cd ~/cigen

make -f baseline_civ/Makefile.plugin 
make -f baseline_civ/Makefile.plugin clean

cd ~/

rm -r gsl-2.7.1 gsl-2.7.1.tar.gz
wget https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz
tar -zxvf gsl-2.7.1.tar.gz
cd gsl-2.7.1

./configure CC=clang CFLAGS="-g -O0 -fpass-plugin=libTraceDiffPass.so"
make -j
cp .libs/libgsl.a ../cigen/gsl_test/libgsl_0_0_1.a
make clean

./configure CC=clang CFLAGS="-g -O3 -ffast-math -fpass-plugin=libTraceDiffPass.so"
make -j
cp .libs/libgsl.a ../cigen/gsl_test/libgsl_3_1_1.a
make clean

./configure CC=clang CFLAGS="-g -O0"
make -j
cp .libs/libgsl.a ../cigen/gsl_test/libgsl_0_0_0.a
make clean

./configure CC=clang CFLAGS="-g -O3 -ffast-math"
make -j
cp .libs/libgsl.a ../cigen/gsl_test/libgsl_3_1_0.a
# no clean for header files
# make clean