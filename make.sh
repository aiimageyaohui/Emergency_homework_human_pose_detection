#! /bin/bash
# cd models/networks/DCNv2

# we using mmdetection's dcn for DCN onnx export
cd models/networks/dcn
rm *.so
sudo rm -r build
sudo chmod -R 777 make.sh
./make.sh
cd -
cd external/
rm *.so
make
echo 'ok'
