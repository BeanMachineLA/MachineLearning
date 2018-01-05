1. Install NVIDIA DRIVERS 

    cd /filepath/to/install/packages (ex: /home/ubuntu)
    tar -zxvf nvidia_installers.tar.gz
    cd nvidia_installers
    sudo apt-get update && sudo apt-get -y upgrade
    sudo apt-get -y install build-essential
    sudo apt-get -y install linux-image-extra-virtual
    echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off" |sudo tee --append /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf > /dev/null
    sudo update-initramfs -u
    sudo apt-get -y install linux-source
    sudo apt-get -y install linux-headers-`uname -r`
    chmod +x cuda_7.5.18_linux.run
    sudo ./cuda_7.5.18_linux.run -silent -driver -toolkit -samples
    sudo /usr/bin/nvidia-uninstall -silent
    echo 'nouveau' | sudo tee -a /etc/modules
    chmod +x NVIDIA-Linux-x86_64-358.16.run
    sudo ./NVIDIA-Linux-x86_64-358.16.run -a --silent

2. Install cuDNN and link libraries
    tar -xzf cudnn-7.0-linux-x64-v4.0-prod.tgz
    cp cuda/lib64/* /usr/local/cuda/lib64/
    cp cuda/include/* /usr/local/cuda/include/
    cd /usr/local/cuda/lib64 && rm libcudnn.so libcudnn.so.4 && ln -s libcudnn.so.4.0.7 libcudnn.so.4 && ln -s libcudnn.so.4 libcudnn.so

3. Make cuda path available for all users    
    vi ~/.bashrc
    export PATH=$PATH:/usr/local/cuda-7.5/bin
    export LD_LIBRARY_PATH=:/usr/local/cuda-7.5/lib64
    source ~/.bashrc

4. Install Repositories:
    add-apt-repository -y 'ppa:ubuntu-toolchain-r/test' 

5. Install Dependencies:
    apt-get -y update 
    apt-get -y install gcc-4.9 
    apt-get -y install g++-4.9  
    apt-get -y install build-essential 
    apt-get -y install cmake 
    apt-get -y install git 
    apt-get -y install wget 
    apt-get -y install python-dev 
    apt-get -y install python-pip 
    apt-get -y install python-boto 
    apt-get -y install python-botocore 
    apt-get -y install libssl-dev      
    apt-get -y install libcurl4-openssl-dev 
    apt-get -y install libjsoncpp-dev 
    apt-get -y install libprotobuf-dev 
    apt-get -y install libleveldb-dev 
    apt-get -y install protobuf-compiler 
    apt-get -y install libgflags-dev 
    apt-get -y install libgoogle-glog-dev 
    apt-get -y install libsnappy-dev 
    apt-get -y install liblmdb-dev 
    apt-get -y install libhdf5-serial-dev 
    apt-get -y install python2.7-dev 
    apt-get -y install unzip 
    apt-get -y install libatlas-base-dev 
    apt-get -y install gfortran 
    apt-get -y install libjasper-dev 
    apt-get -y install libgtk2.0-dev 
    apt-get -y install libavcodec-dev 
    apt-get -y install libavformat-dev 
    apt-get -y install libswscale-dev 
    apt-get -y install libjpeg-dev 
    apt-get -y install libpng-dev 
    apt-get -y install libtiff-dev 
    apt-get -y install libjasper-dev 
    apt-get -y install libv4l-dev 
    apt-get -y install python-scipy 
    apt-get -y install libopenmpi1.6 
    apt-get -y install libstdc++6 
    apt-get -y install libboost-all-dev 

6. Install Pip Packages
    wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
    pip install boto
    pip install boto3 
    pip install protobuf 
    pip install numpy 
    pip install matplotlib

7.  rm -rf /var/lib/apt/lists/*       
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9 

8. Install c-ares
    cd /filepath/to/install/packages (ex: /home/ubuntu)
    tar -zxvf c-ares-1.11.0.tar.gz 
    cd c-ares-1.11.0 
    ./configure 
    make 
    sudo make install 

9. Install curl
    cd /filepath/to/install/packages (ex: /home/ubuntu)
    tar -xzvf curl-7.37.1.tar.gz 
    cd curl-7.37.1/ 
    ./configure --enable-ares=/home/ubuntu/c-ares-1.11.0 
    make 
    sudo make install 

10.  Install OpenCV
    cd /filepath/to/install/packages (ex: /home/ubuntu)
    tar -zxvf opencv-3.1.0.tar.gz 
    cd opencv-3.1.0 
    mkdir build && cd build 
    cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/usr/local \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=ON \ -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \ -D BUILD_EXAMPLES=ON \ -D WITH_FFMPEG=OFF \ -D WITH_CUDA=OFF \ -D WITH_QT=ON \ -D WITH_OPENGL=ON \ -D_FORCE_VTK=ON \ -D WITH_TBB=ON \ -D WITH_GDAL=ON \ -D BUILD_opencv_legacy=OFF \ -D PYTHON_INCLUDE_DIRS=/usr/include/python2.7 \ -D PYTHON_INCLUDE_DIR2=/usr/include/python2.7 \ -D PYTHON_LIBRARIES =/usr/lib/x86_64-linux-gnu/libpython2.7.so \ -D PYTHON_PACKAGES_PATH=/usr/local/lib/python2.7/dist-packages/\ -D PYTHON2_LIBRARIES =/usr/lib/x86_64-linux-gnu/libpython2.7.so .. #CHECK OPENCV_EXTRA_MODULES_PATH IS SPECIFIED CORRECTLY!!!!
    make 
    sudo cp lib/cv2.so /usr/local/lib/python2.7/dist-packages/ 
    sudo make install 
    echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/opencv.conf 
    sudo ldconfig

11. Install Caffe
    cd /filepath/to/install/packages (ex: /home/ubuntu)
    cd caffe
    git fetch && git checkout deepLab1.0.0
    cd matio-1.5.8
    chmod +x configure
    sudo ./configure
    sudo make 
    sudo make check
    sudo make install
    cd ../
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done 
    cp Makefile.config.example Makefile.config 
    sed -i -e 's/# USE_CUDNN := 1/USE_CUDNN := 1/g' Makefile.config      
    sed -i -e 's/# OPENCV_VERSION := 3/OPENCV_VERSION := 3/g' Makefile.config    
    sed -i -e 's/# USE_PKG_CONFIG := 1/USE_PKG_CONFIG := 1/g' Makefile.config 
    sed -i -e 's/# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/g' Makefile.config 
    sed -i -e 's/\/usr\/lib\/python2.7\/dist-packages\/numpy\/core\/include/\/usr\/local\/lib\/python2.7\/dist-packages\/numpy\/core\/include/g' Makefile.config 
    tar -xvzf /filepath/to/opencv/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e/ippicv_linux_20151201.tgz -C /home/ubuntu/opencv/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e #CHECK PATH TO OPENCV IS SPECIFIED CORRECTLY!!!
    sudo cp /filepath/to/opencv/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e/ippicv_lnx/lib/intel64/libippicv.a /usr/local/lib #CHECK PATH TO OPENCV IS SPECIFIED CORRECTLY!!!
    make 
    make pycaffe 
    make distribute 


12. Building detector

   cd /filepath/to/install/packages (ex: /home/ubuntu)   
   cd detector 
   ln -s /filepath/to/caffe/distribute caffe #CHECK PATH TO CAFFE IS SPECIFIED CORRECTLY!!!
   
   git fetch && git checkout vision-server2.3.3     

   mkdir build 
   cd build
   cmake .. 
   make
                  
13. Running:
      
      a) To run in "traditional" server mode (denoted "NUCLEUS" in the server code):
            ./hairColorServer <host> <port> <num_threads>
            
      b) To run in "VERTEX" mode:
            ./hairColorServer <host> <port> <num_threads> -pullRequest <queue_name>
            
         [NOTE: In vertex mode, the num_threads indicates how many threads are pulling from the SQS
                queue. One additional thread is launched to server other incoming requests,
                similar to the traditional server.]
                
      c) To kill the server (in either mode):
            curl "<host>:<port>/kill"
            [NOTE: The server will finish serving whatever requests it has already started, then it
                   will return. If you wish to kill immediately without finishing pending requests, 
                   kill the process via the OS.]
                   
                
14. Sending requests:

      a) All requests, whether they be from SQS or from external IPs, must be in the form of a JSON
         as follows:
         {
            "imageUrl" : "http://blahblah/path/to/image/someimage.jpg",
            "flags" : "some-specific-option,another-option,bold-lip-detect,hair-detect,stuff-like-that",
            "destinationUrl" : "http://somewhere/to/post/result/if/in/vertex/mode"
         }
         
      b) Currently, if the incoming JSON is not valid, the server will return a "bad-request" response.
         If the image url is not valid or does not contain image data, the server will return a "bad-request"
         response.
         Both "flags" and "destinationUrl" are optional entries. The request will still be processed if they
         are absent.
         
      c) NOTE: In bash, curly brackets and quotation marks within quotation marks must be escaped with
               a backslash ('\'). Therefore, when sending requests from the Ubuntu terminal, it needs
               to be sent like this:
         
               curl "localhost:5000/\{\"imageUrl\":\"http://www.whatever.jpg\"\}"
         
               Or, using single quotes on the outside, like this:
         
               curl 'localhost:5000/\{"imageUrl":"http://www.whatever.jpg"\}'
         
      d) NOTE: Right now, the server doesn't like requests that include spaces and the special character "!". Until we fix this,
               just avoid them in the json. (Requests from the SQS may have spaces or special characters.)
      
     

