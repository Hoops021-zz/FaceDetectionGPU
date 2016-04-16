# FaceDetectionGPU
Viola Jones Face Detection Algorithm implemented on GPUs using NVidia CUDA framework


This project requires OpenCV to be installed and setup properly on machine running this code. 

Furthermore, this project requires an NVidia GPU on the machine & the CUDA SDK installed. This code was last tested on CUDA 7.5 on Windows 10
https://developer.nvidia.com/cuda-downloads

NOTE: If using Visual Studio 2015 then use vc14 in all paths below instead of vc12(Visual STudio 2013)

1) Download OpenCV 3.1 (http://opencv.org/downloads.html)
2) Install in C:\ (i.e C:\opencv\)
3) Add "C:\opencv\build\x64\vc12\bin\" to "Path" environment variable for Windows (Right click Computer -> Properties -> Advanced System Settings -> Environment Variable)

4) Open Visual studio 2013
5) Set configuration to x64(configuration manager at top of Debug/Release options)
6) Right click solution Properties 
	a) C/C++ -> Additional INclude Directories -> Add "C:\opencv\build\include\"
	b) Linker -> General -> Additional Libary Dependencies -> Add "C:\opencv\build\x64\vc12\lib\"
	c) Input -> Additional Dependencies -> Edit 
		1) For Debug config, add "opencv_world310d.lib"
		2) For Release config, add "opencv_world310.lib"
   