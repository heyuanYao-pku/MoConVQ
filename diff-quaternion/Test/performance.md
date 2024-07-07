## Performance

Here are experiment settings on CPU:
- ours(1): Our cpp implementation on 1 single thread.
- scipy: using scipy.spatial.transform.Rotation
- pyquaternion: using pyquaternion
  
Here are experiment settings on GPU:
- libtorch: using C++ libtorch
- pytorch: using pytorch
- jit: using pytorch with torch.jit.script
- cuda: Our cuda implementation

My system configuration: 
- Windows 10 x64 + Visual Studio 2022 + Python 3.10.8
- cuda 11.7 + pytorch 1.13.0
- GPU: NVIDIA GTX 1060
- Memory: 32 GB
- CPU: Intel i7-8700 @ 3.20 GHz, 6 cores, 12 threads

Time usage of quaternion multiply on CPU:

| nums   | Ours(openmp) | Ours(1) | scipy |  scipy (1) |
|------- | -------------| --------| ----- | ---------- |
|  1     | Header       | Title   |       |            |
|  10    | Paragraph    | Text    |       |            |
|  1e2   | Paragraph    | Text    |       |            |
|  1e3   | Paragraph    | Text    |       |            |

It is shown that for large data, our implementation on 1 thread is about 50 times faster than scipy and pyquaternion.

Using openmp is slower than single thread. Why? Because of GIL in Python? 