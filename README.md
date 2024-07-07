# MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations (SIGGRAPH 2024 Journal Track)

Project Page: https://moconvq.github.io/

The code provides:

- [x] Pretrained model and code for MoConVQ representation
- [x] Pretrained model and code for MoConGPT
- [ ] Trainging code for MoConVQ
- [ ] Trainging code for MoConGPT

# Install

Environment installation is a bit complicated, so we have prepared a script for installation, please refer to `setup.cmd`


Pretrained data: Download from 

```
https://disk.pku.edu.cn/link/AAAFE3B2DDB1AC420EB5C4E0910196116F
```

and place all file in this folder

# Motion Reconstruction

The `moconvq_base.data` contains a motion encoder and a physiscs-based motion decoder. 

Please refer to `.\Script\track_something.py` to get more information about the output of the motion encoder.

Or you can run the following command to reconstruct a kinematic motion into physics-based version
```
python .\Script\track_something.py base.bvh
```

# Unconditional motion generation

```
python .\Script\unconditional_generation.py
```


# text-to-motion generation


First install some additional packages:
```
pip install transformers sentencepiece
```
Then run the code:
```
 python .\Script\text2motion_generation.py
```

text description can be found in the python script