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
or from OneDrive
```
https://1drv.ms/f/s!AsrkHbtkj4LsbqMZI08Bt9jFPJ4?e=SXkFlg
```

and place all file in this folder


# Motion Reconstruction

The `moconvq_base.data` contains a motion encoder and a physiscs-based motion decoder. 

Please refer to `.\Script\track_something.py` to get more information about the output of the motion encoder.

Or you can run the following command to reconstruct a kinematic motion into physics-based version
```
python .\Script\track_something.py base.bvh
```

# Motion Tokenization and Decoding
## Tokenization
You may use `.\Script\tokenize_motion.py` to convert a motion in bvh format into tokens, e.g.
```
python .\Script\tokenize_motion.py track.bvh -o out\tokens.txt
```

Run
```
python .\Script\tokenize_motion.py track.bvh -h
```
for more information

## Decoding
You may use `.\Script\decode_token.py` to decode a sequence of tokens into simulated motion, e.g.
```
python .\Script\decode_token.py -i 166 410 332 149 419 237 172 305 192 273 174 -o out\decode.bvh
```
or 
```
python .\Script\decode_token.py -f tokens.txt -o out\decode.bvh
```

Run
```
python .\Script\decode_token.py track.bvh -h
```
for more information



# Unconditional motion generation

```
python .\Script\unconditional_generation.py
```
You may use `--seed` argument to choose another random seed. It will generate a different motion.
```
python .\Script\unconditional_generation.py --seed 123
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