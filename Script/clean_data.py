import h5py

f = h5py.File('data_final.h5', 'r')
out = h5py.File('simple_motion_data.h5', 'a')
print(out.keys())
del out['property']
out.create_group('property')
for key in f['property'].keys():
    print(key)
    out['property'][key] = f['property'][key][()]
