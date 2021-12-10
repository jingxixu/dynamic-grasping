import os

models_dir = 'meshes'
target_models_dirname = 'meshes_obj'
target_models_dir = os.path.join(models_dir, target_models_dirname)
if not os.path.exists(target_models_dir):
    os.makedirs(target_models_dir)

model_filenames = [fname for fname in os.listdir(models_dir) if fname.endswith('.ply')]

for fname in model_filenames:
    # cmd = 'meshlabserver -i test_processed.ply -o output.obj -om vc fq wn'
    cmd = 'meshlabserver -i {} -o {} -om vc fq wn'.format(os.path.join(models_dir, fname),
                                                          os.path.join(target_models_dir, fname.replace('.ply', '.obj')))
    print(cmd)
    os.system(cmd)
