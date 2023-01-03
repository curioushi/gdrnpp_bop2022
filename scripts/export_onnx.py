import os
import os.path as osp
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    return args

def export_onnx(config_path, input_dir, output_dir):
    ckpt_files = sorted(glob(osp.join(input_dir, "*.pth")))
    for ckpt_file in tqdm(ckpt_files):
        print(f'exporting {ckpt_file}...')
        name = osp.splitext(osp.basename(ckpt_file))[0]
        os.system(f'./core/gdrn_modeling/test_gdrn.sh {config_path} 0 {ckpt_file}')
        os.system(f'mv gdrn.onnx model.onnx')
        os.system(f'zip {name}.zip model.onnx')
        os.system(f'mv {name}.zip {output_dir}')

if __name__ == '__main__':
    args = parse_args()
    export_onnx(args.config_path, args.input_dir, args.output_dir)

