import os
from pathlib import Path
from subprocess import Popen
import sys
import time
import os

'''
env config
'''

trt_exec_dict = {
    7234: './TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin/trtexec',
    8003: './TensorRT-8.0.0.3/targets/x86_64-linux-gnu/bin/trtexec',
    8206: './TensorRT-8.2.0.6/targets/x86_64-linux-gnu/bin/trtexec'
}
LD_LIBRARY_PATH = {
    7234: './TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib',
    8003: './TensorRT-8.0.0.3/targets/x86_64-linux-gnu/lib',
    8206: './TensorRT-8.2.0.6/targets/x86_64-linux-gnu/lib'
}

'''env config'''
trt_exec_version = 8003
os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH[trt_exec_version]

if __name__ == '__main__':

    '''config'''
    processing_onnx_name = '9A_64k_sim_bs4_8003.onnx'
    # processing_onnx_name='9A_64k.onnx'

    batch_size = 8
    latent_size = 512
    input_size = 256

    '''sim onnx 2 trt'''
    fmt = 16
    workspace = 512 * 80
    iofmt = 16
    print('start convert trt')
    processing_trt_name = processing_onnx_name.split('.')[0] + '_fp%d_ws%d_io%d' % (fmt, workspace, iofmt) + '.trt'
    if batch_size == -1:
        if fmt == 16:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s  --fp%d --workspace=%d --minShapes=target:1x3x256x256,vsid:1x512 --optShapes=target:1x3x256x256,vsid:1x512 --maxShapes=target:2x3x256x256,vsid:2x512' \
                         % (trt_exec_dict[trt_exec_version], processing_onnx_name, processing_trt_name, fmt,
                            abs(workspace * batch_size))
        else:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s   --workspace=%d --minShapes=target:1x3x256x256,vsid:1x512 --optShapes=target:8x3x256x256,vsid:8x512 --maxShapes=target:32x3x256x256,vsid:32x512' \
                         % (trt_exec_dict[trt_exec_version], processing_onnx_name, processing_trt_name,
                            abs(workspace * batch_size))
    else:
        if fmt == 16:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s --explicitBatch --fp%d --workspace=%d' \
                         % (trt_exec_dict[trt_exec_version], processing_onnx_name, processing_trt_name, fmt, workspace)
        else:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s --explicitBatch  --workspace=%d' \
                         % (trt_exec_dict[trt_exec_version], processing_onnx_name, processing_trt_name, workspace)
    if iofmt != 32:
        pb_trt_clt += ' --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw'
    retcode = Popen(pb_trt_clt, shell=True).wait()
