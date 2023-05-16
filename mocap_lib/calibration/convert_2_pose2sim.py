# -- coding: utf-8 --
# @Time : 2022/6/9
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import os


def toml_write(calib_path, C, S, D, K, R, T):
    '''
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    OUTPUTS:
    - a .toml file cameras calibrations
    '''

    with open(os.path.join(calib_path), 'w+') as cal_f:
        for c in range(len(C)):
            cam = f'[cam_{c + 1}]\n'
            name = f'name = "{C[c]}"\n'
            size = f'size = [ {S[c][0]}, {S[c][1]}]\n'
            # print(K[c])
            mat = f'matrix = {K[c]}\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]}]\n'
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]}]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]}]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


if __name__ == '__main__':
    calib_path = './calibration.toml'
    C = ['268', '617', '728', '886']
    S = [[5312, 2988], [5312, 2988], [5312, 2988], [5312, 2988], ]
    D = [[
        -0.008009960635071488,
        0.00712042768222859,
        -0.0003040514304753879,
        -0.00011452984666135047,
        -0.0009440914412564298
    ], [
        -0.022026516849525744,
        0.008864087369986595,
        -0.003077609727293295,
        -0.004171932169766137,
        0.0018143793107834553
    ], [
        -0.020223359187737606,
        0.01419213257459139,
        -0.00145258430728195,
        -0.0025909192353510083,
        -0.004826869520377784
    ], [
        -0.025893216235269002,
        0.02988322798689544,
        -0.0013440290881121357,
        -0.0021785394037137865,
        -0.016227948826491145
    ]]
    K = [[
        [
          2529.1363563921864,
          0.0,
          2654.873698311312
        ],
        [
          0.0,
          2534.7027760640844,
          1492.7664547631634
        ],
        [
          0.0,
          0.0,
          1.0
        ]
      ],[
        [
          2482.7678083422943,
          0.0,
          2595.6016603144585
        ],
        [
          0.0,
          2484.9206282007403,
          1454.8558266888942
        ],
        [
          0.0,
          0.0,
          1.0
        ]
      ],[
        [
          2502.7919216055643,
          0.0,
          2629.329216104915
        ],
        [
          0.0,
          2507.3256892717422,
          1464.05671861415
        ],
        [
          0.0,
          0.0,
          1.0
        ]
      ],[
        [
          2461.9407162684965,
          0.0,
          2622.999214458836
        ],
        [
          0.0,
          2465.0411584455405,
          1473.7689921344427
        ],
        [
          0.0,
          0.0,
          1.0
        ]
      ],]
    R = [[[-0.7930652637144224, -0.6082914261371001, -0.032078472184104354], [-0.14227702487093316, 0.23618601510251042, -0.961235358517295], [0.5922877136016325, -0.7577583435093868, -0.27385681871821466]],
         [[0.08051103145758275, -0.9966897476771589, -0.01129250587238197],
          [-0.16758122566993958, -0.0023671951102416386, -0.985855430167259],
          [0.9825652683748785, 0.08126464952580352, -0.167217074848388]],
         [[-0.8307945122967964, 0.556338615615209, -0.016365302101747584],
          [0.1031372977677241, 0.1249902518712102, -0.9867827191161911],
          [-0.5469398286344673, -0.8215015409069689, -0.16122047680358498]],
         [[-0.9990874030979232, 0.007414683949098665, -0.04206403966552707],
          [0.04270860287571366, 0.16005138191225657, -0.9861843288089616],
          [-0.0005798374363927194, -0.987080836411037, -0.16022199030918133]]
         ]
    T = [[0.46911832936618797, 0.8991916164776033, 2.848475732879579], [0.4719593790845611, 1.158049624381319, 2.07161593828397], [0.09581168929854256, 0.9432026225648324, 3.567928002083741],[0.4472288641378439, 1.1150709069662623, 3.495747069028236]]
    toml_write(calib_path, C, S, D, K, R, T)
