# -- coding: utf-8 --
# @Time : 2022/5/16
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

# install
pip install multical

# separate
multical intrinsic --image_path ./ --boards ./example_boards/intrinsic/charuco_A1_88_intri_4.yaml
multical calibrate --image_path ./ --calibration ./intrinsic.json --fix_intrinsic --boards ./example_boards/charuco_A1_44.yaml

# generate board
multical boards --boards ./example_boards/charuco_A1_44.yaml --paper_size A1 --pixels_mm 10 --write my_images

# intrinsic and extrinsic
multical calibrate --image_path ./ --boards ./example_boards/charuco_A1_44.yaml --limit_images 200 --fix_aspect
multical vis --workspace_file calibration.pkl
