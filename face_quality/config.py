_base_ = [
    '../default_runtime.py'
]

# dataset settings
dataset_type = 'Filelist'
img_norm_cfg = dict(
    mean=[125.09, 102.01, 93.19],
    std=[71.35, 63.75, 61.46],
    to_rgb=True)
train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/fq_0911',
        ann_file='data/fq_0911/train.txt',
        pipeline=train_pipeline,
        classes=['high', 'low']),
    val=dict(
        type=dataset_type,
        data_prefix='data/fq_0911',
        ann_file='data/fq_0911/val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        classes=['high', 'low']),
    test=dict(
        type=dataset_type,
        # data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True,
        classes=['high', 'low']),
)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=2,
        in_channels=576,
        mid_channels=[1280],
        act_cfg=dict(type='HSwish'),
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)
    ),
)

load_from = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[15, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
