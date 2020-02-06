# model settings
# 和maskrcnn做个对比
model = dict(
    type='FCOS',
    # todo: 应该要建个VISFCOS了
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        # 参数无顺序
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # todo: 设置start_level为1，默认是0
        # 融合输出5层，但去掉了0层产生，但是增加了额外的一层
        # 产生这融合的5层
        start_level=1,
        # todo: 默认为false
        add_extra_convs=True,
        # 默认为True 决定了extra_convs_on_inputs的in_channels
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        # 默认为False， 在额外卷积层加一个激活函数
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        # todo: 不知是增加一个maskhead还是改FCOSHead,最好改
        # 并行结构可能不好弄
        # 这里的bbox_head是用来预测点而非框的FCOSHead
        num_classes=41,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        # 定义三个损失
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
    # todo: 此处加一个mask_head或mask_head、mask_roi_extractor

# training and testing settings
# train_cfg和test_cfg参数设置的区别在于one or two stage?
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
# todo: 数据改变意味着fcos原有的预处理方式要重新实现？不用，没什么操作
dataset_type = 'YTVOSDataset'
data_root = 'data/youtubevos/'
# todo: 这里的参数有变且，to_rgb=False
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
# 不使用pipline做预处理操作
# train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(
        # type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        # flip=False,
        # transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            # dict(type='Collect', keys=['img']),
        # ])
# ]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_sub.json',
        img_prefix=data_root + 'train/JPEGImages',
        # pipeline=train_pipeline
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        # todo：把with_mask，with_track变为False,当成目标检测任务做？
        with_mask=False,
        with_crowd=True,
        with_label=True,
        with_track=True
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_sub.json',
        img_prefix=data_root + 'valid/JPEGImages',
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        with_track=True
        # pipeline=test_pipeline
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_sub.json',
        img_prefix=data_root + 'valid/JPEGImages',
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        with_track=True
        # pipeline=test_pipeline
        ))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
# todo: grad_clip和mask不一样
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # 试试TB
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/vis_fcos_r50_caffe_fpn_gn_1x_4gpu'
load_from = None
resume_from = None
workflow = [('train', 1)]
