
import argparse

import numpy
from chainer.links.caffe import CaffeFunction
from chainer.serializers import load_npz, save_npz
from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb

from CocoPoseNet import CocoPoseNet
from CocoPoseSsdNet import CocoPoseSsdNet

pose_layers = [
    "conv4_4_CPM",
    "conv5_1_CPM_L1",
    "conv5_2_CPM_L1",
    "conv5_3_CPM_L1",
    "conv5_4_CPM_L1",
    "conv5_5_CPM_L1",
    "conv5_1_CPM_L2",
    "conv5_2_CPM_L2",
    "conv5_3_CPM_L2",
    "conv5_4_CPM_L2",
    "conv5_5_CPM_L2",
    "Mconv1_stage2_L1",
    "Mconv2_stage2_L1",
    "Mconv3_stage2_L1",
    "Mconv4_stage2_L1",
    "Mconv5_stage2_L1",
    "Mconv6_stage2_L1",
    "Mconv7_stage2_L1",
    "Mconv1_stage2_L2",
    "Mconv2_stage2_L2",
    "Mconv3_stage2_L2",
    "Mconv4_stage2_L2",
    "Mconv5_stage2_L2",
    "Mconv6_stage2_L2",
    "Mconv7_stage2_L2",
    "Mconv1_stage3_L1",
    "Mconv2_stage3_L1",
    "Mconv3_stage3_L1",
    "Mconv4_stage3_L1",
    "Mconv5_stage3_L1",
    "Mconv6_stage3_L1",
    "Mconv7_stage3_L1",
    "Mconv1_stage3_L2",
    "Mconv2_stage3_L2",
    "Mconv3_stage3_L2",
    "Mconv4_stage3_L2",
    "Mconv5_stage3_L2",
    "Mconv6_stage3_L2",
    "Mconv7_stage3_L2",
    "Mconv1_stage4_L1",
    "Mconv2_stage4_L1",
    "Mconv3_stage4_L1",
    "Mconv4_stage4_L1",
    "Mconv5_stage4_L1",
    "Mconv6_stage4_L1",
    "Mconv7_stage4_L1",
    "Mconv1_stage4_L2",
    "Mconv2_stage4_L2",
    "Mconv3_stage4_L2",
    "Mconv4_stage4_L2",
    "Mconv5_stage4_L2",
    "Mconv6_stage4_L2",
    "Mconv7_stage4_L2",
    "Mconv1_stage5_L1",
    "Mconv2_stage5_L1",
    "Mconv3_stage5_L1",
    "Mconv4_stage5_L1",
    "Mconv5_stage5_L1",
    "Mconv6_stage5_L1",
    "Mconv7_stage5_L1",
    "Mconv1_stage5_L2",
    "Mconv2_stage5_L2",
    "Mconv3_stage5_L2",
    "Mconv4_stage5_L2",
    "Mconv5_stage5_L2",
    "Mconv6_stage5_L2",
    "Mconv7_stage5_L2",
    "Mconv1_stage6_L1",
    "Mconv2_stage6_L1",
    "Mconv3_stage6_L1",
    "Mconv4_stage6_L1",
    "Mconv5_stage6_L1",
    "Mconv6_stage6_L1",
    "Mconv7_stage6_L1",
    "Mconv1_stage6_L2",
    "Mconv2_stage6_L2",
    "Mconv3_stage6_L2",
    "Mconv4_stage6_L2",
    "Mconv5_stage6_L2",
    "Mconv6_stage6_L2",
    "Mconv7_stage6_L2"
    ]

my_pose_layer = [
    "pose_net.conv4_4_CPM",
    "pose_net.conv5_1_CPM_L1",
    "pose_net.conv5_2_CPM_L1",
    "pose_net.conv5_3_CPM_L1",
    "pose_net.conv5_4_CPM_L1",
    "pose_net.conv5_5_CPM_L1",
    "pose_net.conv5_1_CPM_L2",
    "pose_net.conv5_2_CPM_L2",
    "pose_net.conv5_3_CPM_L2",
    "pose_net.conv5_4_CPM_L2",
    "pose_net.conv5_5_CPM_L2",
    "pose_net.Mconv1_stage2_L1",
    "pose_net.Mconv2_stage2_L1",
    "pose_net.Mconv3_stage2_L1",
    "pose_net.Mconv4_stage2_L1",
    "pose_net.Mconv5_stage2_L1",
    "pose_net.Mconv6_stage2_L1",
    "pose_net.Mconv7_stage2_L1",
    "pose_net.Mconv1_stage2_L2",
    "pose_net.Mconv2_stage2_L2",
    "pose_net.Mconv3_stage2_L2",
    "pose_net.Mconv4_stage2_L2",
    "pose_net.Mconv5_stage2_L2",
    "pose_net.Mconv6_stage2_L2",
    "pose_net.Mconv7_stage2_L2",
    "pose_net.Mconv1_stage3_L1",
    "pose_net.Mconv2_stage3_L1",
    "pose_net.Mconv3_stage3_L1",
    "pose_net.Mconv4_stage3_L1",
    "pose_net.Mconv5_stage3_L1",
    "pose_net.Mconv6_stage3_L1",
    "pose_net.Mconv7_stage3_L1",
    "pose_net.Mconv1_stage3_L2",
    "pose_net.Mconv2_stage3_L2",
    "pose_net.Mconv3_stage3_L2",
    "pose_net.Mconv4_stage3_L2",
    "pose_net.Mconv5_stage3_L2",
    "pose_net.Mconv6_stage3_L2",
    "pose_net.Mconv7_stage3_L2",
    "pose_net.Mconv1_stage4_L1",
    "pose_net.Mconv2_stage4_L1",
    "pose_net.Mconv3_stage4_L1",
    "pose_net.Mconv4_stage4_L1",
    "pose_net.Mconv5_stage4_L1",
    "pose_net.Mconv6_stage4_L1",
    "pose_net.Mconv7_stage4_L1",
    "pose_net.Mconv1_stage4_L2",
    "pose_net.Mconv2_stage4_L2",
    "pose_net.Mconv3_stage4_L2",
    "pose_net.Mconv4_stage4_L2",
    "pose_net.Mconv5_stage4_L2",
    "pose_net.Mconv6_stage4_L2",
    "pose_net.Mconv7_stage4_L2",
    "pose_net.Mconv1_stage5_L1",
    "pose_net.Mconv2_stage5_L1",
    "pose_net.Mconv3_stage5_L1",
    "pose_net.Mconv4_stage5_L1",
    "pose_net.Mconv5_stage5_L1",
    "pose_net.Mconv6_stage5_L1",
    "pose_net.Mconv7_stage5_L1",
    "pose_net.Mconv1_stage5_L2",
    "pose_net.Mconv2_stage5_L2",
    "pose_net.Mconv3_stage5_L2",
    "pose_net.Mconv4_stage5_L2",
    "pose_net.Mconv5_stage5_L2",
    "pose_net.Mconv6_stage5_L2",
    "pose_net.Mconv7_stage5_L2",
    "pose_net.Mconv1_stage6_L1",
    "pose_net.Mconv2_stage6_L1",
    "pose_net.Mconv3_stage6_L1",
    "pose_net.Mconv4_stage6_L1",
    "pose_net.Mconv5_stage6_L1",
    "pose_net.Mconv6_stage6_L1",
    "pose_net.Mconv7_stage6_L1",
    "pose_net.Mconv1_stage6_L2",
    "pose_net.Mconv2_stage6_L2",
    "pose_net.Mconv3_stage6_L2",
    "pose_net.Mconv4_stage6_L2",
    "pose_net.Mconv5_stage6_L2",
    "pose_net.Mconv6_stage6_L2",
    "pose_net.Mconv7_stage6_L2"
    ]

ssd_layers = [
    "conv1_1",
    "conv1_2",
    "conv2_1",
    "conv2_2",
    "conv3_1",
    "conv3_2",
    "conv3_3",
    "conv4_1",
    "conv4_2",
    "conv4_3",
    "norm4",
    "conv5_1",
    "conv5_2",
    "conv5_3",
    "fc6",
    "fc7",
    "conv6_1",
    "conv6_2",
    "conv7_1",
    "conv7_2",
    "conv8_1",
    "conv8_2",
    "conv9_1",
    "conv9_2",
    "conv4_3_norm_mbox_conf",
    "fc7_mbox_conf",
    "conv6_2_mbox_conf",
    "conv7_2_mbox_conf",
    "conv8_2_mbox_conf",
    "conv9_2_mbox_conf",
    "conv4_3_norm_mbox_loc",
    "fc7_mbox_loc",
    "conv6_2_mbox_loc",
    "conv7_2_mbox_loc",
    "conv8_2_mbox_loc",
    "conv9_2_mbox_loc"
    ]

my_ssd_layers = [
    "conv_net.conv1_1",
    "conv_net.conv1_2",
    "conv_net.conv2_1",
    "conv_net.conv2_2",
    "conv_net.conv3_1",
    "conv_net.conv3_2",
    "conv_net.conv3_3",
    "conv_net.conv4_1",
    "conv_net.conv4_2",
    "conv_net.conv4_3",
    "ssd_net.norm4",
    "ssd_net.conv5_1",
    "ssd_net.conv5_2",
    "ssd_net.conv5_3",
    "ssd_net.conv6",
    "ssd_net.conv7",
    "ssd_net.conv8_1",
    "ssd_net.conv8_2",
    "ssd_net.conv9_1",
    "ssd_net.conv9_2",
    "ssd_net.conv10_1",
    "ssd_net.conv10_2",
    "ssd_net.conv11_1",
    "ssd_net.conv11_2",
    "ssd_net.multibox.conf[0]",
    "ssd_net.multibox.conf[1]",
    "ssd_net.multibox.conf[2]",
    "ssd_net.multibox.conf[3]",
    "ssd_net.multibox.conf[4]",
    "ssd_net.multibox.conf[5]",
    "ssd_net.multibox.loc[0]",
    "ssd_net.multibox.loc[1]",
    "ssd_net.multibox.loc[2]",
    "ssd_net.multibox.loc[3]",
    "ssd_net.multibox.loc[4]",
    "ssd_net.multibox.loc[5]"
    ]



def copy_conv_net(src, dst, src_lname, dst_lname):
    exec("dst.{}.W.data = src.{}.W.data.copy()". format(dst_lname, src_lname))
    exec("dst.{}.b.data = src.{}.b.data.copy()". format(dst_lname, src_lname))
    print("Copy layer: dst({}) / src({})". format(dst_lname, src_lname))


def copy_ssd_norm_layer(caffe_ssd_path, pose_ssd_model):
    net = caffe_pb.NetParameter()
    with open(caffe_ssd_path, 'rb') as model_file:
        net.MergeFromString(model_file.read())

    def get_normalize_layer(net):
        for layer in net.layer:
            if layer.type == 'Normalize':
                return layer
    norm_layer = get_normalize_layer(net)
    pose_ssd_model.ssd_net.norm4.scale.data = numpy.array(norm_layer.blobs[0].data)
    print("Copy layer: dst({}) / src({})". format(
        "ssd_net.norm4", "conv4_3_norm"))

def main():

    parser = argparse.ArgumentParser( \
            description="""Convert caffemodel SSD, and load pose model,
            and the save PoseSsdNet model.""")
    parser.add_argument("--pose_model_path",
            default="coco_posenet.npz",
            help="Path to the pose model in chainer")
    parser.add_argument("--ssd_model_path",
            default="ssd/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel",
            help="Path to the SSD model in caffe")
    parser.add_argument("--save_model_path",
            default="coco_pose_ssd_net.npz",
            help="Path to the pose ssd model")
    args  = parser.parse_args()

    n_class = 80 # coco
    pose_ssd_model = CocoPoseSsdNet(n_class)
    print("Loaded pose_ssd_net")

    # copy pose net params
    pose_model = CocoPoseNet()
    load_npz(args.pose_model_path, pose_model)
    print("Loaded pose_net")
    for src_lname, dst_lname in zip(pose_layers, my_pose_layer):
        copy_conv_net(pose_model, pose_ssd_model, src_lname, dst_lname)

    # copy ssd net params
    ssd_model = CaffeFunction(args.ssd_model_path)
    print("Loaded caffe_ssd_net")
    for src_lname, dst_lname in zip(ssd_layers, my_ssd_layers):
        if src_lname == 'norm4':
            copy_ssd_norm_layer(args.ssd_model_path, pose_ssd_model)
        else:
            copy_conv_net(ssd_model, pose_ssd_model, src_lname, dst_lname)

    # save pose ssd model
    save_npz(args.save_model_path, pose_ssd_model)


if __name__ == "__main__":
    main()
