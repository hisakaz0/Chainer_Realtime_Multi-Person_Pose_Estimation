import chainer
from chainer import link
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe

from chainercv.links.model.ssd.multibox import Multibox
from chainercv.links.model.ssd.ssd_vgg16 import VGG16Extractor300


class ConvNet(link.Chain):

    """ConvNet is a former part of SSD model in chainercv.links.
    This model is from conv1_1 to conv4_3."""

    def __init__(self):
        super(ConvNet, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 3, pad=1)
            self.conv2_1 = L.Convolution2D(128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 3, pad=1)

            self.conv3_1 = L.Convolution2D(256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 3, pad=1)

            self.conv4_1 = L.Convolution2D(512, 3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 3, pad=1)
            self.conv4_3 = L.Convolution2D(512, 3, pad=1)


    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        return h


class LatterSsdNet(link.Chain):

    """LatterSsdNet is a latter part of SSD model in chainercv.links.
    This model is from conv5_1 to the last layer in the original model."""

    def __init__(self, n_fg_class):
        super(LatterSsdNet, self).__init__()
        with self.init_scope():
            # remained conv net in VGG16 class: chainercv/links/model/ssd/ssd_vgg16.py
            self.conv5_1 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_2 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_3 = L.DilatedConvolution2D(512, 3, pad=1)

            self.conv6 = L.DilatedConvolution2D(1024, 3, pad=6, dilate=6)
            self.conv7 = L.Convolution2D(1024, 1)

            # based on VGG16Extractor300 class: chainercv/links/model/ssd/ssd_vgg16.py
            self.conv8_1 = L.Convolution2D(256, 1)
            self.conv8_2 = L.Convolution2D(512, 3, stride=2, pad=1)

            self.conv9_1 = L.Convolution2D(128, 1)
            self.conv9_2 = L.Convolution2D(256, 3, stride=2, pad=1)

            self.conv10_1 = L.Convolution2D(128, 1)
            self.conv10_2 = L.Convolution2D(256, 3)

            self.conv11_1 = L.Convolution2D(128, 1)
            self.conv11_2 = L.Convolution2D(256, 3)

            # Multibox class: chainercv/links/model/ssd/ssd_vgg16.py
            self.multibox = Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)))


    def __call__(self, h):
        # `h` is output from ConvNet.
        ys = []

        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        ys.append(h)

        for i in range(8, 11 + 1):
            h = ys[-1]
            h = F.relu(self['conv{:d}_1'.format(i)](h))
            h = F.relu(self['conv{:d}_2'.format(i)](h))
            ys.append(h)

        return self.multibox(ys)


class LatterPoseNet(link.Chain):

    def __init__(self):
        super(LatterPoseNet, self).__init__(
            # cnn to make feature map
            conv4_4_CPM=L.Convolution2D(in_channels=256, out_channels=128, ksize=3, stride=1, pad=1),

            # stage1
            conv5_1_CPM_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_2_CPM_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_3_CPM_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_4_CPM_L1=L.Convolution2D(in_channels=128, out_channels=512, ksize=1, stride=1, pad=0),
            conv5_5_CPM_L1=L.Convolution2D(in_channels=512, out_channels=38, ksize=1, stride=1, pad=0),
            conv5_1_CPM_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_2_CPM_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_3_CPM_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_4_CPM_L2=L.Convolution2D(in_channels=128, out_channels=512, ksize=1, stride=1, pad=0),
            conv5_5_CPM_L2=L.Convolution2D(in_channels=512, out_channels=19, ksize=1, stride=1, pad=0),

            # stage2
            Mconv1_stage2_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage2_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage2_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage2_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage3
            Mconv1_stage3_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage3_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage3_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage3_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage4
            Mconv1_stage4_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage4_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage4_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage4_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage5
            Mconv1_stage5_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage5_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage5_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage5_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage6
            Mconv1_stage6_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage6_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage6_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage6_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),
        )

    def __call__(self, h):
        heatmaps = []
        pafs = []

        h = F.relu(self.conv4_4_CPM(h))
        feature_map = h

        # stage1
        h1 = F.relu(self.conv5_1_CPM_L1(feature_map)) # branch1
        h1 = F.relu(self.conv5_2_CPM_L1(h1))
        h1 = F.relu(self.conv5_3_CPM_L1(h1))
        h1 = F.relu(self.conv5_4_CPM_L1(h1))
        h1 = self.conv5_5_CPM_L1(h1)
        h2 = F.relu(self.conv5_1_CPM_L2(feature_map)) # branch2
        h2 = F.relu(self.conv5_2_CPM_L2(h2))
        h2 = F.relu(self.conv5_3_CPM_L2(h2))
        h2 = F.relu(self.conv5_4_CPM_L2(h2))
        h2 = self.conv5_5_CPM_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage2
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage2_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage2_L1(h1))
        h1 = F.relu(self.Mconv3_stage2_L1(h1))
        h1 = F.relu(self.Mconv4_stage2_L1(h1))
        h1 = F.relu(self.Mconv5_stage2_L1(h1))
        h1 = F.relu(self.Mconv6_stage2_L1(h1))
        h1 = self.Mconv7_stage2_L1(h1)
        h2 = F.relu(self.Mconv1_stage2_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage2_L2(h2))
        h2 = F.relu(self.Mconv3_stage2_L2(h2))
        h2 = F.relu(self.Mconv4_stage2_L2(h2))
        h2 = F.relu(self.Mconv5_stage2_L2(h2))
        h2 = F.relu(self.Mconv6_stage2_L2(h2))
        h2 = self.Mconv7_stage2_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage3
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage3_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage3_L1(h1))
        h1 = F.relu(self.Mconv3_stage3_L1(h1))
        h1 = F.relu(self.Mconv4_stage3_L1(h1))
        h1 = F.relu(self.Mconv5_stage3_L1(h1))
        h1 = F.relu(self.Mconv6_stage3_L1(h1))
        h1 = self.Mconv7_stage3_L1(h1)
        h2 = F.relu(self.Mconv1_stage3_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage3_L2(h2))
        h2 = F.relu(self.Mconv3_stage3_L2(h2))
        h2 = F.relu(self.Mconv4_stage3_L2(h2))
        h2 = F.relu(self.Mconv5_stage3_L2(h2))
        h2 = F.relu(self.Mconv6_stage3_L2(h2))
        h2 = self.Mconv7_stage3_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage4
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage4_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage4_L1(h1))
        h1 = F.relu(self.Mconv3_stage4_L1(h1))
        h1 = F.relu(self.Mconv4_stage4_L1(h1))
        h1 = F.relu(self.Mconv5_stage4_L1(h1))
        h1 = F.relu(self.Mconv6_stage4_L1(h1))
        h1 = self.Mconv7_stage4_L1(h1)
        h2 = F.relu(self.Mconv1_stage4_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage4_L2(h2))
        h2 = F.relu(self.Mconv3_stage4_L2(h2))
        h2 = F.relu(self.Mconv4_stage4_L2(h2))
        h2 = F.relu(self.Mconv5_stage4_L2(h2))
        h2 = F.relu(self.Mconv6_stage4_L2(h2))
        h2 = self.Mconv7_stage4_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage5
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage5_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage5_L1(h1))
        h1 = F.relu(self.Mconv3_stage5_L1(h1))
        h1 = F.relu(self.Mconv4_stage5_L1(h1))
        h1 = F.relu(self.Mconv5_stage5_L1(h1))
        h1 = F.relu(self.Mconv6_stage5_L1(h1))
        h1 = self.Mconv7_stage5_L1(h1)
        h2 = F.relu(self.Mconv1_stage5_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage5_L2(h2))
        h2 = F.relu(self.Mconv3_stage5_L2(h2))
        h2 = F.relu(self.Mconv4_stage5_L2(h2))
        h2 = F.relu(self.Mconv5_stage5_L2(h2))
        h2 = F.relu(self.Mconv6_stage5_L2(h2))
        h2 = self.Mconv7_stage5_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage6
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage6_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage6_L1(h1))
        h1 = F.relu(self.Mconv3_stage6_L1(h1))
        h1 = F.relu(self.Mconv4_stage6_L1(h1))
        h1 = F.relu(self.Mconv5_stage6_L1(h1))
        h1 = F.relu(self.Mconv6_stage6_L1(h1))
        h1 = self.Mconv7_stage6_L1(h1)
        h2 = F.relu(self.Mconv1_stage6_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage6_L2(h2))
        h2 = F.relu(self.Mconv3_stage6_L2(h2))
        h2 = F.relu(self.Mconv4_stage6_L2(h2))
        h2 = F.relu(self.Mconv5_stage6_L2(h2))
        h2 = F.relu(self.Mconv6_stage6_L2(h2))
        h2 = self.Mconv7_stage6_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        return pafs, heatmaps


class CocoPoseSsdNet(link.Chain):

    def __init__(self, n_fg_class):
        super(CocoPoseSsdNet, self).__init__()
        with self.init_scope():
            # shared conv net
            self.conv_net = ConvNet()
            self.ssd_net  = LatterSsdNet(n_fg_class)
            self.pose_net = LatterPoseNet()


    def __call__(self, x):

        # feature maps
        h = self.conv_net(x)
        # scores from ssd_net
        mb_locs, mb_confs = self.ssd_net(h)
        # scores from pose_net
        pafs, heatmaps = self.pose_net(h)


