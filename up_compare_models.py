import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
# from loss_funcs.svd_loss import sv_loss
import torch.nn.functional as F
from grl_m import WarmStartGradientReverseLayer
from typing import Optional, List, Dict, Tuple, Callable, Sequence

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class AdaptiveFeatureNorm(nn.Module):
    r"""
    The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`_

    Instead of using restrictive scalar R to match the corresponding feature norm, Stepwise Adaptive Feature Norm
    is used in order to learn task-specific features with large norms in a progressive manner.
    We denote parameters of backbone :math:`G` as :math:`\theta_g`, parameters of bottleneck :math:`F_f` as :math:`\theta_f`
    , parameters of classifier head :math:`F_y` as :math:`\theta_y`, and features extracted from sample :math:`x_i` as
    :math:`h(x_i;\theta)`. Full loss is calculated as follows

    .. math::
        L(\theta_g,\theta_f,\theta_y)=\frac{1}{n_s}\sum_{(x_i,y_i)\in D_s}L_y(x_i,y_i)+\frac{\lambda}{n_s+n_t}
        \sum_{x_i\in D_s\cup D_t}L_d(h(x_i;\theta_0)+\Delta_r,h(x_i;\theta))\\

    where :math:`L_y` denotes classification loss, :math:`L_d` denotes norm loss, :math:`\theta_0` and :math:`\theta`
    represent the updated and updating model parameters in the last and current iterations respectively.

    Args:
        delta (float): positive residual scalar to control the feature norm enlargement.

    Inputs:
        - f (tensor): feature representations on source or target domain.

    Shape:
        - f: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    Examples::

        >>> adaptive_feature_norm = AdaptiveFeatureNorm(delta=1)
        >>> f_s = torch.randn(32, 1000)
        >>> f_t = torch.randn(32, 1000)
        >>> norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
    """

    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss

class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0),
                                          predictions)  # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss

class MarginDisparityDiscrepancy(nn.Module):
    r"""The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.

    MDD can measure the distribution discrepancy in domain adaptation.

    The :math:`y^s` and :math:`y^t` are logits output by the main head on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial head.

    The definition can be described as:

    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        -\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} L_s (y^s, y_{adv}^s) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} L_t (y^t, y_{adv}^t),

    where :math:`\gamma` is a margin hyper-parameter, :math:`L_s` refers to the disparity function defined on the source domain
    and :math:`L_t` refers to the disparity function defined on the target domain.

    Args:
        source_disparity (callable): The disparity function defined on the source domain, :math:`L_s`.
        target_disparity (callable): The disparity function defined on the target domain, :math:`L_t`.
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - y_s: output :math:`y^s` by the main head on the source domain
        - y_s_adv: output :math:`y^s` by the adversarial head on the source domain
        - y_t: output :math:`y^t` by the main head on the target domain
        - y_t_adv: output :math:`y_{adv}^t` by the adversarial head on the target domain
        - w_s (optional): instance weights for source domain
        - w_t (optional): instance weights for target domain

    Examples::

        >>> num_outputs = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=4., source_disparity=F.l1_loss, target_disparity=F.l1_loss)
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_outputs), torch.randn(batch_size, num_outputs)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, margin: Optional[float] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction


    def source_discrepancy(self, y: torch.Tensor, y_adv: torch.Tensor):
        _, prediction = y.max(dim=1)
        return F.cross_entropy(y_adv, prediction, reduction='none')

    def target_discrepancy(self, y: torch.Tensor, y_adv: torch.Tensor):
        _, prediction = y.max(dim=1)
        return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_discrepancy(y_s, y_s_adv)
        target_loss = self.target_discrepancy(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        loss = source_loss + target_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
        r"""
        First shift, then calculate log, which can be described as:

        .. math::
            y = \max(\log(x+\text{offset}), 0)

        Used to avoid the gradient explosion problem in log(x) function when x=0.

        Args:
            x (torch.Tensor): input tensor
            offset (float, optional): offset size. Default: 1e-6

        .. note::
            Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
        """
        return torch.log(torch.clamp(x + offset, max=1.))

# class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
#     r"""
#     The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.
#
#     It measures the distribution discrepancy in domain adaptation
#     for classification.
#
#     When margin is equal to 1, it's also called disparity discrepancy (DD).
#
#     The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
#     The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
#     They are expected to contain raw, unnormalized scores for each class.
#
#     The definition can be described as:
#
#     .. math::
#         \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
#         \gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} \log\left(\frac{\exp(y_{adv}^s[h_{y^s}])}{\sum_j \exp(y_{adv}^s[j])}\right) +
#         \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} \log\left(1-\frac{\exp(y_{adv}^t[h_{y^t}])}{\sum_j \exp(y_{adv}^t[j])}\right),
#
#     where :math:`\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
#     You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.
#
#     Args:
#         margin (float): margin :math:`\gamma`. Default: 4
#         reduction (str, optional): Specifies the reduction to apply to the output:
#           ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#           ``'mean'``: the sum of the output will be divided by the number of
#           elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
#
#     Inputs:
#         - y_s: logits output :math:`y^s` by the main classifier on the source domain
#         - y_s_adv: logits output :math:`y^s` by the adversarial classifier on the source domain
#         - y_t: logits output :math:`y^t` by the main classifier on the target domain
#         - y_t_adv: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain
#
#     Shape:
#         - Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
#           with :math:`K \geq 1` in the case of `K`-dimensional loss.
#         - Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
#           :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.
#
#     Examples::
#
#         >>> num_classes = 2
#         >>> batch_size = 10
#         >>> loss = ClassificationMarginDisparityDiscrepancy(margin=4.)
#         >>> # logits output from source domain and target domain
#         >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
#         >>> # adversarial logits output from source domain and target domain
#         >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
#         >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
#     """
#
#     def __init__(self, margin: Optional[float] = 4, **kwargs):
#         def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
#             _, prediction = y.max(dim=1)
#             return F.cross_entropy(y_adv, prediction, reduction='none')
#
#         def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
#             _, prediction = y.max(dim=1)
#             return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')
#
#         super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin,
#                                                                        **kwargs)

def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix
class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks (ICML 2017) <https://arxiv.org/abs/1605.06636>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as

    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\

    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None

    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`

    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.

    Examples::

        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True, thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss

class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))

class TransferNet(nn.Module):
    def __init__(self, n_bands, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True,
                 bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.base_network.conv1 = nn.Conv2d(n_bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()
        #######MCC#############
        self.mcc = MinimumClassConfusionLoss(temperature=2.5)
        #######################

        #######MDD#############
        self.adv_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )
        # self.finetune = finetune
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=100,
                                                       auto_step=False)
        self.mdd = MarginDisparityDiscrepancy(4)
        #######################

        ######JDA###################
        self.jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=False, thetas=None)
        ############################

        ##############AFN############
        # self.adaptive_feature_norm = AdaptiveFeatureNorm(1)
        #############################




    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer_loss = sv_loss(source, target)
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target_clf = nn.Softmax(dim=1)(tar_clf)
            source_clf = nn.Softmax(dim=1)(source_clf)
        # transfer_loss = self.adapt_loss(source, target, **kwargs)

        #######mcc######
        transfer_loss = self.mcc(tar_clf)
        ###############

        #####mdd#####
        # feat = torch.cat([source, target], dim=0)
        # features_adv = self.grl_layer(feat)
        # outputs_adv = self.adv_head(features_adv)
        #
        # y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)
        #
        # transfer_loss = -self.mdd(source_clf, y_s_adv, target_clf, y_t_adv)
        #############

        ######jan####################
        # transfer_loss = self.jmmd_loss(
        #     (source, source_clf),
        #     (target, target_clf)
        # )
        #############################

        #########AFN##############
        # transfer_loss = 0.05 * self.adaptive_feature_norm(source) + 0.05 * self.adaptive_feature_norm(target)
        ##########################

        return clf_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass