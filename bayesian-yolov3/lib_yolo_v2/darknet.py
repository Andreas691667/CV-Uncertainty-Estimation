import os
import glob
import numpy as np
import tensorflow as tf


def darknet53(model_builder, training, trainable):
    mb = model_builder

    mb.make_darknet_conv_layer(32, 3, training, trainable)  # 0

    # Downsample (factor 2)
    mb.make_darknet_downsample_layer(64, 3, training, trainable)  # 1

    mb.make_darknet_residual_block(32, training, trainable)  # 2 - 4

    # Downsample (factor 4)
    mb.make_darknet_downsample_layer(128, 3, training, trainable)  # 5

    for i in range(2):
        mb.make_darknet_residual_block(64, training, trainable)  # 6 - 11

    # Downsample (factor 8)
    mb.make_darknet_downsample_layer(256, 3, training, trainable)  # 12

    for i in range(8):
        mb.make_darknet_residual_block(128, training, trainable)  # 13 - 36

    # Downsample (factor 16)
    mb.make_darknet_downsample_layer(512, 3, training, trainable)  # 37

    for i in range(8):
        mb.make_darknet_residual_block(256, training, trainable)  # 38 - 61

    # Downsample (factor 32)
    mb.make_darknet_downsample_layer(1024, 3, training, trainable)  # 62

    for i in range(4):
        mb.make_darknet_residual_block(512, training, trainable)  # 63 - 74


def load_darknet_weights(net_layers, weightfile):
    with open(weightfile, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    tmp = tf.compat.v1.global_variables()
    vars = {}
    for var in tmp:
        vars[var.name] = var
        
    # save vars as json
    import json
    # with open('vars.json', 'w') as f:
    #     json.dump(vars, f, indent=4, default=lambda x: str(x))

    assign_ops = []

    for i, l in enumerate(net_layers):
        print(f'{i}: {l.name}')
        if 'LeakyRelu' not in l.name:
            continue

        batch_norm = 'detection' not in l.name
        load_bias = not batch_norm
        if batch_norm:
            ptr = _load_batch_norm(l, vars, ptr, weights, assign_ops)

        if 'conv' in l.name:
            ptr = _load_conv2d(l, vars, ptr, weights, assign_ops, load_bias)

    print(f'len(weights): {len(weights)}')
    print(f'ptr: {ptr}')
    print(f'len(assign_ops): {len(assign_ops)}')    
    assert ptr == len(weights)
    return assign_ops


def _load_conv2d(l, vars, ptr, weights, assign_ops, load_bias):
    namespace = l.name.split('/')
    if 'conv_1' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_2')   
    elif 'conv_2' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_3')
    elif 'conv_3' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_5')
    elif 'conv_4' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_6')
    elif 'conv_5' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_7')
    elif 'conv_6' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_8')
    elif 'conv_7' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_10')
    elif 'conv_8' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_11')
    elif 'conv_9' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_12')
    elif 'conv_10' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_13')
    elif 'conv_11' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_14')
    elif 'conv_12' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_15')
    elif 'conv_13' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_16')
    elif 'conv_14' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_17')
    elif 'conv_15' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_18')
    elif 'conv_16' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_19')
    elif 'conv_17' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_20')
    elif 'conv_18' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_21')
    elif 'conv_19' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_22')
    elif 'conv_20' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_23')
    elif 'conv_21' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_24')
    elif 'conv_22' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_25')
    elif 'conv_23' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_27')
    elif 'conv_24' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_28')
    elif 'conv_25' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_29')
    elif 'conv_26' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_30')
    elif 'conv_27' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_31')
    elif 'conv_28' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_32')
    elif 'conv_29' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_33')
    elif 'conv_30' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_34')
    elif 'conv_31' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_35')
    elif 'conv_32' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_36')
    elif 'conv_33' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_37')
    elif 'conv_34' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_38')
    elif 'conv_35' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_39')
    elif 'conv_36' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_40')
    elif 'conv_37' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_41')
    elif 'conv_38' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_42')
    elif 'conv_39' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_44')
    elif 'conv_40' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_45')
    elif 'conv_41' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_46')
    elif 'conv_42' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_47')
    elif 'conv_43' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_48')
    elif 'conv_44' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_49')
    elif 'conv_45' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_50')
    elif 'conv_46' in namespace:
        namespace = os.path.join(*namespace[:2], 'conv2d_51')
    else:
        namespace = os.path.join(*namespace[:2], 'conv2d')

    kernel_name = os.path.join(namespace, 'kernel:0')
    kernel = vars[kernel_name]

    if load_bias:
        bias_name = os.path.join(namespace, 'bias:0')
        bias = vars[bias_name]

        bias_shape = bias.shape.as_list()
        bias_params = np.prod(bias_shape)
        bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
        ptr += bias_params
        assign_ops.append(tf.compat.v1.assign(bias, bias_weights, validate_shape=True))

    kernel_shape = kernel.shape.as_list()
    kernel_params = np.prod(kernel_shape)

    [h, w, c, n] = kernel_shape
    kernel_weights = weights[ptr:ptr + kernel_params].reshape([n, c, h, w])
    # transpose to [h, w, c, n]
    kernel_weights = np.transpose(kernel_weights, (2, 3, 1, 0))

    ptr += kernel_params
    assign_ops.append(tf.compat.v1.assign(kernel, kernel_weights, validate_shape=True))

    return ptr


def _load_batch_norm(l, vars, ptr, weights, assign_ops):
    namespace = l.name.split('/')
    
    if 'downsample' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_1')
    elif 'conv_1' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_2')
    elif 'conv_2' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_3')
    elif 'downsample_1' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_4')
    elif 'conv_3' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_5')
    elif 'conv_4' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_6')
    elif 'conv_5' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_7')
    elif 'conv_6' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_8')
    elif 'downsample_2' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_9')
    elif 'conv_7' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_10')
    elif 'conv_8' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_11')
    elif 'conv_9' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_12')
    elif 'conv_10' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_13')
    elif 'conv_11' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_14')
    elif 'conv_12' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_15')
    elif 'conv_13' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_16')
    elif 'conv_14' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_17')
    elif 'conv_15' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_18')
    elif 'conv_16' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_19')
    elif 'conv_17' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_20')
    elif 'conv_18' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_21')
    elif 'conv_19' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_22')
    elif 'conv_20' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_23')
    elif 'conv_21' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_24')
    elif 'conv_22' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_25')
    elif 'downsample_3' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_26')
    elif 'conv_23' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_27')
    elif 'conv_24' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_28')
    elif 'conv_25' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_29')
    elif 'conv_26' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_30')
    elif 'conv_27' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_31')
    elif 'conv_28' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_32')
    elif 'conv_29' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_33')
    elif 'conv_30' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_34')
    elif 'conv_31' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_35')
    elif 'conv_32' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_36')
    elif 'conv_33' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_37')
    elif 'conv_34' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_38')
    elif 'conv_35' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_39')
    elif 'conv_36' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_40')
    elif 'conv_37' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_41')
    elif 'conv_38' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_42')
    elif 'downsample_4' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_43')
    elif 'conv_39' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_44')
    elif 'conv_40' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_45')
    elif 'conv_41' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_46')
    elif 'conv_42' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_47')
    elif 'conv_43' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_48')
    elif 'conv_44' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_49')
    elif 'conv_45' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_50')
    elif 'conv_46' in namespace:
        namespace = os.path.join(*namespace[:2], 'batch_normalization_51')
        
    else:    
        namespace = os.path.join(*namespace[:2], 'batch_normalization')

    gamma = os.path.join(namespace, 'gamma:0')
    beta = os.path.join(namespace, 'beta:0')
    moving_mean = os.path.join(namespace, 'moving_mean:0')
    moving_variance = os.path.join(namespace, 'moving_variance:0')

    gamma = vars[gamma]
    beta = vars[beta]
    moving_mean = vars[moving_mean]
    moving_variance = vars[moving_variance]

    for var in [beta, gamma, moving_mean, moving_variance]:
        shape = var.shape.as_list()
        num_params = np.prod(shape)
        var_weights = weights[ptr:ptr + num_params].reshape(shape)
        ptr += num_params
        assign_ops.append(tf.compat.v1.assign(var, var_weights, validate_shape=True))

    return ptr

# import os

# import numpy as np
# import tensorflow as tf

# def darknet53(model_builder, training, trainable):
#     mb = model_builder

#     mb.make_darknet_conv_layer(32, 3, training, trainable)  # 0

#     # Downsample (factor 2)
#     mb.make_darknet_downsample_layer(64, 3, training, trainable)  # 1

#     mb.make_darknet_residual_block(32, training, trainable)  # 2 - 4

#     # Downsample (factor 4)
#     mb.make_darknet_downsample_layer(128, 3, training, trainable)  # 5

#     for i in range(2):
#         mb.make_darknet_residual_block(64, training, trainable)  # 6 - 11

#     # Downsample (factor 8)
#     mb.make_darknet_downsample_layer(256, 3, training, trainable)  # 12

#     for i in range(8):
#         mb.make_darknet_residual_block(128, training, trainable)  # 13 - 36

#     # Downsample (factor 16)
#     mb.make_darknet_downsample_layer(512, 3, training, trainable)  # 37

#     for i in range(8):
#         mb.make_darknet_residual_block(256, training, trainable)  # 38 - 61

#     # Downsample (factor 32)
#     mb.make_darknet_downsample_layer(1024, 3, training, trainable)  # 62

#     for i in range(4):
#         mb.make_darknet_residual_block(512, training, trainable)  # 63 - 74


# def load_darknet_weights(net_layers, weightfile):
#     with open(weightfile, "rb") as f:
#         header = np.fromfile(f, dtype=np.int32, count=5)
#         weights = np.fromfile(f, dtype=np.float32)

#     ptr = 0
#     tmp = tf.compat.v1.global_variables()
#     vars = {var.name: var for var in tmp}

#     assign_ops = []

#     for l in net_layers:
#         if 'LeakyRelu' not in l.name:
#             continue

#         batch_norm = 'detection' not in l.name
#         load_bias = not batch_norm

#         if batch_norm:
#             ptr = _load_batch_norm(l, vars, ptr, weights, assign_ops)

#         ptr = _load_conv2d(l, vars, ptr, weights, assign_ops, load_bias)

#     assert ptr <= len(weights), f"Pointer {ptr} exceeded weights length {len(weights)}"
#     return assign_ops


# def _load_conv2d(l, vars, ptr, weights, assign_ops, load_bias):
#     namespace = l.name.split('/')
#     namespace = os.path.join(*namespace[:-1], 'conv2d')

#     kernel_name = os.path.join(namespace, 'kernel:0')
#     if kernel_name not in vars:
#         return ptr

#     kernel = vars[kernel_name]

#     if load_bias:
#         bias_name = os.path.join(namespace, 'bias:0')
#         if bias_name not in vars:
#             return ptr

#         bias = vars[bias_name]
#         bias_shape = bias.shape.as_list()
#         bias_params = np.prod(bias_shape)
#         bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
#         ptr += bias_params
#         assign_ops.append(tf.compat.v1.assign(bias, bias_weights, validate_shape=True))

#     kernel_shape = kernel.shape.as_list()
#     kernel_params = np.prod(kernel_shape)

#     [h, w, c, n] = kernel_shape
#     kernel_weights = weights[ptr:ptr + kernel_params].reshape([n, c, h, w])
#     kernel_weights = np.transpose(kernel_weights, (2, 3, 1, 0))

#     ptr += kernel_params
#     assign_ops.append(tf.compat.v1.assign(kernel, kernel_weights, validate_shape=True))

#     return ptr


# def _load_batch_norm(l, vars, ptr, weights, assign_ops):
#     namespace = l.name.split('/')
#     namespace = os.path.join(*namespace[:-1], 'batch_normalization')

#     gamma = vars.get(os.path.join(namespace, 'gamma:0'))
#     beta = vars.get(os.path.join(namespace, 'beta:0'))
#     moving_mean = vars.get(os.path.join(namespace, 'moving_mean:0'))
#     moving_variance = vars.get(os.path.join(namespace, 'moving_variance:0'))

#     if None in (gamma, beta, moving_mean, moving_variance):
#         return ptr

#     for var in [beta, gamma, moving_mean, moving_variance]:
#         shape = var.shape.as_list()
#         num_params = np.prod(shape)
#         var_weights = weights[ptr:ptr + num_params].reshape(shape)
#         ptr += num_params
#         assign_ops.append(tf.compat.v1.assign(var, var_weights, validate_shape=True))

#     return ptr
