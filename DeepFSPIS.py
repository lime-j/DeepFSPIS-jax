from typing import Callable, Sequence, TypeVar, Tuple, Any
import functools
import einops
from flax import linen as nn
import jax.numpy as jnp
import jax
T = TypeVar('T')

def gradient_calc(image : jnp.array) -> jnp.array :
    r"""
    Calculate image gradients horizontally and vertically, then add them up.
    Args :
        images : jnp.array, input image in [N, H, W, C] shape
    Return :
        Edge maps, jnp.array
    """

    edg_x, edg_y = jnp.pad(jnp.abs(image[:,:-1,:,:] - image[:,1:,:,:]) , ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant'), \
                   jnp.pad(jnp.abs(image[:,:,:-1,:] - image[:,:,1:,:]) , ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant')
    return (edg_x + edg_y)

class GlobalAvgPool1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=-1, keepdims=True)

class GlobalAvgPool2D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=(-3, -2), keepdims=True)

class InstanceNorm2d(nn.Module) :
    r'''
    A Flax warp of PyTorch style InstanceNorm2d
    '''
    
    num_features : int
    eps : float = 1e-5
    momentum : float = 0.1
    affine : bool = False
    track_running_stats = False
    
    @nn.compact
    def __call__(self, x):
        ret = nn.LayerNorm(epsilon=self.eps, 
                            use_bias=self.affine,
                            use_scale=self.affine,
                            reduction_axes=(-3, -2))(x) #assuming data in N, H, W, C
        return ret

class GuiderResidualBlock(nn.Module):
    r'''
    ResidualBlock for Adjuster module. Equipped with InstanceNorm
    '''
    out_features : int = 32
    dilation : int = 1
    
    @nn.compact
    def __call__(self, ipt) :
        x = nn.Conv(self.out_features, 
                    kernel_size=(3, 3), 
                    use_bias=True, 
                    kernel_dilation=self.dilation,
                    name='conv_block.0')(ipt) # padding='SAME'

        x = InstanceNorm2d(self.out_features)(x)
        x = nn.PReLU(name='conv_block.2')(x)
        x = nn.Conv(self.out_features,
                    kernel_size=(1, 1), 
                    use_bias=True,
                    name='conv_block.3')(x)
        x = InstanceNorm2d(self.out_features)(x)
        x = nn.PReLU(name='conv_block.5')(x)
        return x + ipt

class GuiderConvBlock(nn.Module):
    out_features : int
    kernel_size : Tuple[int] = (3, 3)
    padding : Any = 'SAME'
    stride : Tuple[int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_features, 
                    kernel_size = self.kernel_size,
                    use_bias = True,
                    padding = self.padding,
                    strides = self.stride,
                    name = 'conv')(x)
        x = InstanceNorm2d(self.out_features)(x)
        x = nn.PReLU(name='leakyReLU')(x)
        return x

class Blender(nn.Module) :
    out_features : int

    @nn.compact
    def __call__(self, state, siamese_state, lam):
        x = jnp.concatenate((state, lam * siamese_state), axis=-1)
        x = nn.Conv(self.out_features, kernel_size = (1, 1), name='fuser.0')(x)
        x = InstanceNorm2d(self.out_features)(x)
        x = nn.PReLU(name='fuser.2')(x)
        return x

class PALayer(nn.Module):
    out_features : int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.out_features // 8, kernel_size=(1, 1), use_bias=True, name='pa0')(x)
        y = nn.relu(y)
        y = nn.Conv(1, kernel_size=(1, 1), use_bias = True, name='pa1.0')(y)
        y = nn.sigmoid(y)
        return x * y

class CALayer(nn.Module):
    out_features : int

    @nn.compact
    def __call__(self, x):
        y = GlobalAvgPool2D()(x)
        y = nn.Conv(self.out_features // 8, kernel_size=(1, 1), use_bias=True, name='ca0')(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_features, kernel_size=(1, 1), use_bias=True, name='ca1')(y)
        y = nn.sigmoid(y)
        return x * y 

class SmootherBasicBlock(nn.Module):
    out_features : int
    stride : int = 1

    @nn.compact
    def __call__(self, x): 
        x = nn.Sequential([
            nn.Conv(self.out_features, kernel_size=(3, 3), 
                    strides=(self.stride, self.stride),
                    padding='SAME', name='conv_block.0'
            ),
            PALayer(self.out_features, name='conv_block.2'),
            CALayer(self.out_features, name='conv_block.3'),
            nn.PReLU(name='conv_block.4')
        ], name='conv_block')(x)
        return x

class SmootherResNetBlock(nn.Module):
    out_features : int
    dilation : int = 1
    use_bias : bool = True

    @nn.compact
    def __call__(self, x):
        y = nn.Sequential([
            nn.Conv(self.out_features, kernel_size=(3, 3),
                    kernel_dilation=self.dilation, 
                    use_bias=self.use_bias, name='conv_block.0'),
            PALayer(self.out_features, name='conv_block.1'),
            CALayer(self.out_features, name='conv_block.2'),
            nn.PReLU(name='conv_block.4'),
            nn.Conv(self.out_features, kernel_size=(3, 3),
                    kernel_dilation=self.dilation, 
                    use_bias = self.use_bias, name='conv_block.5'),
            PALayer(self.out_features, name='conv_block.6'),
            CALayer(self.out_features, name='conv_block.7'),
            nn.PReLU(name='conv_block.9')
        ], name='conv_block')(x)
        return y + x

class _UNetBlock(nn.Module) :
    out_features : int = 32
    kernel_size : int = 3
    padding : str = 'SAME'
    stride : int = 1
    norm : bool = False
    bias : bool = True
    dilation : int = 1


class UNetUpBlock(_UNetBlock):

    @nn.compact
    def __call__(self, x1, x2):
        n, h, w, c = x1.shape
        # jax performs align_corner=True by default,
        # see https://github.com/google/flax/discussions/2211
        x1 = jax.image.resize(x1, (n, h * 2, w * 2, c), "bilinear") 
        x = jnp.concatenate((x2, x1), axis=-1)
        x = nn.Sequential([
            nn.Conv(self.out_features, kernel_size=(self.kernel_size, self.kernel_size),
                    padding = self.padding, strides=self.stride, use_bias=self.bias,
                    kernel_dilation=self.dilation, name='model.0'),
            PALayer(self.out_features, name='model.1'),
            CALayer(self.out_features, name='model.2')
        ], name='model')(x)
        x = nn.PReLU(name='relu')(x)
        return x

class UNetDownBlock(_UNetBlock):
    @nn.compact
    def __call__(self, x):
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Sequential([
            nn.Conv(self.out_features, kernel_size=(self.kernel_size, self.kernel_size),
                    padding = self.padding, strides=self.stride, use_bias=self.bias,
                    kernel_dilation=self.dilation, name='model.1'),
            PALayer(self.out_features, name='model.2'),
            CALayer(self.out_features, name='model.3')
        ], name='model')(x)
        x = nn.PReLU(name='relu')(x)
        return x

class UNet(nn.Module) :
    out_features : int = 32

    def setup(self) -> None:
        self.padder_size = 2 ** 4

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        mod_pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - W % self.padder_size) % self.padder_size 
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = jnp.pad(x, ((0, 0), (0, mod_pad_h), (0, mod_pad_w), (0, 0)), mode='constant')
        d1 = UNetDownBlock(self.out_features, name='down1')(x)
        d2 = UNetDownBlock(self.out_features, name='down2')(d1)
        d3 = UNetDownBlock(self.out_features, name='down3')(d2)
        d4 = UNetDownBlock(self.out_features, name='down4')(d3)
        
        u1 = UNetUpBlock(self.out_features, name='up1')(d4, d3)
        u2 = UNetUpBlock(self.out_features, name='up2')(u1, d2)
        u3 = UNetUpBlock(self.out_features, name='up3')(u2, d1)
        u4 = UNetUpBlock(self.out_features, name='up4')(u3, x)
    
        return jax.lax.dynamic_slice(u4, (0, 0, 0, 0), (B, H, W, C))
    
class Adjuster(nn.Module) :


    @nn.compact
    def __call__(self, x, lam):
        
        gradient = jnp.clip(jnp.max(gradient_calc(x), axis=-1), 0, 1)
        # n, h, w, c = x.shape
        x2 = nn.Sequential([GuiderConvBlock(32, name='to_feature.0')] + [
            GuiderResidualBlock(32, 1, name='to_feature.%d'%(_ + 1)) for _ in range(4)],
        )(x)

        state = nn.Sequential(
            [GuiderResidualBlock(32, 2 ** (i // 2), name='body.%d'%i) for i in range(8)] + \
            [GuiderResidualBlock(32, 1, name='body.8')], 
        )(x2)
        
        siamese_state = nn.Sequential(
            [GuiderResidualBlock(32, 1, name='siamese_body.0')], name='siamese_body'
        )(x2)

        out_feat = Blender(32, name='fuser')(state, siamese_state, lam)

        mask = nn.Sequential([
            nn.Conv(16, kernel_size=(3, 3), name='to_edge.0'),
            InstanceNorm2d(16),
            nn.PReLU(name='to_edge.2'),
            nn.Conv(1, kernel_size=(3, 3), name='to_edge.3'),
            nn.sigmoid],
            name='to_edge'
        )(out_feat)

        return jnp.sqrt(jnp.expand_dims(gradient, axis=-1) * mask)

 
class Smoother(nn.Module):
    @nn.compact
    def __call__(self, x, edge):
        in_stage0 = nn.Sequential([
            SmootherBasicBlock(32, name='backbone0.0'),
            SmootherBasicBlock(32, name='backbone0.1'),
            SmootherResNetBlock(32, 1, name='backbone0.2')],
            name='backbone0'
        )(jnp.concatenate((x, edge), axis=-1))
        body_stage0 = UNet(32, name="body0.0")(in_stage0)

        in_stage1 = nn.max_pool(
            nn.Sequential([
                    SmootherResNetBlock(32, 1, name='backbone1.0'),
                    SmootherResNetBlock(32, 1, name='backbone1.1')],
                    name='backbone1'
                )(body_stage0), 
            (2, 2), strides=(2, 2)
        )
        
        body_stage1 = UNet(32, name="body1.0")(in_stage1)

        in_stage2 = nn.max_pool(
            nn.Sequential([
                SmootherResNetBlock(32, 1, name='backbone2.0'),
                SmootherResNetBlock(32, 1, name='backbone2.1')],
                name='backbone2'
            )(body_stage1),
            (2, 2), strides=(2, 2)
        )

        body_stage2 = UNet(32, name='body2.0')(in_stage2)

        out_stage2 = nn.Sequential([
            SmootherBasicBlock(32, 1, name='outy.0'),
            nn.Conv(3, (3, 3), 1, name='outy.1')
        ], name='outy')(
                UNetUpBlock(32, name='up21')(
                UNetUpBlock(32, name='up22')(
                    body_stage2, body_stage1
                ), body_stage0
            )
        )

        return out_stage2


class DeepFSPIS(nn.Module):
    
    @nn.compact
    def __call__(self, x, lam):
        edge = Adjuster(name='adjuster')(x, lam)
        return edge, Smoother(name='smoother')(x, edge)

import collections
import flax

def align(jax_params, pytorch_params, parent=""):

    if type(jax_params) == dict or type(jax_params) == flax.core.frozen_dict.FrozenDict:
        for jax_key in jax_params.keys():
            jax_params[jax_key] = align(jax_params[jax_key], pytorch_params, parent=parent + str(jax_key) + '.')
            # print(jax_key) 
    else :
        module, parent = parent.split('.')[0], ".".join(parent.split('.')[1:-1])
        if 'fuser' in parent :
            parent = parent[6:]
            print('stripped fuser from parent : %s' % parent)
        if ('kernel' in parent):
            pytorch_parent = parent.replace('kernel', 'weight')
            to_align = pytorch_params[module][pytorch_parent].detach().cpu().numpy()
            to_align = jnp.transpose(to_align, (2, 3, 1, 0))
            jax_params = to_align
        elif 'negative_slope' in parent : 
            pytorch_parent = parent.replace('negative_slope', 'weight')
            to_align = pytorch_params[module][pytorch_parent].detach().cpu().numpy()
            jax_params = to_align[0]
        else : #if ('bias' in parent) :
            pytorch_parent = parent #parent.replace('bias', '')
            to_align = pytorch_params[module][pytorch_parent].detach().cpu().numpy()
            jax_params = to_align 
    return jax_params

if __name__ == '__main__':
    import torch, cv2
    import numpy as np
    from PIL import Image
    model = DeepFSPIS()
    def init_model():
        return model.init(
            jax.random.PRNGKey(0),
            # Discard the "num_local_devices" dimension for initialization.
            jnp.ones((1, 128, 128, 3), jnp.float32), jnp.ones((1, 1), jnp.float32))
    adjuster_weight_path="./adjuster.pth" 
    smoother_weight_path="./smoother.pth"
    variables = jax.jit(init_model, backend='cpu')()
    pytorch_params = {"adjuster" : torch.load(adjuster_weight_path, map_location='cpu')['icnn'], 
                      "smoother" : torch.load(smoother_weight_path, map_location='cpu')['model']
    }
    params = variables["params"].unfreeze()
    params = align(params, pytorch_params)
    params = flax.core.frozen_dict.freeze(params)
    x = cv2.imread("./input.png")
    x = jnp.expand_dims(x, axis=0)
    x = x[...,::-1]
    print(x.shape)
    x = jnp.asarray(x, jnp.float32) / 255.
    # x = jnp.ones((1, 128, 128, 3), jnp.float32)
    lam = jnp.ones((1, 1), jnp.float32) * 0.2
    
    edge_out, out = model.apply(dict(params=params), x, lam)
    print(out.shape, edge_out.shape)
    cv2.imwrite("edge_out.png", (np.asarray(edge_out)[0, ...] * 255).clip(0, 255).astype(np.uint8))

    cv2.imwrite("ret.png", (np.asarray(out)[0, ..., ::-1] * 255).clip(0, 255).astype(np.uint8))
    # print(jax.tree_util.tree_flatten(params))
    
