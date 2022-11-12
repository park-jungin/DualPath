import typing
from collections import Counter

from fvcore.nn import flop_count
from fvcore.nn.jit_handles import batchnorm_flop_jit, matmul_flop_jit, generic_activation_jit, get_shape

def generic_pooling_jit(name, multiplier=1):
    def pool_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
        # Inputs[0] contains the shape of the input.
        input_shape = get_shape(inputs[0])
        output_shape = get_shape(outputs[0])
        assert 2 <= len(input_shape) <= 5, input_shape
        flop = prod(input_shape) + prod(output_shape)  # summing all elements + denominating in each for output
        flop_counter = Counter({name: flop * multiplier})
        return flop_counter

    return lambda inputs, outputs: pool_jit(inputs, outputs)

def softmax_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    flop = prod(input_shape) * 2 + prod(output_shape) # exponentiating & summing inputs + denominating in each batch
    flop_counter = Counter({"softmax": flop})
    return flop_counter

def bmm_flop_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
    input1_shape = get_shape(inputs[0])
    input2_shape = get_shape(inputs[1])
    assert len(input1_shape) == len(input2_shape) == 3
    assert input1_shape[0] == input2_shape[0] and input1_shape[2] == input2_shape[1], [input1_shape, input2_shape]
    flop = prod(input1_shape) * input2_shape[-1]  # matmul of bnk * bkm -> bnm; flop = bnkm
    flop_counter = Counter({"bmm": flop})
    return flop_counter

def count_flops(model, inputs):
    flops, skips = flop_count(
            ForwardWrapper(model),
            inputs=(example_input,),
            supported_ops={
                "aten::batch_norm": batchnorm_flop_jit,
                "aten::group_norm": batchnorm_flop_jit,
                "aten::layer_norm": batchnorm_flop_jit,
                "aten::add": generic_activation_jit("add"),
                "aten::sub": generic_activation_jit("sub"),
                "aten::mul": generic_activation_jit("mul"),
                "aten::div": generic_activation_jit("div"),
                "aten::sqrt": generic_activation_jit("sqrt"),
                "aten::sigmoid": generic_activation_jit("sigmoid"),
                "aten::sigmoid_": generic_activation_jit("sigmoid_"),
                "aten::relu": generic_activation_jit("relu"),
                "aten::relu_": generic_activation_jit("relu_"),
                "aten::gelu": generic_activation_jit("gelu"),
                "aten::add_": generic_activation_jit("add_"),
                "aten::sub_": generic_activation_jit("sub_"),
                "aten::mul_": generic_activation_jit("mul_"),
                "aten::div_": generic_activation_jit("div_"),
                "aten::sqrt_": generic_activation_jit("sqrt_"),
                "aten::adaptive_avg_pool2d": generic_pooling_jit("adaptive_avg_pool2d"),
                "aten::adaptive_max_pool2d": generic_pooling_jit("adaptive_max_pool2d"),
                "aten::avg_pool2d": generic_pooling_jit("avg_pool2d"),
                "aten::max_pool2d": generic_pooling_jit("max_pool2d"),
                "aten::bmm": bmm_flop_jit,
                "aten::mean": generic_pooling_jit("mean"),
                "aten::var": generic_pooling_jit("var", multiplier=3),  # subtracting mean, exponentiate, summing
                "aten::var_mean": generic_pooling_jit("mean_var", multiplier=4),
                "aten::softmax": softmax_jit,
                "aten::dropout": generic_activation_jit("dropout"),
                "aten::frobenius_norm": generic_pooling_jit("frobenius_norm"),
            }
        )
    return flops