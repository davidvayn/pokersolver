"""Run identity for MLX's process-start matrix-multiplication precision switch.

MLX can cache this setting on first use. Set it in the launch environment, not
mid-training. Unset/1 retains legacy identities; opting into full FP32 must
create a new run and cannot silently resume weights/optimizers from default math.
"""


def matmul_precision(environment):
    setting = environment.get('MLX_ENABLE_TF32')
    if setting not in (None,'0','1'):
        raise ValueError('MLX_ENABLE_TF32 must be unset, 0, or 1 at process launch')
    return 'full-fp32-v1' if setting == '0' else None
