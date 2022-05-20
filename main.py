from time import perf_counter

import click
import torch
from loguru import logger
from tqdm import trange

from efficientnet import efficientnet_b0


def timeit(f):

    def timed(*args, **kwargs):
        start = perf_counter()
        output = f(*args, **kwargs)
        end = perf_counter()
        logger.info("{} took {:.2f} seconds", f.__name__, end - start)
        return output

    return timed

@click.command()
@click.option('-n', '--num-iterations', default=5, help='Number of iterations')
@click.option('-b', '--batch-size', default=32, help='Batch size')
@click.option('-s', '--image-size', default=224, help='Image size')
def main(num_iterations, batch_size, image_size):
    x = torch.randn(batch_size, 3, image_size, image_size)
    net = efficientnet_b0()

    @timeit
    def cpu_forward():
        for _ in trange(num_iterations):
            net(x)

    cpu_forward()

    x = x.to('mps')
    net.to('mps')

    @timeit
    def mps_forward():
        for _ in trange(num_iterations):
            net(x)

    mps_forward()


if __name__ == '__main__':
    main()
