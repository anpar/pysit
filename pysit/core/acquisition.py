import copy

import numpy as np

from shot import *

from receivers import *
from sources import *

from pysit.util.parallel import ParallelWrapShotNull

__all__ = ['equispaced_acquisition', 'equispaced_acquisition_single_free_src']

def equispaced_acquisition(mesh, wavelet,
                           sources=1,
                           receivers='max',
                           source_depth=None,
                           source_kwargs={},
                           receiver_depth=None,
                           receiver_kwargs={},
                           parallel_shot_wrap=ParallelWrapShotNull()
                           ):

    m = mesh
    d = mesh.domain

    xmin = d.x.lbound
    xmax = d.x.rbound

    zmin = d.z.lbound
    zmax = d.z.rbound

    if m.dim == 3:
        ymin = d.y.lbound
        ymax = d.y.rbound


    if source_depth is None:
        source_depth = zmin

    if receiver_depth is None:
        receiver_depth = zmin

    shots = list()

    max_sources = m.x.n

    if m.dim == 2:
        if receivers == 'max':
            receivers = m.x.n
        if sources == 'max':
            sources = m.x.n

        if receivers > m.x.n:
            raise ValueError('Number of receivers exceeds mesh nodes.')
        if sources > m.x.n:
            raise ValueError('Number of sources exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers)
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, receiver_depth), **receiver_kwargs) for x in xpos])

        local_sources = sources / parallel_shot_wrap.size, 1

    if m.dim == 3:

        if receivers == 'max':
            receivers = (m.x.n, m.y.n) # x, y
        if sources == 'max':
            sources = (m.x.n, m.y.n) # x, y

        if receivers[0] > m.x.n or receivers[1] > m.y.n:
            raise ValueError('Number of receivers exceeds mesh nodes.')
        if sources[0] > m.x.n or sources[1] > m.y.n:
            raise ValueError('Number of sources exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers[0])
        ypos = np.linspace(ymin, ymax, receivers[1])
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, y, receiver_depth), **receiver_kwargs) for x in xpos for y in ypos])

        local_sources = sources[0] / parallel_shot_wrap.size, sources[1] / parallel_shot_wrap.size

    for i in xrange(local_sources[0]):
        for j in xrange(local_sources[1]):

            idx = i + local_sources[0]*parallel_shot_wrap.rank
            jdx = j + local_sources[1]*parallel_shot_wrap.rank

            if m.dim == 2:
                srcpos = (xmin + (xmax-xmin)*(idx+1.0)/(sources+1.0), source_depth)
            elif m.dim == 3:
                srcpos = (xmin + (xmax-xmin)*(idx+1.0)/(sources[0]+1.0), ymin + (ymax-ymin)*(jdx+1.0)/(sources[1]+1.0), source_depth)

            # Define source location and type
            source = PointSource(m, srcpos, wavelet, **source_kwargs)

            # Define set of receivers
            receivers = copy.deepcopy(receiversbase)

            # Create and store the shot
            shot = Shot(source, receivers)
            shots.append(shot)

    return shots

def equispaced_acquisition_single_free_src(mesh, wavelet,
                                           receivers='max',
                                           source_pos=None,
                                           source_kwargs={},
                                           receiver_depth=None,
                                           receiver_kwargs={},
                                           ):

    m = mesh
    d = mesh.domain

    xmin = d.x.lbound
    xmax = d.x.rbound

    zmin = d.z.lbound
    zmax = d.z.rbound

    if m.dim == 3:
        ymin = d.y.lbound
        ymax = d.y.rbound

    if source_pos == None:
        raise ValueError('Source position not defined')

    if receiver_depth is None:
        receiver_depth = zmin

    shots = list()

    if m.dim == 2:
        if receivers == 'max':
            receivers = m.x.n

        if receivers > m.x.n:
            raise ValueError('Number of receivers exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers)
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, receiver_depth), **receiver_kwargs) for x in xpos])

    if m.dim == 3:
        if receivers == 'max':
            receivers = (m.x.n, m.y.n) # x, y

        if receivers[0] > m.x.n or receivers[1] > m.y.n:
            raise ValueError('Number of receivers exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers[0])
        ypos = np.linspace(ymin, ymax, receivers[1])
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, y, receiver_depth), **receiver_kwargs) for x in xpos for y in ypos])


    if m.dim == 2:
        x, z = source_pos

        if x < xmin or x > xmax or z < zmin or z > zmax:
            raise ValueError('Source position not in the domain.')
    elif m.dim == 3:
        x, z, y = source_pos

        if x < xmin or x > xmax or z < zmin or z > zmax or y < ymin or y > ymax:
            raise ValueError('Source position not in the domain.')

    # Define source location and type
    source = PointSource(m, source_pos, wavelet, **source_kwargs)

    # Define set of receivers
    receivers = copy.deepcopy(receiversbase)

    # Create and store the shot
    shot = Shot(source, receivers)
    shots.append(shot)

    return shots
