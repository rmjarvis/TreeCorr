import numpy as np
import time

try:
    import kdcount
except:
    kdcount = None

try:
    import treecorr
except:
    treecorr = None

try:
    import halotools
    from halotools.mock_observables.pair_counters import npairs_3d as halotools_npairs
except:
    halotools = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


try:
    from Corrfunc.theory.DD import DD
except ImportError:
    Corrfunc = None

try:
    from sklearn.neighbors import KDTree
except ImportError:
    KDTree = None

def kdcount_timing(data1, data2, rbins, period):
    tree1 = kdcount.KDTree(data1).root
    tree2 = kdcount.KDTree(data2).root
    return tree1.count(tree2, r=rbins)


def halotools_timing(data1, data2, rbins, period):
    return halotools_npairs(data1, data2, rbins)

def treecorr_timing(data1, data2, rbins, period):
    cat1 = treecorr.Catalog(x=data1[:, 0], y=data1[:, 1], z=data1[:, 2])
    cat2 = treecorr.Catalog(x=data2[:, 0], y=data2[:, 1], z=data2[:, 2])
    nn = treecorr.NNCorrelation(nbins=len(rbins), min_sep=rbins[0], max_sep=rbins[-1])
    return nn.process(cat1, cat2, num_threads=1)

def cKDTree_timing(data1, data2, rbins, period):
    tree1 = cKDTree(data1)
    tree2 = cKDTree(data2)
    return tree1.count_neighbors(tree2, rbins)
    

def KDTree_timing(data1, data2, rbins, period):
    tree1 = KDTree(data1)
    return tree1.two_point_correlation(data2, rbins)

def corrfunc_timing(data1, data2, rbins, period):
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]
    nthreads = 1
    autocorr = 0
    return DD(autocorr, nthreads, rbins,
              x1, y1, z1,
              X2=x2, Y2=y2, Z2=z2, isa='avx512f',
              periodic=False)
    
def corrfunc_timing_avx(data1, data2, rbins, period):
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]
    nthreads = 1
    autocorr = 0
    return DD(autocorr, nthreads, rbins,
              x1, y1, z1,
              X2=x2, Y2=y2, Z2=z2, isa='avx',
              periodic=False)
    

def corrfunc_timing_sse(data1, data2, rbins, period):
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]
    nthreads = 1
    autocorr = 0
    return DD(autocorr, nthreads, rbins,
              x1, y1, z1,
              X2=x2, Y2=y2, Z2=z2, isa='sse42',
              periodic=False)


def corrfunc_timing_fallback(data1, data2, rbins, period):
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]
    
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]
    nthreads = 1
    autocorr = 0
    return DD(autocorr, nthreads, rbins,
            x1, y1, z1,
              X2=x2, Y2=y2, Z2=z2, isa='fallback',
              periodic=False)
    
    
def main():
    functions_to_test = []
    function_names = []
    function_versions = []
    #kdcount = cKDTree = KDTree = halotools = None
    if kdcount is not None:
        functions_to_test.append(kdcount_timing)
        function_names.append("kdcount")
        function_versions.append(kdcount.__version__)
        
    if cKDTree is not None:
        functions_to_test.append(cKDTree_timing)
        function_names.append("cKDTree")
        from scipy import __version__ as scipy_ver
        function_versions.append(scipy_ver)
        
    if KDTree is not None:
        functions_to_test.append(KDTree_timing)
        from sklearn import __version__ as sklearn_ver
        function_names.append("KDTree")
        function_versions.append(sklearn_ver)

    if treecorr is not None:
        functions_to_test.append(treecorr_timing)
        function_names.append("TreeCorr")
        function_versions.append(treecorr.__version__)

    if halotools is not None:
        functions_to_test.append(halotools_timing)
        function_names.append("halotools")
        function_versions.append(halotools.__version__)

    if Corrfunc is not None:
        functions_to_test.append(corrfunc_timing)
        function_names.append("Corrfunc(AVX512f)")
        function_versions.append(corrfunc.__version__)

        functions_to_test.append(corrfunc_timing_avx)
        function_names.append("Corrfunc(AVX)")
        function_versions.append(corrfunc.__version__)
    
        functions_to_test.append(corrfunc_timing_sse)
        function_names.append("Corrfunc(SSE42)")
        function_versions.append(corrfunc.__version__)
    
        functions_to_test.append(corrfunc_timing_fallback)
        function_names.append("Corrfunc(fallback)")
        function_versions.append(corrfunc.__version__)
        for fn, fn_version in zip(function_names, function_versions):
            print("## {} version = {}".format(fn, fn_version))

    npts = [1e4, 5e4, 1e5, 5e5, 1e6,]# 2e6, 5e6, 1e7]
    npts = [int(i) for i in npts]
    boxsize = 1.0
    rbins = np.logspace(-3, np.log10(0.05), 10) * boxsize

    print("## boxsize = {}. Bins are :".format(boxsize))
    for ll, uu in zip(rbins[0:-2], rbins[1:]):
        print("## {:0.5e} {:0.5e} ".format(ll, uu))

    print("## Npts          ", end="")
    for ifunc in function_names:
        print(" {:14s} ".format(ifunc), end="")
                
    print("")
        
    for n in npts:
        npts1 = n
        npts2 = n
        data1 = np.random.random(npts1*3).reshape(npts1, 3)
        data2 = np.random.random(npts2*3).reshape(npts2, 3)

        if data1.max() > boxsize or data2.max() > boxsize:
            msg = "Max data1 = {0} or max data2 = {1} must be less than "\
                    "boxsize = {2}".format(data1.max(), data2.max(), boxsize)
            raise ValueError(msg)

        print("{0:7d}".format(n), end="")
        for ii, func_name in enumerate(function_names):
            t0 = time.time()
            functions_to_test[ii](data1, data2, rbins, boxsize)
            t1 = time.time()
            print(" {0:14.3f}  ".format(t1-t0), end="", flush=True)

        print("")

if __name__ == '__main__':
    main()
