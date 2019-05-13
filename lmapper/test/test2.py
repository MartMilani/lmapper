import lmapper as lm
from lmapper.filter import Projection
from lmapper.cover import UniformCover
from lmapper.cluster import Linkage
from lmapper.cutoff import FirstGap
from lmapper.datasets import cat


def main():
    data = cat()
    filter = Projection(ax=0)
    cover = UniformCover(nintervals=15,
                         overlap=0.4)
    cutoff = FirstGap(0.05)
    cluster = Linkage(method='single',
                      metric="euclidean",
                      cutoff=cutoff)
    mapper = lm.Mapper(data=data,
                       filter=filter,
                       cover=cover,
                       cluster=cluster)
    mapper.fit()
    mapper.plot()

	cluster = Linkage(method='single',
					  metric="euclidean",
                      cutoff=cutoff)
	mapper.set_params(cluster=cluster)
	mapper.fit()
	mapper.plot()


if __name__ == "__main__":
    main()
