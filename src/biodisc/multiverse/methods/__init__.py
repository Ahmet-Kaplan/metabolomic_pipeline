"""Method package — eagerly imports every method module so each can self-register.

Adding a new method:
  1. Drop a file at ``methods/<category>/<name>.py``
  2. Implement your function and call ``register(MethodSpec(...))`` at module bottom
  3. Add an import line below — that's the only edit needed in this file
"""

from biodisc.multiverse.methods.drift import (
    qc_rsc,  # noqa: F401
    serrf,  # noqa: F401
)
from biodisc.multiverse.methods.impute import (
    halfmin,  # noqa: F401
    missforest,  # noqa: F401
    ns_knn,  # noqa: F401
    qrilc,  # noqa: F401
)
from biodisc.multiverse.methods.multivariate import (
    plsda,  # noqa: F401
    rf,  # noqa: F401
    splsda,  # noqa: F401
)
from biodisc.multiverse.methods.normalise import (
    median,  # noqa: F401
    mstus,  # noqa: F401
    pqn,  # noqa: F401
)
from biodisc.multiverse.methods.scale import (
    pareto,  # noqa: F401
)
from biodisc.multiverse.methods.transform import (
    glog,  # noqa: F401
    log2,  # noqa: F401
)

__all__: list[str] = []
